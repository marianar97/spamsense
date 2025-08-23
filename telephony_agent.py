from __future__ import annotations

import logging
import os
import sys
import asyncio
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import json
from datetime import datetime
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from typing import Any
import urllib.request
import urllib.error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    get_job_context,
    function_tool,
)
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit import api, rtc

load_dotenv()

logger = logging.getLogger("telephony_agent")


def convert_transcript_to_messages(transcript: dict) -> list[dict[str, str]]:
    """
    Convert a transcript of the shape {"items": [{"role": str, "content": list[str]|str, ...}, ...]}
    into a simplified list of message dicts: {"role": "agent"|"user", "response": str}.

    - Map role "assistant" -> "agent"; keep "user" as-is; skip other roles.
    - Flatten content arrays into a single string; ignore non-string entries.
    - Skip items that have empty or missing textual content.
    """
    items = transcript.get("items", [])
    messages: list[dict[str, str]] = []
    
    for item in items:
        role = item.get("role")
        role = "agent" if role == "assistant" else "user"
        content = "".join(item.get("content")).strip()
        
        # Skip empty content
        if not content:
            continue
            
        confidence = item.get("transcript_confidence", 0.80)
        timestamp = datetime.now().isoformat() + "Z"
        
        messages.append({
            "role": role, 
            "response": content, 
            "confidence": confidence, 
            "timestamp": timestamp
        })

    return messages

def post_to_convex(call_id: str, transcript):
    """
    Send transcript data to the SpamSense API publish transcript endpoint.
    
    Args:
        call_id: The call ID from the previously created call
        transcript: List of message dictionaries with role, response, confidence, timestamp
    """
    base_url = os.getenv("SPAMSENSE_API_BASE_URL", "https://spamsense.vercel.app")
    transcript_url = f"{base_url}/api/transcripts"
    
    # Create full transcript string from messages
    full_transcript_parts = []
    for msg in transcript:
        role = msg.get("role", "user")
        response = msg.get("response", "")
        full_transcript_parts.append(f"{role}: {response}")
    
    full_transcript = "\\n".join(full_transcript_parts)
    
    # Calculate duration (mock data - could be calculated from timestamps)
    duration = 420  # 7 minutes default
    
    payload = {
        "callId": call_id,
        "transcript": transcript,
        "fullTranscript": full_transcript,
        "language": "en",
        "duration": duration,
        "createdAt": datetime.now().isoformat()
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        transcript_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            try:
                result = json.loads(raw.decode("utf-8"))
            except Exception:
                result = {"status": getattr(resp, "status", None), "body": raw.decode("utf-8")}
            
            if isinstance(result, dict) and "transcriptId" in result:
                logger.info("Published transcript with transcriptId=%s", result["transcriptId"])
            else:
                logger.warning("Publish transcript response: %s", result)
            return result
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        logger.error("Failed to publish transcript (HTTP %s): %s", getattr(e, "code", "?"), body)
        raise
    except urllib.error.URLError as e:
        logger.error("Failed to publish transcript (URL error): %s", getattr(e, "reason", e))
        raise 


def create_call_id():
    """
    Step 1: Create a call record in SpamSense API and return its response.

    This function will be extended in subsequent steps to also publish the transcript.

    Args:
        call_id: Session identifier (used for notes/traceability).
        timestamp: ISO 8601 string or datetime for when the transcript was captured.
        transcript: Simplified transcript messages (unused in this step).

    Returns:
        Parsed JSON response from the create-call endpoint; contains `callId` on success.
    """
    base_url = os.getenv("SPAMSENSE_API_BASE_URL", "https://spamsense.vercel.app")
    create_call_url = f"{base_url}/api/calls"

    ts_str = datetime.now().isoformat()
    phone_number = os.getenv("DEFAULT_CALLER_PHONE", "+1-555-TEST-001")
    payload: dict[str, Any] = {
        "phoneNumber": phone_number,
        "type": "personal",
        "status": "allowed",
        "duration": 420,
        "timestamp": ts_str,
        "isSpam": False,
        "confidence": 5,
        "location": "Unknown",
        "carrierInfo": "Unknown",
        "action": "allow",
        "notes": f"",
        "hasTranscript": False,
        "hasSummary": False,
        "transcriptStatus": "pending",
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        create_call_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            try:
                result = json.loads(raw.decode("utf-8"))
            except Exception:
                result = {"status": getattr(resp, "status", None), "body": raw.decode("utf-8")}

            # Log and return; caller can extract `callId` for next step
            if isinstance(result, dict) and "callId" in result:
                logger.info("Created call with callId=%s", result["callId"])
                return result["callId"]
            else:
                logger.warning("Create call response missing callId: %s", result)
            return result
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        logger.error("Failed to create call (HTTP %s): %s", getattr(e, "code", "?"), body)
        raise
    except urllib.error.URLError as e:
        logger.error("Failed to create call (URL error): %s", getattr(e, "reason", e))
        raise

class TelephonyAgent(Agent):
    def __init__(self, *, timezone: str) -> None:
        self.tz = ZoneInfo(timezone)
        instructions = """
                # Phone Personal Assistant Prompt for Mariana
                ## Core Identity
                You are Mariana's professional phone personal assistant. Your voice should be warm, polite, and efficient. You represent Mariana professionally while maintaining a friendly demeanor.

                ## Primary Objective
                Listen to incoming calls and gather only the essential context needed to properly triage the call for later AI processing. Be concise and purposeful in your questioning.

                ## Key Behaviors

                ### Opening Protocol
                - Answer with: "Hello, this is Mariana's assistant. How may I help you today?"
                - Listen actively to the caller's initial request or reason for calling

                ### Information Gathering Strategy
                Ask only the **minimum necessary questions** to determine:

                1. **Caller Identity** (if not provided)
                - "May I have your name please?"
                - "What company are you with?" (if business-related)

                2. **Call Purpose** (if unclear)
                - "What is this regarding?"
                - "Is this a business or personal matter?"

                3. **Urgency Level** (if ambiguous)
                - "Is this time-sensitive?"
                - "When would you need to hear back?"

                4. **Contact Preference** (if callback needed)
                - "What's the best number to reach you?"
                - "Is there a preferred time to call back?"

                ### What NOT to Do
                - Don't ask for excessive details about the matter itself
                - Don't try to solve the caller's problem
                - Don't make commitments on Mariana's behalf
                - Don't ask redundant questions if information is already clear
                - Don't keep callers on hold unnecessarily

                ### Triage Categories to Identify
                Listen for cues that indicate:
                - **Urgent/Emergency**: Immediate attention required
                - **Business**: Work-related matters, meetings, deals
                - **Personal**: Friends, family, personal services
                - **Sales/Marketing**: Cold calls, promotional offers
                - **Administrative**: Appointments, confirmations, routine matters

                ### Closing Protocol
                - Summarize briefly: "So I have [caller's name] from [company/context] calling about [brief topic], and you can be reached at [number]."
                - Set expectations: "I'll make sure Mariana gets this message" or "Mariana will get back to you by [timeframe]"
                - Thank the caller professionally

                ### Voice and Tone Guidelines
                - Speak clearly and at a moderate pace
                - Use a professional but approachable tone
                - Show active listening with appropriate verbal cues ("I understand," "Okay")
                - Remain calm and helpful even with difficult callers
                - Maintain confidentiality about Mariana's schedule or personal matters

                ### Special Situations
                - **Known Important Contacts**: Expedite appropriately - "Let me see if Mariana is available"
                - **Unknown Callers**: Be courteous but protective of Mariana's time
                - **Potential Emergencies**: Ask clarifying questions and escalate immediately if needed
                - **Repeat Callers**: Acknowledge if you remember previous conversations

                ## Success Metrics
                Your effectiveness is measured by:
                - Gathering sufficient context for proper call triage
                - Minimizing unnecessary questions
                - Maintaining professional representation of Mariana
                - Ensuring no important calls are mishandled or delayed"
                """
        
        super().__init__(
            instructions=instructions
        )



async def entrypoint(ctx: JobContext):
    async def write_transcript():
        try:
            # Create call record and get call ID
            call_id = create_call_id()
            if not call_id:
                logger.error("Failed to create call - cannot publish transcript")
                return
            
            # Get transcript and convert to messages format
            transcript = session.history.to_dict()
            messages = convert_transcript_to_messages(transcript)
            
            # Save messages to temp file for debugging
            current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            messages_file = f"/tmp/messages_{ctx.room.name}_{current_date}.json"
            
            with open(messages_file, 'w') as f:
                json.dump(messages, f, indent=2)
            
            # Publish transcript to API
            post_to_convex(call_id, messages)
            
            logger.info(f"Transcript for {ctx.room.name} saved to {messages_file}")
            
        except Exception as e:
            logger.error(f"Error in write_transcript: {e}")
            raise

    ctx.add_shutdown_callback(write_transcript)
    await ctx.connect()

    timezone = "utc"

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(agent=TelephonyAgent(timezone=timezone), room=ctx.room)


if __name__ == "__main__":
    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Run the agent with the name that matches your dispatch rule
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint,
        agent_name="telephony_agent"  # This must match your dispatch rule
    ))