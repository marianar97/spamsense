from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from zoneinfo import ZoneInfo

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit import api, rtc

load_dotenv()


@dataclass
class Userdata:
    cal: Calendar


logger = logging.getLogger("telephony_agent")


# Add this function definition anywhere
async def hangup_call():
    ctx = get_job_context()
    if ctx is None:
        # Not running in a job context
        return
    
    await ctx.api.room.delete_room(
        api.DeleteRoomRequest(
            room=ctx.room.name,
        )
    )

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
                - Call the end_call function to hang up the call

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


    # to hang up the call as part of a function call
    @function_tool
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call"""
        # let the agent finish speaking
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()

        await hangup_call()

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    timezone = "utc"

    session = AgentSession[Userdata](
        userdata=Userdata(),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o", parallel_tool_calls=False, temperature=0.45),
        tts=cartesia.TTS(voice="39b376fc-488e-4d0c-8b37-e00b72059fdd", speed="fast"),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        max_tool_steps=1,
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