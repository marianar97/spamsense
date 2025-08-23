import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class TranscriptRequest(BaseModel):
    transcript: str = Field(..., description="Full transcript text of the call")


AllowedIntent = Literal[
    "technical_support",
    "billing",
    "sales",
    "cancellation",
    "appointment",
    "greeting",
    "complaint",
    "feedback",
    "account_access",
    "other",
]


class IntentResult(BaseModel):
    caller_name: Optional[str] = Field(None, description="Name of the caller")
    summary: Optional[str] = Field(None, description="One-sentence summary of caller's need in natural language"
    )
    urgency: Optional[Literal["low", "medium", "high"]] = Field(None, description="Urgency level of the caller's need")
    category: Optional[Literal["personal", "leads", "spam"]] = Field(None, description="Category of the caller's need")

app = FastAPI(title="SpanSense Intent API", version="0.1.0")


@app.post("/intent", response_model=IntentResult)
async def classify_intent(req: TranscriptRequest) -> IntentResult:  # type: ignore[override]
    """Classify the call transcript into a structured intent using OpenAI.

    This uses OpenAI structured output validated by Pydantic. The OpenAI API key
    must be provided via the OPENAI_API_KEY environment variable. Optionally, set
    OPENAI_MODEL (defaults to gpt-4o-mini).
    """
    # Lazy import to avoid hard dependency at import-time (enables smoke tests without OpenAI installed)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_instructions = (
    "You are a call transcript analyzer. Extract key information from phone call transcripts and return structured data.\n\n"
    
    "Required output fields:\n"
    "• caller_name: Name mentioned by or for the caller (null if not provided)\n"
    "• summary: One clear sentence describing what the caller wants or why they called\n"
    "• urgency: 'low' (routine/informational), 'medium' (time-sensitive), or 'high' (emergency/critical)\n"
    "• category: 'personal' (friends/family), 'leads' (business opportunity), or 'spam' (unsolicited sales/marketing)\n"
    "• confidence: Float 0.0-1.0 representing certainty in your analysis\n\n"
    
    "Analysis approach:\n"
    "- Use only information explicitly stated or clearly implied in the transcript\n"
    "- When in doubt, choose lower urgency and confidence scores\n"
    "- For mixed-topic calls, focus on the primary/main purpose\n"
    "- Consider emotional indicators (frustrated, rushed, casual tone) for urgency\n"
    "- Mark as spam if caller is clearly selling something unsolicited or uses robotic language\n\n"
    
    "Return null for any field where information is genuinely unclear or missing.")

    user_prompt = f"Analyze this phone call transcript and extract the structured information:\n\n{req.transcript}"

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        # Use OpenAI Responses API with structured (Pydantic) output
        response = client.responses.parse(
            model=model_name,
            input=[
                {"role": "system", "content": [{"type": "text", "text": system_instructions}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
            response_format=IntentResult,
        )

        parsed: IntentResult = response.output_parsed  # type: ignore[assignment]

        # FastAPI can return a Pydantic model directly, but ensure broad compatibility
        try:
            return IntentResult(**parsed.model_dump())  # Pydantic v2
        except AttributeError:  # Pydantic v1 fallback
            return IntentResult(**parsed.dict())

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - surfaces model or SDK errors cleanly
        raise HTTPException(status_code=500, detail=f"LLM classification failed: {e}")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )


