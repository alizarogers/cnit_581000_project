"""
models.py: request and response shapes for the API.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique per browser session")
    message: str = Field(..., min_length=1, description="The user's question or pun")
    history: List[ConversationTurn] = Field(default_factory=list)
    analyze: Optional[bool] = Field(
        None, description="Force (true) or skip (false) the Stage 1-2 pipeline; default decides automatically")


class ChatResponse(BaseModel):
    answer: str
    model: str
    mode: str = Field(..., description='"full" (pipeline + Claude) or "llm_only"')
    sent_message: str = Field(..., description="What Claude received; store this in history for follow-ups")
    pipeline: Optional[dict] = Field(None, description="Stage 1-2 results when the pipeline ran")


class HealthResponse(BaseModel):
    status: str
    model: str
    mode: str
    pipeline: str
