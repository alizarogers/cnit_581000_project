"""
models.py: request and response shapes for the API.
"""

from typing import List, Literal
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique per browser session")
    message: str = Field(..., min_length=1, description="The user's question or pun")
    history: List[ConversationTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
