"""
main.py: FastAPI app for PunBot.

Run from the project root:  uvicorn backend.main:app --reload
Then open http://127.0.0.1:8000
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import config
from backend.llm_adapter import LLMAdapter
from backend.logger import log_query
from backend.models import ChatRequest, ChatResponse, HealthResponse
from backend.prompts import SYSTEM_PROMPT

llm: Optional[LLMAdapter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    if llm is None:
        try:
            llm = LLMAdapter()
            print(f"PunBot ready, using {config.CLAUDE_MODEL}")
        except EnvironmentError as e:
            print(f"Config error: {e}")
    yield


app = FastAPI(title="PunBot", description="Ask Claude questions about puns", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.CORS_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    status = "ok" if llm is not None else "degraded: ANTHROPIC_API_KEY not set"
    return HealthResponse(status=status, model=config.CLAUDE_MODEL)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set. Add it to .env and restart.")

    history = [{"role": t.role, "content": t.content} for t in request.history]
    try:
        answer = llm.generate(system_prompt=SYSTEM_PROMPT, history=history, user_message=request.message)
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e}")

    log_query(session_id=request.session_id, model=config.CLAUDE_MODEL, query=request.message, response=answer)
    return ChatResponse(answer=answer, model=config.CLAUDE_MODEL)


frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(os.path.join(frontend_path, "index.html"))
