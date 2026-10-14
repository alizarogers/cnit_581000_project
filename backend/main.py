"""
main.py: FastAPI app for PunBot.

Run from the project root:  uvicorn backend.main:app --reload
Then open http://127.0.0.1:8000
"""

import json
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
from backend.punbot import PunBot, build_pipeline

bot: Optional[PunBot] = None
pipeline_status = "not loaded"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot, pipeline_status
    if bot is None:
        try:
            llm = LLMAdapter()
        except EnvironmentError as e:
            print(f"Config error: {e}")
            yield
            return
        pipeline = None
        if config.PIPELINE_MODE == "full":
            print("Loading Stages 1-2 (spaCy, BERT, WordNet); the first run downloads the models...")
            try:
                pipeline = build_pipeline(config)
                pipeline_status = f"ready ({config.SPACY_MODEL} + {config.BERT_MODEL} + WordNet)"
            except Exception as e:  # missing packages or models: fall back to Claude alone
                pipeline_status = f"unavailable, running Claude only: {e}"
        else:
            pipeline_status = "off (PIPELINE_MODE=llm_only)"
        bot = PunBot(llm, pipeline)
        print(f"PunBot ready: {config.CLAUDE_MODEL}; pipeline {pipeline_status}")
    yield


app = FastAPI(title="PunBot", description="Ask questions about puns: spaCy + BERT + WordNet + Claude",
              version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.CORS_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    status = "ok" if bot is not None else "degraded: ANTHROPIC_API_KEY not set"
    return HealthResponse(status=status, model=config.CLAUDE_MODEL,
                          mode=bot.mode if bot else "none", pipeline=pipeline_status)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    if bot is None:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set. Add it to .env and restart.")

    history = [{"role": t.role, "content": t.content} for t in request.history]
    try:
        out = bot.respond(request.message, history, analyze=request.analyze)
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e}")

    logged_query = request.message
    if out["pipeline"]:
        logged_query += "\n[pipeline] " + json.dumps(
            {"call": out["pipeline"]["call"], "pivot": (out["pipeline"]["best"] or {}).get("text")})
    log_query(session_id=request.session_id, model=config.CLAUDE_MODEL, query=logged_query, response=out["answer"])
    return ChatResponse(model=config.CLAUDE_MODEL, **out)


frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(os.path.join(frontend_path, "index.html"))
