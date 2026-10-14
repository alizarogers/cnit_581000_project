"""
config.py: all settings come from environment variables (.env locally).
Never hardcode the API key.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
    MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))
    LOG_DB_PATH: str = os.getenv("LOG_DB_PATH", "./logs/chat_log.db")
    CORS_ORIGIN: str = os.getenv("CORS_ORIGIN", "*")

    # Pun pipeline (Stages 1-2 in front of Claude)
    # "full": spaCy + BERT + WordNet evidence goes to Claude; "llm_only": Claude alone
    PIPELINE_MODE: str = os.getenv("PIPELINE_MODE", "full").strip().lower()
    BERT_MODEL: str = os.getenv("BERT_MODEL", "bert-base-uncased")
    SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    PUN_THRESHOLD: float = float(os.getenv("PUN_THRESHOLD", "0.30"))
    MAX_SENSES: int = int(os.getenv("MAX_SENSES", "8"))

    @classmethod
    def validate(cls) -> None:
        if not cls.ANTHROPIC_API_KEY:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )


config = Config()
