"""
logger.py: log every question and answer to a local SQLite file.

The log is handy evidence for the evaluation section of the project:
export it with `sqlite3 logs/chat_log.db ".mode csv" "select * from chat_log"`.
"""

import os
import sqlite3
from datetime import datetime, timezone

from backend.config import config


def _connect() -> sqlite3.Connection:
    db_dir = os.path.dirname(config.LOG_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(config.LOG_DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chat_log (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               session_id TEXT NOT NULL,
               model TEXT NOT NULL,
               query TEXT NOT NULL,
               response TEXT NOT NULL,
               timestamp TEXT NOT NULL
           )"""
    )
    return conn


def log_query(session_id: str, model: str, query: str, response: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO chat_log (session_id, model, query, response, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, model, query, response, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
