"""
API tests with the Claude call mocked, so they run without an API key.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.llm_adapter import LLMAdapter
from backend.prompts import SYSTEM_PROMPT


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.config.LOG_DB_PATH", str(tmp_path / "log.db"))
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "**Pivot**: interest (homographic)"
    main_module.llm = mock_llm
    with patch("backend.main.log_query") as mock_log:
        yield TestClient(main_module.app), mock_llm, mock_log
    main_module.llm = None


def payload(**overrides):
    body = {"session_id": "s1", "message": "I used to be a banker, but I lost interest.", "history": []}
    body.update(overrides)
    return body


def test_health_ok(client):
    c, _, _ = client
    data = c.get("/health").json()
    assert data["status"] == "ok"
    assert data["model"]


def test_chat_returns_answer(client):
    c, _, _ = client
    r = c.post("/chat", json=payload())
    assert r.status_code == 200
    assert "interest" in r.json()["answer"]


def test_chat_uses_pun_system_prompt(client):
    c, mock_llm, _ = client
    c.post("/chat", json=payload())
    assert mock_llm.generate.call_args.kwargs["system_prompt"] == SYSTEM_PROMPT


def test_chat_passes_history_for_follow_ups(client):
    c, mock_llm, _ = client
    history = [
        {"role": "user", "content": "I used to be a banker, but I lost interest."},
        {"role": "assistant", "content": "Pivot: interest"},
    ]
    c.post("/chat", json=payload(message="Why is it funny?", history=history))
    assert len(mock_llm.generate.call_args.kwargs["history"]) == 2


def test_chat_logs_each_exchange(client):
    c, _, mock_log = client
    c.post("/chat", json=payload(session_id="log-me"))
    mock_log.assert_called_once()
    assert mock_log.call_args.kwargs["session_id"] == "log-me"


def test_missing_message_is_422(client):
    c, _, _ = client
    assert c.post("/chat", json={"session_id": "s1"}).status_code == 422


def test_bad_role_is_422(client):
    c, _, _ = client
    bad = payload(history=[{"role": "system", "content": "x"}])
    assert c.post("/chat", json=bad).status_code == 422


def test_no_api_key_returns_503(client):
    c, _, _ = client
    main_module.llm = None
    assert c.post("/chat", json=payload()).status_code == 503


def test_frontend_served(client):
    c, _, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "PunBot" in r.text


def test_logger_writes_sqlite(tmp_path, monkeypatch):
    import sqlite3
    from backend.logger import log_query

    db = tmp_path / "logs" / "chat.db"
    monkeypatch.setattr("backend.config.config.LOG_DB_PATH", str(db))
    log_query("s1", "model-x", "q", "a")
    rows = sqlite3.connect(db).execute("select session_id, model, query, response from chat_log").fetchall()
    assert rows == [("s1", "model-x", "q", "a")]


class TestLLMAdapter:
    def test_build_messages_appends_user_turn(self):
        msgs = LLMAdapter.build_messages([], "pun here")
        assert msgs == [{"role": "user", "content": "pun here"}]

    def test_build_messages_drops_leading_assistant_turn(self, monkeypatch):
        monkeypatch.setattr("backend.config.config.MAX_HISTORY_TURNS", 1)
        history = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]
        msgs = LLMAdapter.build_messages(history, "u4")
        assert msgs[0]["role"] == "user"
        assert msgs[-1] == {"role": "user", "content": "u4"}

    def test_generate_calls_messages_api(self):
        fake_client = MagicMock()
        block = MagicMock(type="text", text="analysis")
        fake_client.messages.create.return_value = MagicMock(content=[block])
        adapter = LLMAdapter(client=fake_client)

        out = adapter.generate("sys", [], "pun")

        assert out == "analysis"
        kwargs = fake_client.messages.create.call_args.kwargs
        assert kwargs["system"] == "sys"
        assert kwargs["messages"][-1] == {"role": "user", "content": "pun"}


def test_generate_retries_without_temperature_if_rejected():
    import anthropic
    import httpx

    fake_client = MagicMock()
    block = MagicMock(type="text", text="ok")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.BadRequestError(
        "temperature is not supported for this model",
        response=httpx.Response(400, request=request),
        body=None,
    )
    fake_client.messages.create.side_effect = [err, MagicMock(content=[block])]
    adapter = LLMAdapter(client=fake_client)

    assert adapter.generate("sys", [], "pun", temperature=0.0) == "ok"
    second_call = fake_client.messages.create.call_args_list[1].kwargs
    assert "temperature" not in second_call


def test_generate_retries_without_temperature_if_sdk_rejects_it():
    fake_client = MagicMock()
    block = MagicMock(type="text", text="ok")
    fake_client.messages.create.side_effect = [
        TypeError("Messages.create() got an unexpected keyword argument 'temperature'"),
        MagicMock(content=[block]),
    ]
    adapter = LLMAdapter(client=fake_client)

    assert adapter.generate("sys", [], "pun", temperature=0.0) == "ok"
    assert "temperature" not in fake_client.messages.create.call_args_list[1].kwargs
