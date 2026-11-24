import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def login(client, user: str) -> dict:
    token = client.post("/auth/login", json={"username": user, "password": f"{user}123"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_login_rejects_bad_password(client):
    assert client.post("/auth/login", json={"username": "admin", "password": "nope"}).status_code == 401


def test_endpoints_require_a_token(client):
    assert client.post("/chat/stream", json={"message": "hi", "session_id": "abcd"}).status_code == 401
    assert client.get("/me", headers={"Authorization": "Bearer forged"}).status_code == 401


def test_only_admin_can_inject_failures(client):
    assert client.post("/admin/chaos", json={"llm_fail": True}, headers=login(client, "analyst")).status_code == 403
    admin = login(client, "admin")
    assert client.post("/admin/chaos", json={"mcp_fail": True}, headers=admin).json()["mcp_fail"] is True
    client.post("/admin/chaos", json={}, headers=admin)


def test_chat_stream_emits_activity_and_final_answer(client):
    headers = login(client, "viewer")
    with client.stream("POST", "/chat/stream", headers=headers,
                       json={"message": "What is the incident escalation policy?", "session_id": "api1"}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    types = {e["type"] for e in events}
    assert {"run_started", "node_start", "retrieval", "validation", "final", "run_finished"} <= types
    history = client.get("/chat/history/api1", headers=headers).json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_sessions_are_private_per_user(client):
    assert client.get("/chat/history/api1", headers=login(client, "analyst")).json()["messages"] == []


def test_resume_without_pending_approval_is_rejected(client):
    r = client.post("/chat/resume", json={"session_id": "api1", "approved": True}, headers=login(client, "viewer"))
    assert r.status_code == 409
