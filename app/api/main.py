"""FastAPI backend.

Endpoints
  POST /auth/login       username/password -> JWT (hardcoded demo users, PBKDF2 hashes)
  POST /chat/stream      one turn; Server-Sent Events: activity events, answer tokens, final
  POST /chat/resume      approve/reject a paused state-changing tool call (human in the loop)
  POST /feedback         thumbs up/down, attached to the LangSmith trace of that turn
  GET/POST /admin/chaos  failure injection for the demo (administrators only)
  GET  /health           dependency status: LLM, Pinecone, MCP, circuit breakers
  GET  /me               current user and rate-limit bucket

Request path for /chat/stream
  JWT -> user (role from the server-side record) -> token-bucket rate limit
  -> graph.astream(...) with thread_id = "<username>:<session_id>"
  (a user can never resume or read another user's conversation)
  -> every custom event is forwarded as one SSE `data:` line.
Errors inside the stream become a typed `error` event with a safe message; the full
exception goes to the structured log only.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents.graph import build_graph
from app.agents.llm import get_gateway
from app.agents.state import new_turn_input
from app.config import configure_tracing, get_settings
from app.errors import AssistantError, RateLimitExceeded, ToolPermissionError
from app.logging_setup import bind_request_context, configure_logging, get_logger
from app.resilience import BREAKERS, chaos
from app.security.auth import User, authenticate, decode_token, issue_token
from app.security.rate_limit import TokenBucketLimiter
from app.security.rbac import allowed_tools, require_tool
from app.services import get_retriever

log = get_logger(__name__)
settings = get_settings()
limiter = TokenBucketLimiter(settings.rate_limit_capacity, settings.rate_limit_refill_per_minute)
bearer = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    tracing = configure_tracing(settings)
    from pathlib import Path
    Path(settings.checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as saver:
        app.state.graph = build_graph(checkpointer=saver)
        get_retriever()  # load the chunk store and BM25 stats once, at startup
        log.info("startup", tracing=tracing, llm=get_gateway().configured,
                 pinecone=bool(settings.pinecone_api_key))
        yield


app = FastAPI(title=settings.assistant_name, version="1.0.0", lifespan=lifespan)


# --- errors ------------------------------------------------------------------------------------
@app.exception_handler(AssistantError)
async def assistant_error_handler(_: Request, exc: AssistantError) -> JSONResponse:
    headers = {"Retry-After": f"{exc.retry_after_s:.0f}"} if isinstance(exc, RateLimitExceeded) else None
    log.warning("request_failed", error=type(exc).__name__, detail=str(exc))
    return JSONResponse({"error": type(exc).__name__, "message": exc.user_message},
                        status_code=exc.status_code, headers=headers)


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_error", error=type(exc).__name__)
    return JSONResponse({"error": "InternalError", "message": AssistantError.user_message}, status_code=500)


# --- auth --------------------------------------------------------------------------------------
async def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> User:
    user = decode_token(creds.credentials) if creds else None
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


def user_context(user: User) -> dict:
    return {"username": user.username, "display_name": user.display_name,
            "role": user.role.value, "department": user.department}


class LoginRequest(BaseModel):
    username: str = Field(..., max_length=50)
    password: str = Field(..., max_length=100)


@app.post("/auth/login")
async def login(body: LoginRequest) -> dict:
    user = authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"token": issue_token(user), "user": user_context(user), "tools": allowed_tools(user.role)}


@app.get("/me")
async def me(user: User = Depends(current_user)) -> dict:
    return {"user": user_context(user), "tools": allowed_tools(user.role),
            "rate_limit": await limiter.status(user.username)}


# --- chat --------------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{4,64}$")


class ResumeRequest(BaseModel):
    session_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{4,64}$")
    approved: bool


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


async def _run_graph(graph, graph_input, user: User, session_id: str) -> AsyncIterator[str]:
    run_id = str(uuid.uuid4())
    thread_id = f"{user.username}:{session_id}"
    config = {"configurable": {"thread_id": thread_id}, "run_id": run_id, "run_name": "assistant_turn",
              "metadata": {"user": user.username, "role": user.role.value, "session_id": session_id},
              "tags": [f"role:{user.role.value}"]}
    bind_request_context(request_id=run_id, user=user.username, role=user.role.value, session=session_id)
    yield _sse({"type": "run_started", "run_id": run_id, "session_id": session_id})
    try:
        async for _ns, mode, event in graph.astream(graph_input, config, stream_mode=["custom", "updates"],
                                                    subgraphs=True):
            if mode == "custom":
                yield _sse(event)
            elif mode == "updates" and "__interrupt__" in event:
                payload = event["__interrupt__"][0].value
                yield _sse({**payload, "type": "approval_required", "run_id": run_id})
    except AssistantError as exc:
        log.warning("turn_failed", error=type(exc).__name__, detail=str(exc))
        yield _sse({"type": "error", "error": type(exc).__name__, "message": exc.user_message})
    except Exception as exc:  # never leak internals into the chat
        log.exception("turn_crashed", error=type(exc).__name__)
        yield _sse({"type": "error", "error": "InternalError", "message": AssistantError.user_message})
    yield _sse({"type": "run_finished", "run_id": run_id})


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request, user: User = Depends(current_user)):
    await limiter.acquire(user.username)
    graph_input = new_turn_input(body.message, user_context(user), body.session_id)
    return StreamingResponse(_run_graph(request.app.state.graph, graph_input, user, body.session_id),
                             media_type="text/event-stream")


@app.post("/chat/resume")
async def chat_resume(body: ResumeRequest, request: Request, user: User = Depends(current_user)):
    graph = request.app.state.graph
    state = await graph.aget_state({"configurable": {"thread_id": f"{user.username}:{body.session_id}"}})
    if not state.next:
        raise HTTPException(status_code=409, detail="Nothing is waiting for approval in this session")
    return StreamingResponse(_run_graph(graph, Command(resume={"approved": body.approved}), user, body.session_id),
                             media_type="text/event-stream")


@app.get("/chat/history/{session_id}")
async def chat_history(session_id: str, request: Request, user: User = Depends(current_user)) -> dict:
    state = await request.app.state.graph.aget_state({"configurable": {"thread_id": f"{user.username}:{session_id}"}})
    messages = state.values.get("messages", []) if state.values else []
    return {"summary": state.values.get("summary", "") if state.values else "",
            "messages": [{"role": "user" if m.type == "human" else "assistant", "content": m.content}
                         for m in messages]}


# --- feedback (quality loop) -------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    run_id: str = Field(..., pattern=r"^[0-9a-f-]{36}$")
    score: int = Field(..., ge=0, le=1)
    comment: str = Field("", max_length=500)


@app.post("/feedback")
async def feedback(body: FeedbackRequest, user: User = Depends(current_user)) -> dict:
    log.info("user_feedback", run_id=body.run_id, score=body.score, comment=body.comment)
    if not configure_tracing(settings):
        return {"stored": "log_only"}
    try:
        from langsmith import Client
        Client().create_feedback(body.run_id, key="user_score", score=body.score, comment=body.comment,
                                 source_info={"user": user.username})
        return {"stored": "langsmith"}
    except Exception as exc:
        log.warning("feedback_failed", error=f"{type(exc).__name__}: {exc}")
        return {"stored": "log_only"}


# --- admin: failure injection ------------------------------------------------------------------
class ChaosRequest(BaseModel):
    llm_fail: bool = False
    vector_db_fail: bool = False
    mcp_fail: bool = False
    tool_slow: bool = False


@app.get("/admin/chaos")
async def get_chaos(user: User = Depends(current_user)) -> dict:
    require_tool(user.role, "admin_chaos_control")
    return chaos.as_dict()


@app.post("/admin/chaos")
async def set_chaos(body: ChaosRequest, user: User = Depends(current_user)) -> dict:
    try:
        require_tool(user.role, "admin_chaos_control")
    except ToolPermissionError:
        log.warning("chaos_denied", user=user.username)
        raise
    for key, value in body.model_dump().items():
        setattr(chaos, key, value)
    for breaker in BREAKERS.values():
        breaker.record_success()  # reset so the demo reflects the new switches immediately
    log.info("chaos_updated", user=user.username, **chaos.as_dict())
    return chaos.as_dict()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "llm_configured": get_gateway().configured,
            "pinecone_configured": bool(settings.pinecone_api_key),
            "langsmith_tracing": configure_tracing(settings),
            "breakers": [b.snapshot() for b in BREAKERS.values()], "chaos": chaos.as_dict()}
