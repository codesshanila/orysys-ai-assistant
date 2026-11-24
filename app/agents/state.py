"""Graph state shared by every agent.

Two lifetimes live in one state object, because the checkpointer persists the whole
state per conversation thread:

  * Conversation-scoped: `messages`, `summary`. They must survive across turns; this is
    the short-term memory.
  * Turn-scoped: evidence, tool results, plan, validation... They must start empty on
    every new question, or turn 2 would cite turn 1's evidence.

Turn-scoped list fields use `merge_or_reset`: a node *appends* by returning a list, and
the API *resets* the field at the start of a turn by sending `None`. This keeps parallel
writers (several retrieval or tool calls in one step) safe without manual bookkeeping.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_or_reset(left: list | None, right: list | None) -> list:
    if right is None:
        return []
    return (left or []) + right


class UserContext(TypedDict):
    username: str
    display_name: str
    role: str
    department: str


class AssistantState(TypedDict, total=False):
    # --- conversation-scoped (short-term memory) ---------------------------------------
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str  # rolling summary of turns that fell out of the message window

    # --- request identity: set by the API from the verified JWT, never by the model ----
    user: UserContext
    session_id: str

    # --- turn-scoped --------------------------------------------------------------------
    question: str
    blocked: bool
    guard: dict
    memory_context: str  # long-term memory recalled for this turn

    intent: str
    standalone_query: str
    sub_questions: list[str]
    filters: dict
    plan: list[str]       # ordered worker steps chosen by the supervisor
    plan_reason: str
    step_index: int

    evidence: Annotated[list[dict], merge_or_reset]      # retrieved, sanitised chunks
    pending_tool_calls: list[dict]                        # chosen by tool_planner, run by tool_executor
    tool_results: Annotated[list[dict], merge_or_reset]  # tool name, args, output, status
    research: dict                                        # RLM findings, if research ran
    degraded: Annotated[list[str], merge_or_reset]       # dependencies that failed this turn

    answer: str
    validation: dict
    validation_attempts: int
    final: bool


def new_turn_input(question: str, user: UserContext, session_id: str) -> dict[str, Any]:
    """Initial state for one turn: identity + question, and every turn-scoped field reset."""
    return {
        "question": question,
        "user": user,
        "session_id": session_id,
        "blocked": False,
        "guard": {},
        "memory_context": "",
        "intent": "",
        "standalone_query": "",
        "sub_questions": [],
        "filters": {},
        "plan": [],
        "plan_reason": "",
        "step_index": 0,
        "evidence": None,
        "pending_tool_calls": [],
        "tool_results": None,
        "research": {},
        "degraded": None,
        "answer": "",
        "validation": {},
        "validation_attempts": 0,
        "final": False,
    }
