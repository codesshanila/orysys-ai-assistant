"""Nodes of the main assistant graph.

Each node is one agent with one job. They communicate only through the shared state
(app/agents/state.py), which makes every hand-off visible in LangSmith and in the
activity panel, and lets any node be tested on its own.

Failure policy (the "butterfly effect" guard): a dependency failure is contained in the
node that hit it. The node records the dependency under `degraded`, falls back to a
deterministic behaviour, and the graph keeps going. The response agent tells the user
what was unavailable. Only programming errors propagate.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import date
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.agents import prompts
from app.agents.events import emit, node
from app.agents.llm import get_gateway
from app.agents.memory import format_memory_context
from app.agents.research import research_graph
from app.agents.state import AssistantState
from app.config import get_settings
from app.errors import LLMUnavailableError, ToolPermissionError
from app.logging_setup import get_logger
from app.retrieval.hybrid import SearchFilters
from app.security.guardrails import (
    sanitize_retrieved,
    strip_invalid_citations,
    validate_answer,
    validate_user_input,
    wrap_evidence,
)
from app.security.rbac import APPROVAL_REQUIRED, allowed_tools, can_use_tool, require_tool
from app.services import get_memory, get_retriever
from app.tools.registry import TOOLS, ToolContext, ToolResult, execute_tool, tool_schemas_for

log = get_logger(__name__)

BLOCKED_PLACEHOLDER = "[request blocked by input guardrail]"
MAX_EVIDENCE_IN_PROMPT = 14
MAX_RESEARCH_EVIDENCE = 70
EVIDENCE_CHARS = 700  # per chunk in the response prompt; keeps a 70-chunk research answer ~12k tokens


def _history_text(messages: list[BaseMessage], limit: int = 8) -> str:
    """Recent turns as plain text (excluding the current question, which is last)."""
    lines = []
    for m in messages[:-1][-limit:]:
        who = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{who}: {str(m.content)[:500]}")
    return "\n".join(lines) or "(no earlier turns)"


def _evidence_from_hit(hit: dict, origin: str) -> tuple[dict, list[str]]:
    text, findings = sanitize_retrieved(hit["text"])
    keep = ("chunk_id", "doc_id", "title", "section", "document_type", "created_date",
            "access_level", "dense_score", "sparse_score", "hybrid_score", "rerank_score", "source")
    return {**{k: hit.get(k) for k in keep}, "text": text, "origin": origin}, findings


# --- 1. input guard ---------------------------------------------------------------------------
@node("input_guard", "guardrails")
async def input_guard(state: AssistantState) -> dict:
    result = validate_user_input(state["question"])
    emit("validation", stage="input", **result.as_event())
    if result.allowed:
        return {"guard": result.as_event(), "messages": [HumanMessage(result.sanitized)]}
    answer = (f"I can't help with that request. {result.reason} "
              "If you think this is a mistake, please rephrase your question or contact the IT service desk.")
    # The blocked text is not stored in conversation memory, so it can't influence later turns.
    return {"guard": result.as_event(), "blocked": True, "answer": answer, "final": True,
            "messages": [HumanMessage(BLOCKED_PLACEHOLDER)]}


def after_input_guard(state: AssistantState) -> str:
    return "memory_save" if state.get("blocked") else "memory_load"


# --- 2. memory load ---------------------------------------------------------------------------
async def _summarise(old: list[BaseMessage], previous: str) -> str:
    transcript = "\n".join(f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {str(m.content)[:400]}"
                           for m in old)
    try:
        return await get_gateway().text(prompts.SUMMARISER,
                                        f"Previous summary: {previous or '(none)'}\n\n{transcript}",
                                        tier="fast", run_name="memory_summarise")
    except LLMUnavailableError:
        questions = [str(m.content)[:120] for m in old if isinstance(m, HumanMessage)]
        return (previous + " Earlier questions: " + "; ".join(questions)).strip()[-1200:]


@node("memory_load", "memory")
async def memory_load(state: AssistantState) -> dict:
    user, messages = state["user"], state.get("messages", [])
    window = get_settings().memory_window_messages
    summary, update = state.get("summary", ""), {}

    if len(messages) > window:
        old = messages[:-window]
        summary = await _summarise(old, summary)
        update = {"summary": summary, "messages": [RemoveMessage(id=m.id) for m in old]}
        emit("memory", action="summarised", folded_messages=len(old), summary=summary[:300])

    recalled = await get_memory().recall(user["username"], state["question"])
    emit("memory", action="recalled", session_messages=len(messages),
         recent_questions=recalled["recent_questions"][:3],
         related=[{"q": r["q"], "overlap": r["overlap"]} for r in recalled["related"]],
         long_term_interactions=recalled["total_interactions"])
    return {**update, "memory_context": format_memory_context(user, recalled, summary)}


# --- 3. supervisor ----------------------------------------------------------------------------
class SupervisorDecision(BaseModel):
    intent: Literal["small_talk", "factual_lookup", "multi_document_research", "live_data",
                    "analysis", "action", "other"]
    standalone_query: str = Field(description="The question rewritten to stand alone")
    sub_questions: list[str] = Field(description="0-3 sub-questions; empty if not needed")
    steps: list[Literal["retrieval", "research", "tools"]]
    reason: str = Field(description="One sentence explaining the routing")
    document_types: list[str]
    departments: list[str]
    date_from: str | None
    date_to: str | None


RESEARCH_HINTS = re.compile(r"\b(summari[sz]e|all|every|recurring|trends?|across|compare|root causes|patterns?|last year|over the)\b", re.I)
TOOL_HINTS = re.compile(r"\b(who|owner|on-?call|contact|email|service catalog|status of|open incidents?|incident records?|update|resolve|close|count|how many|calculate)\b", re.I)
SMALL_TALK = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\b[\s!.]*$", re.I)


def heuristic_decision(question: str, role: str) -> SupervisorDecision:
    """Keyword routing used when the LLM is unavailable. Simple and predictable."""
    has_tools = any(t.startswith("mcp_") or t == "python_analysis" for t in allowed_tools(role))
    if SMALL_TALK.match(question):
        steps, intent = [], "small_talk"
    elif RESEARCH_HINTS.search(question):
        steps, intent = ["research"], "multi_document_research"
    elif TOOL_HINTS.search(question) and has_tools:
        steps, intent = ["retrieval", "tools"], "live_data"
    else:
        steps, intent = ["retrieval"], "factual_lookup"
    doc_types = ["incident"] if re.search(r"\b(outage|incident)s?\b", question, re.I) else []
    return SupervisorDecision(intent=intent, standalone_query=question, sub_questions=[], steps=steps,
                              reason="keyword routing (LLM unavailable)", document_types=doc_types,
                              departments=[], date_from=None, date_to=None)


@node("supervisor", "supervisor")
async def supervisor(state: AssistantState) -> dict:
    role = state["user"]["role"]
    tools = [t for t in allowed_tools(role) if t in TOOLS]
    system = prompts.SUPERVISOR.format(tools=", ".join(tools) or "none", today=date.today().isoformat())
    user_prompt = (f"Memory:\n{state.get('memory_context', '')}\n\nConversation so far:\n"
                   f"{_history_text(state.get('messages', []))}\n\nCurrent question: {state['question']}")
    degraded = []
    try:
        decision = await get_gateway().structured(SupervisorDecision, system, user_prompt, run_name="supervisor")
    except LLMUnavailableError:
        decision, degraded = heuristic_decision(state["question"], role), ["llm"]

    steps = list(dict.fromkeys(decision.steps))  # dedupe, keep order
    if "research" in steps and "retrieval" in steps:
        steps.remove("retrieval")
    if "tools" in steps and not tools:
        steps.remove("tools")
    filters = {"document_types": decision.document_types or None, "departments": decision.departments or None,
               "date_from": decision.date_from, "date_to": decision.date_to}
    emit("agent_state", agent="supervisor", intent=decision.intent, plan=steps, reason=decision.reason,
         standalone_query=decision.standalone_query, sub_questions=decision.sub_questions,
         filters={k: v for k, v in filters.items() if v})
    return {"intent": decision.intent, "standalone_query": decision.standalone_query or state["question"],
            "sub_questions": decision.sub_questions[:3], "plan": steps, "plan_reason": decision.reason,
            "filters": filters, "step_index": 0, "degraded": degraded}


def next_step(state: AssistantState) -> str:
    """Route to the next planned worker, or to the response agent when the plan is done."""
    plan, i = state.get("plan") or [], state.get("step_index", 0)
    if i >= len(plan):
        return "response"
    return {"retrieval": "retrieval", "research": "research", "tools": "tool_planner"}[plan[i]]


# --- 4a. retrieval agent ------------------------------------------------------------------------
def _search_filters(state: AssistantState) -> SearchFilters:
    f = state.get("filters") or {}
    return SearchFilters(document_types=f.get("document_types"), departments=f.get("departments"),
                         date_from=f.get("date_from"), date_to=f.get("date_to"))


@node("retrieval", "retrieval")
async def retrieval(state: AssistantState) -> dict:
    queries = [state.get("standalone_query") or state["question"], *state.get("sub_questions", [])[:2]]
    emit("retrieval", status="started", queries=queries)
    retriever, role, filters = get_retriever(), state["user"]["role"], _search_filters(state)
    results = await asyncio.gather(*[retriever.search(q, role, filters) for q in queries])

    evidence, seen, injection_findings, degraded = [], set(), [], []
    for res in results:
        emit("retrieval", status="done", **res.summary())
        if res.degraded:
            degraded.append(f"vector_db: {res.degraded_reason}")
        for hit in res.hits:
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                ev, findings = _evidence_from_hit(hit.to_dict(), "retrieval")
                evidence.append(ev)
                injection_findings += [f"{hit.chunk_id}: {x}" for x in findings]
    if injection_findings:
        emit("validation", stage="retrieved_content", allowed=True, findings=injection_findings,
             reason="instruction-like text in retrieved documents was neutralised")
    return {"evidence": evidence, "degraded": sorted(set(degraded)), "step_index": state.get("step_index", 0) + 1}


# --- 4b. research agent (RLM subgraph) ------------------------------------------------------------
@node("research", "research")
async def research(state: AssistantState) -> dict:
    role = state["user"]["role"]
    try:
        require_tool(role, "deep_research")
    except ToolPermissionError:
        return {"step_index": state.get("step_index", 0) + 1, "degraded": ["research: not permitted"]}
    result = await research_graph.ainvoke({
        "question": state.get("standalone_query") or state["question"], "role": role,
        "filters": state.get("filters") or {}, "depth": 0})
    evidence = [{**e, "origin": "research"} for e in result.get("evidence", [])]
    summary = {k: result.get(k) for k in ("synthesis", "themes", "follow_ups", "sub_results", "plan_source",
                                          "plan_code", "stats", "errors")}
    summary["batches"] = len(result.get("batches") or [])
    summary["findings"] = result.get("findings", [])
    degraded = ["research: " + e for e in result.get("errors", [])]
    return {"research": summary, "evidence": evidence, "degraded": degraded,
            "step_index": state.get("step_index", 0) + 1}


# --- 4c. tools agent: planner + executor (split so a HITL interrupt never re-runs the LLM) -----
SERVICE_NAMES = ("Payment Switch", "Fund Transfer Engine", "Card Authorisation Service",
                 "Bill Payments Service", "Mobile Payments API")


def heuristic_tool_calls(question: str, allowed: list[str]) -> list[dict]:
    q = question.lower()
    service = next((s for s in SERVICE_NAMES if s.lower() in q), None)
    calls = []
    inc = re.search(r"INC-\d{4}-\d{3}", question)
    if inc and re.search(r"\b(resolve|close|mark|update|set)\b", q):
        status = "resolved" if re.search(r"resolve|close", q) else "monitoring"
        calls.append({"name": "mcp_update_incident_status",
                      "args": {"incident_id": inc.group(0), "status": status, "note": "Requested via assistant"}})
    elif re.search(r"\bincidents?\b", q):
        calls.append({"name": "mcp_incident_records", "args": {"service": service} if service else {}})
    if re.search(r"\b(who|owner|on-?call|contact|team)\b", q):
        calls.append({"name": "mcp_service_catalog", "args": {"service": service} if service else {}})
    return [c for c in calls if c["name"] in allowed]


@node("tool_planner", "tools")
async def tool_planner(state: AssistantState) -> dict:
    role = state["user"]["role"]
    allowed = [t for t in allowed_tools(role) if t in TOOLS]
    evidence_brief = "\n".join(f"- [{e['chunk_id']}] {e['title']} / {e['section']}" for e in (state.get("evidence") or [])[:10])
    prompt = (f"Question: {state.get('standalone_query') or state['question']}\n"
              f"Conversation:\n{_history_text(state.get('messages', []))}\n"
              f"Evidence already retrieved:\n{evidence_brief or '(none)'}")
    degraded = []
    try:
        calls = await get_gateway().tool_calls(prompts.TOOLS_AGENT, prompt, tool_schemas_for(allowed))
    except LLMUnavailableError:
        calls, degraded = heuristic_tool_calls(state["question"], allowed), ["llm"]
    calls = calls[:3]
    emit("agent_state", agent="tools", planned_calls=calls)
    return {"pending_tool_calls": calls, "degraded": degraded}


@node("tool_executor", "tools")
async def tool_executor(state: AssistantState) -> dict:
    user = state["user"]
    calls = state.get("pending_tool_calls") or []

    # Human in the loop: state-changing tools pause the graph until a person approves.
    # Only calls the role is allowed to make are put to a human; others are denied below.
    needs_approval = [c for c in calls if c["name"] in APPROVAL_REQUIRED and can_use_tool(user["role"], c["name"])]
    decisions: dict = {}
    if needs_approval:
        # No emit here: LangGraph re-runs this node on resume, so an event before interrupt()
        # would fire twice. The API forwards the interrupt itself as `approval_required`.
        decisions = interrupt({"type": "approval_required", "calls": needs_approval})
        emit("approval_result", decision=decisions)

    ctx = ToolContext(role=user["role"], username=user["username"], evidence=state.get("evidence") or [])

    async def run(call: dict) -> ToolResult:
        if call in needs_approval and not decisions.get("approved"):
            return ToolResult(name=call["name"], args=call["args"], status="rejected",
                              error="rejected by human reviewer")
        emit("tool_call", tool=call["name"], args=call["args"], status="running")
        return await execute_tool(call["name"], call["args"], ctx)

    results = await asyncio.gather(*[run(c) for c in calls])
    degraded = []
    for r in results:
        emit("tool_call", tool=r.name, args=r.args, status=r.status, error=r.error, elapsed_ms=r.elapsed_ms,
             output_preview=json.dumps(r.output)[:400] if r.output else None)
        if r.status == "error":
            degraded.append(f"tool {r.name}: {r.error}")
    return {"tool_results": [r.to_dict() for r in results], "pending_tool_calls": [],
            "degraded": degraded, "step_index": state.get("step_index", 0) + 1}


# --- 5. response agent -----------------------------------------------------------------------------
def _prompt_evidence(state: AssistantState) -> list[dict]:
    """Evidence shown to the response agent: everything the research findings cite first,
    then the best remaining chunks, so research claims can be cited and verified."""
    evidence = state.get("evidence") or []
    cited = {cid for f in (state.get("research") or {}).get("findings", []) for cid in f.get("chunk_ids", [])}
    first = [e for e in evidence if e["chunk_id"] in cited]
    rest = [e for e in evidence if e["chunk_id"] not in cited]
    return (first + rest)[:max(MAX_EVIDENCE_IN_PROMPT, len(first))][:MAX_RESEARCH_EVIDENCE]


def _response_prompt(state: AssistantState) -> str:
    evidence = _prompt_evidence(state)
    parts = [f"Memory:\n{state.get('memory_context', '')}",
             f"Conversation so far:\n{_history_text(state.get('messages', []))}",
             f"Question: {state.get('standalone_query') or state['question']}"]
    if evidence:
        parts.append("EVIDENCE:\n" + "\n".join(wrap_evidence(e["chunk_id"], e["title"], e["text"][:EVIDENCE_CHARS]) for e in evidence))
    if state.get("research"):
        r = state["research"]
        parts.append(f"RESEARCH SYNTHESIS (from {r.get('batches')} parallel batch analyses):\n{r.get('synthesis')}\n"
                     f"Themes: {json.dumps(r.get('themes'))[:1500]}\nFindings: {json.dumps(r.get('findings'))[:3000]}\n"
                     f"Sub-research: {json.dumps(r.get('sub_results'))[:1500]}")
    if state.get("tool_results"):
        parts.append("TOOL RESULTS:\n" + json.dumps(state["tool_results"], default=str)[:4000])
    if state.get("degraded"):
        parts.append("DEGRADED (unavailable this turn): " + "; ".join(state["degraded"]))
    v = state.get("validation") or {}
    if v and not v.get("valid", True):
        parts.append(f"Your previous draft failed validation: {v.get('issues')}. Invalid citations: "
                     f"{v.get('invalid_citations')}. Rewrite it, citing only evidence ids listed above.")
    return "\n\n".join(parts)


def extractive_answer(state: AssistantState) -> str:
    """Answer without the LLM: the best passages, research counts and tool output, with citations."""
    lines = ["The AI model is temporarily unavailable, so here is what I found directly in the sources:"]
    r = state.get("research") or {}
    if r.get("synthesis"):
        lines.append(f"\n{r['synthesis']}")
        for t in (r.get("themes") or [])[:4]:
            lines.append(f"- {t['label']}: {t['count']} document(s) ({', '.join(t['doc_ids'][:6])})")
    for e in (state.get("evidence") or [])[:4]:
        text = e["text"]
        prefix = f"{e['title']} - {e['section']}: "  # chunks are stored as "<title> - <section>: <body>"
        snippet = (text[len(prefix):] if text.startswith(prefix) else text).strip()[:220]
        label = e["title"] if e["section"] in ("", e["title"]) else f"{e['title']} / {e['section']}"
        lines.append(f"- {label}: {snippet} [{e['chunk_id']}]")
    for t in state.get("tool_results") or []:
        if t["status"] == "ok":
            lines.append(f"- Tool {t['name']}: {json.dumps(t['output'])[:300]} [tool:{t['name']}]")
    if len(lines) == 1:
        lines.append("I couldn't find relevant information. Please try again shortly.")
    return "\n".join(lines)


@node("response", "response")
async def response(state: AssistantState) -> dict:
    emit("answer_reset")
    try:
        answer = await get_gateway().stream(prompts.RESPONSE, _response_prompt(state),
                                            on_token=lambda t: emit("token", text=t), run_name="response")
        degraded = []
    except LLMUnavailableError:
        answer, degraded = extractive_answer(state), ["llm"]
        emit("token", text=answer)
    return {"answer": answer, "degraded": degraded}


# --- 6. validator ---------------------------------------------------------------------------------
SAFE_BRAND_REPLY = ("I'm not able to provide that response. For product, investment or service commitments, "
                    "please contact the relevant Commercial Bank team through official channels.")


@node("validator", "guardrails")
async def validator(state: AssistantState) -> dict:
    evidence_ids = {e["chunk_id"] for e in state.get("evidence") or []}
    check = validate_answer(state.get("answer", ""), evidence_ids, require_citations=bool(evidence_ids))
    attempts = state.get("validation_attempts", 0)
    event = {**check.as_event(), "attempt": attempts + 1}

    if check.valid:
        emit("validation", stage="output", action="accepted", **event)
        return {"validation": event, "answer": check.text, "final": True}
    if attempts == 0 and get_gateway().configured and "llm" not in (state.get("degraded") or []):
        emit("validation", stage="output", action="retry", **event)
        return {"validation": event, "validation_attempts": 1}

    # Second failure: repair deterministically instead of looping.
    text = strip_invalid_citations(check.text, check.invalid_citations)
    if any(i.startswith("brand:") for i in check.issues):
        text = SAFE_BRAND_REPLY
    emit("validation", stage="output", action="repaired", **event)
    return {"validation": {**event, "repaired": True}, "answer": text, "final": True}


def after_validator(state: AssistantState) -> str:
    return "memory_save" if state.get("final") else "response"


# --- 7. memory save ----------------------------------------------------------------------------------
@node("memory_save", "memory")
async def memory_save(state: AssistantState) -> dict:
    answer = state.get("answer", "")
    citations = (state.get("validation") or {}).get("citations", [])
    if not state.get("blocked"):
        await get_memory().remember_interaction(state["user"]["username"], state["question"], answer, citations)
        emit("memory", action="saved", long_term=True, citations=citations)
    else:
        emit("memory", action="skipped", reason="blocked requests are not remembered")
    emit("final", answer=answer, citations=citations, degraded=state.get("degraded") or [],
         intent=state.get("intent", ""), plan=state.get("plan") or [])
    return {"messages": [AIMessage(answer)]}
