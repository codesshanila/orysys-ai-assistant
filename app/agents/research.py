"""Research agent: a Recursive Language Model (RLM) style subgraph.

The idea: never paste whole documents into the context window. Treat the collection as
an environment the model *programs against*, then work on small pieces in parallel.

    explore ──► plan ──► execute_plan ──► analyze_batch ×N (parallel, Send) ──► aggregate
                 ▲                                                               │
                 └──────── recurse: the same subgraph, on follow-up questions ◄──┘

  explore       Catalog statistics only (types, services, dates). No document text.
  plan          The LLM writes a short *Python search plan* against a 4-function DSL
                (catalog_filter, search, sections, batch). The code is AST-validated by
                the sandbox; if it's invalid or the LLM is down, a deterministic default
                plan is used, so the agent always has a plan.
  execute_plan  Runs the plan in the sandbox. It retrieves targeted sections
                (e.g. only "Root cause" of each incident) and splits them into batches.
  analyze_batch One fast-model call per batch, fanned out with LangGraph `Send`. A failed
                batch records an error and returns no findings; the others carry on.
  aggregate     Deterministic theme counts + an LLM synthesis. If the synthesis names
                follow-up questions and depth allows, `recurse` invokes this same compiled
                subgraph on each follow-up (true recursion, bounded by rlm_max_depth).

Everything the agent does is emitted as `research` events, so the evaluator can watch
the plan code, the batches and each sub-agent in the activity panel.
"""
from __future__ import annotations

import asyncio
import operator
import re
from collections import Counter, defaultdict
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from app.agents import prompts
from app.agents.events import emit, node
from app.agents.llm import get_gateway
from app.config import get_settings
from app.errors import LLMUnavailableError, SandboxError
from app.logging_setup import get_logger
from app.retrieval.hybrid import SearchFilters
from app.security.guardrails import sanitize_retrieved
from app.security.sandbox import run_sandboxed, validate_code
from app.services import get_retriever
from app.tools.analysis import role_catalog

log = get_logger(__name__)


class ResearchState(TypedDict, total=False):
    question: str
    role: str
    filters: dict
    depth: int
    stats: dict
    plan_code: str
    plan_source: str
    focus: str
    batches: list[list[dict]]
    findings: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    evidence: list[dict]
    synthesis: str
    themes: list[dict]
    follow_ups: list[str]
    sub_results: list[dict]


# --- structured outputs --------------------------------------------------------------------
class SearchPlan(BaseModel):
    rationale: str = Field(description="One sentence: why this plan answers the question")
    code: str = Field(description="Python search plan that assigns `batches` and `focus`")


class Finding(BaseModel):
    doc_id: str
    finding: str
    theme: str = Field(description="2-5 word normalised root-cause or theme label")
    chunk_ids: list[str]


class BatchFindings(BaseModel):
    findings: list[Finding]


class Theme(BaseModel):
    label: str
    count: int
    doc_ids: list[str]


class Synthesis(BaseModel):
    summary: str
    themes: list[Theme]
    follow_ups: list[str] = Field(description="0-2 narrower follow-up questions, or empty")


# --- default plan (used when the LLM plan is unavailable or rejected) ----------------------
# Written in the same DSL as LLM plans and run through the same sandbox. `question` and
# `filters` are injected variables, never string-formatted into the code.
DEFAULT_PLAN = """
docs = catalog_filter(document_type=filters.get("document_type"),
                      date_from=filters.get("date_from"), date_to=filters.get("date_to"))
allowed = [d["doc_id"] for d in docs]
hits = search(question, doc_ids=allowed or None, top_k=16)
doc_ids = []
for h in hits:
    if h["doc_id"] not in doc_ids:
        doc_ids.append(h["doc_id"])
chunks = []
for d in doc_ids[:MAX_DOCS]:
    picked = sections(d, ["Summary", "Root cause", "Decision", "Actions"])
    chunks = chunks + (picked if picked else [h for h in hits if h["doc_id"] == d])
batches = batch(chunks, BATCH_SIZE)
focus = "Extract the key finding and the root cause or theme for each document."
""".strip()

THEME_KEYWORDS = {
    "certificate expiry": ("certificate", "tls", "expired"),
    "connection pool exhaustion": ("connection pool", "connections", "pool"),
    "untested configuration change": ("configuration change", "config change", "misconfigur"),
    "third-party gateway failure": ("gateway provider", "third-party", "vendor", "upstream"),
    "capacity / peak load": ("peak load", "capacity", "saturat"),
}


def heuristic_theme(text: str) -> str:
    lowered = text.lower()
    for theme, words in THEME_KEYWORDS.items():
        if any(w in lowered for w in words):
            return theme
    return "other"


# --- DSL exposed to plans --------------------------------------------------------------------
class PlanDSL:
    """The only capabilities a search plan has. Runs in a worker thread; async retrieval is
    bridged back to the event loop with run_coroutine_threadsafe."""

    def __init__(self, role: str, loop: asyncio.AbstractEventLoop):
        settings = get_settings()
        self.role = role
        self.loop = loop
        self.catalog = role_catalog(role)
        self.allowed_ids = {row["doc_id"] for row in self.catalog}
        self.retriever = get_retriever()
        self.max_chunks = settings.rlm_max_docs * 3
        self.timeout = settings.retrieval_timeout_s + 5
        self.degraded = False

    def catalog_filter(self, document_type=None, department=None, service=None, severity=None,
                       date_from=None, date_to=None, title_contains=None) -> list[dict]:
        rows = []
        for r in self.catalog:
            if document_type and r["document_type"] != document_type:
                continue
            if department and r["department"] != department:
                continue
            if service and (r.get("service") or "").lower() != str(service).lower():
                continue
            if severity and r.get("severity") != severity:
                continue
            if date_from and r["created_date"] < date_from:
                continue
            if date_to and r["created_date"] > date_to:
                continue
            if title_contains and str(title_contains).lower() not in r["title"].lower():
                continue
            rows.append(dict(r))
        return rows

    def search(self, query, doc_ids=None, top_k=8) -> list[dict]:
        filters = SearchFilters(doc_ids=list(doc_ids) if doc_ids else None)
        future = asyncio.run_coroutine_threadsafe(
            self.retriever.search(str(query), self.role, filters, top_k=min(int(top_k), 20)), self.loop)
        result = future.result(timeout=self.timeout)
        self.degraded = self.degraded or result.degraded
        return [h.to_dict() for h in result.hits]

    def sections(self, doc_id, names) -> list[dict]:
        if doc_id not in self.allowed_ids:  # role filter applies to direct access too
            return []
        wanted = [str(n).lower() for n in names]
        return [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title, "section": c.section,
             "text": c.text, "document_type": c.document_type, "created_date": c.created_date,
             "service": c.service, "severity": c.severity}
            for c in self.retriever.chunks.values()
            if c.doc_id == doc_id and any(w in c.section.lower() for w in wanted)
        ]

    @staticmethod
    def batch(items, size) -> list[list]:
        size = max(1, int(size))
        items = list(items)
        return [items[i:i + size] for i in range(0, len(items), size)]

    def functions(self) -> dict[str, Any]:
        return {"catalog_filter": self.catalog_filter, "search": self.search,
                "sections": self.sections, "batch": self.batch}


def _normalise_batches(raw: Any, max_chunks: int, batch_size: int) -> list[list[dict]]:
    """Plans are model-written: re-check shape, dedupe, sanitise and cap before use."""
    if not isinstance(raw, list):
        raise SandboxError("plan must assign `batches` as a list of lists")
    seen, chunks = set(), []
    for group in raw:
        for item in group if isinstance(group, list) else [group]:
            if isinstance(item, dict) and item.get("chunk_id") and item["chunk_id"] not in seen:
                seen.add(item["chunk_id"])
                text, _ = sanitize_retrieved(str(item.get("text", "")))
                chunks.append({**item, "text": text})
    chunks = chunks[:max_chunks]
    return [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]


# --- nodes ----------------------------------------------------------------------------------
@node("research.explore", "research")
async def explore(state: ResearchState) -> dict:
    f = state.get("filters") or {}
    rows = role_catalog(state["role"])
    if f.get("document_types"):
        rows = [r for r in rows if r["document_type"] in f["document_types"]]
    dates = sorted(r["created_date"] for r in rows)
    stats = {
        "documents_visible": len(rows),
        "by_type": dict(Counter(r["document_type"] for r in rows)),
        "by_service": dict(Counter(r.get("service") for r in rows if r.get("service"))),
        "date_range": [dates[0], dates[-1]] if dates else [],
    }
    emit("research", phase="explore", depth=state.get("depth", 0), question=state["question"], **stats)
    return {"stats": stats}


@node("research.plan", "research")
async def plan(state: ResearchState) -> dict:
    settings = get_settings()
    prompt = (f"Question: {state['question']}\nFilters from the supervisor: {state.get('filters') or {}}\n"
              f"Collection statistics: {state['stats']}\n"
              f"Variables available: question, filters, MAX_DOCS={settings.rlm_max_docs}, "
              f"BATCH_SIZE={settings.rlm_batch_size}.")
    code, source, rationale = DEFAULT_PLAN, "default", "deterministic fallback plan"
    try:
        result = await get_gateway().structured(SearchPlan, prompts.RESEARCH_PLANNER, prompt,
                                                run_name="research_plan")
        validate_code(result.code)
        code, source, rationale = result.code, "llm", result.rationale
    except LLMUnavailableError as exc:
        log.info("research_plan_fallback", reason=str(exc))
    except SandboxError as exc:
        emit("research", phase="plan_rejected", reason=str(exc))
        rationale = f"LLM plan rejected by sandbox ({exc}); using default plan"
    emit("research", phase="plan", depth=state.get("depth", 0), source=source,
         rationale=rationale, code=code)
    return {"plan_code": code, "plan_source": source}


@node("research.execute_plan", "research")
async def execute_plan(state: ResearchState) -> dict:
    settings = get_settings()
    dsl = PlanDSL(state["role"], asyncio.get_running_loop())
    f = state.get("filters") or {}
    injected = {**dsl.functions(), "question": state["question"],
                "filters": {"document_type": (f.get("document_types") or [None])[0],
                            "date_from": f.get("date_from"), "date_to": f.get("date_to")},
                "MAX_DOCS": settings.rlm_max_docs, "BATCH_SIZE": settings.rlm_batch_size}

    async def run(code: str) -> dict:
        return await asyncio.to_thread(run_sandboxed, code, injected)

    errors: list[str] = []
    code = state["plan_code"]
    try:
        env = await run(code)
        batches = _normalise_batches(env.get("batches"), dsl.max_chunks, settings.rlm_batch_size)
        if not batches:
            raise SandboxError("plan produced no evidence")
        focus = str(env.get("focus") or state["question"])[:300]
    except SandboxError as exc:
        if code == DEFAULT_PLAN:
            raise
        errors.append(f"plan failed at runtime: {exc}; reran default plan")
        emit("research", phase="plan_rejected", reason=str(exc))
        env = await run(DEFAULT_PLAN)
        batches = _normalise_batches(env.get("batches"), dsl.max_chunks, settings.rlm_batch_size)
        focus = str(env.get("focus"))

    evidence = [c for b in batches for c in b]
    emit("research", phase="execute", depth=state.get("depth", 0), batches=len(batches),
         chunks=len(evidence), documents=sorted({c["doc_id"] for c in evidence}),
         retrieval_degraded=dsl.degraded)
    return {"batches": batches, "focus": focus, "evidence": evidence, "errors": errors}


def fan_out(state: ResearchState) -> list[Send] | str:
    if not state.get("batches"):
        return "aggregate"
    return [Send("analyze_batch", {"batch": b, "batch_index": i, "focus": state["focus"],
                                   "question": state["question"], "depth": state.get("depth", 0)})
            for i, b in enumerate(state["batches"])]


def _heuristic_findings(batch: list[dict]) -> list[dict]:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in batch:
        by_doc[c["doc_id"]].append(c)
    out = []
    for doc_id, chunks in by_doc.items():
        best = next((c for c in chunks if "root cause" in c["section"].lower()), chunks[0])
        body = best["text"].split(":", 1)[-1].strip()
        first_sentence = re.split(r"(?<=[.!?])\s", body)[0][:250]
        out.append({"doc_id": doc_id, "finding": first_sentence,
                    "theme": heuristic_theme(" ".join(c["text"] for c in chunks)),
                    "chunk_ids": [best["chunk_id"]]})
    return out


@node("research.analyze_batch", "research")
async def analyze_batch(payload: dict) -> dict:
    batch, index = payload["batch"], payload["batch_index"]
    excerpts = "\n\n".join(f'<document id="{c["chunk_id"]}" doc="{c["doc_id"]}">\n{c["text"]}\n</document>'
                           for c in batch)
    source = "llm"
    try:
        result = await get_gateway().structured(
            BatchFindings, prompts.BATCH_ANALYST,
            f"Question: {payload['question']}\nFocus: {payload['focus']}\n\n{excerpts}",
            tier="fast", run_name=f"analyze_batch_{index}")
        valid_ids = {c["chunk_id"] for c in batch}
        findings = [{**f.model_dump(), "chunk_ids": [i for i in f.chunk_ids if i in valid_ids]}
                    for f in result.findings]
    except LLMUnavailableError:
        source, findings = "heuristic", _heuristic_findings(batch)
    except Exception as exc:  # one bad batch must not sink the others
        emit("research", phase="analyze_batch", index=index, status="failed", error=type(exc).__name__)
        return {"findings": [], "errors": [f"batch {index} failed: {type(exc).__name__}"]}
    emit("research", phase="analyze_batch", depth=payload.get("depth", 0), index=index, source=source,
         documents=sorted({c["doc_id"] for c in batch}), findings=len(findings))
    return {"findings": findings}


def count_themes(findings: list[dict]) -> list[dict]:
    docs_by_theme: dict[str, set[str]] = defaultdict(set)
    for f in findings:
        docs_by_theme[f["theme"].strip().lower()].add(f["doc_id"])
    ranked = sorted(docs_by_theme.items(), key=lambda kv: len(kv[1]), reverse=True)
    return [{"label": label, "count": len(docs), "doc_ids": sorted(docs)} for label, docs in ranked]


@node("research.aggregate", "research")
async def aggregate(state: ResearchState) -> dict:
    findings = state.get("findings") or []
    counted = count_themes(findings)
    try:
        lines = "\n".join(f"- {f['doc_id']} [{f['theme']}]: {f['finding']} (chunks: {', '.join(f['chunk_ids'])})"
                          for f in findings)
        result = await get_gateway().structured(
            Synthesis, prompts.AGGREGATOR,
            f"Question: {state['question']}\nTheme counts (computed): {counted}\nFindings:\n{lines}",
            run_name="research_aggregate")
        summary, themes = result.summary, [t.model_dump() for t in result.themes] or counted
        follow_ups = result.follow_ups[:2]
    except LLMUnavailableError:
        top = ", ".join(f"{t['label']} ({t['count']})" for t in counted[:4])
        summary = (f"Analysed {len({f['doc_id'] for f in findings})} documents in "
                   f"{len(state.get('batches') or [])} batches. Most frequent themes: {top or 'none found'}.")
        themes, follow_ups = counted, []
    settings = get_settings()
    depth = state.get("depth", 0)
    if depth + 1 >= settings.rlm_max_depth:
        follow_ups = []
    emit("research", phase="aggregate", depth=depth, themes=themes[:6], follow_ups=follow_ups,
         findings=len(findings), errors=state.get("errors") or [])
    return {"synthesis": summary, "themes": themes, "follow_ups": follow_ups}


@node("research.recurse", "research")
async def recurse(state: ResearchState) -> dict:
    depth = state.get("depth", 0) + 1
    emit("research", phase="recurse", depth=depth, follow_ups=state["follow_ups"])
    children = await asyncio.gather(
        *[research_graph.ainvoke({"question": q, "role": state["role"], "filters": state.get("filters") or {},
                                  "depth": depth})
          for q in state["follow_ups"]],
        return_exceptions=True,
    )
    sub_results, evidence, errors = [], list(state.get("evidence") or []), []
    for q, child in zip(state["follow_ups"], children):
        if isinstance(child, Exception):
            errors.append(f"sub-research '{q}' failed: {type(child).__name__}")
            continue
        sub_results.append({"question": q, "synthesis": child.get("synthesis", ""),
                            "themes": child.get("themes", [])[:4]})
        evidence.extend(child.get("evidence") or [])
    return {"sub_results": sub_results, "evidence": evidence, "errors": errors}


def after_aggregate(state: ResearchState) -> str:
    return "recurse" if state.get("follow_ups") else END


def build_research_graph():
    g = StateGraph(ResearchState)
    g.add_node("explore", explore)
    g.add_node("plan", plan)
    g.add_node("execute_plan", execute_plan)
    g.add_node("analyze_batch", analyze_batch)
    g.add_node("aggregate", aggregate)
    g.add_node("recurse", recurse)
    g.add_edge(START, "explore")
    g.add_edge("explore", "plan")
    g.add_edge("plan", "execute_plan")
    g.add_conditional_edges("execute_plan", fan_out, ["analyze_batch", "aggregate"])
    g.add_edge("analyze_batch", "aggregate")
    g.add_conditional_edges("aggregate", after_aggregate, ["recurse", END])
    g.add_edge("recurse", END)
    return g.compile(name="research_agent")


research_graph = build_research_graph()
