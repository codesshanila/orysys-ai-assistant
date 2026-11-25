"""Streamlit frontend: chat on the left, live agent activity on the right.

Deliberately thin: all logic lives in the backend. The UI only
  * logs in and keeps the JWT,
  * streams /chat/stream (Server-Sent Events) and renders two things from it:
      - answer tokens into the chat bubble (then the answer with its sources),
      - every other event into the Agent Activity panel: a pipeline stepper, the supervisor's
        plan, retrieval scores, research batches and themes, tool calls, safety checks, memory,
  * shows a readable approval card when a state-changing tool is waiting for a human,
  * sends thumbs up/down feedback for the turn's LangSmith run,
  * lets administrators toggle failure injection to demonstrate graceful degradation.

Run:  streamlit run frontend/streamlit_app.py   (theme in .streamlit/config.toml)
"""
from __future__ import annotations

import html
import json
import os
import re
import uuid

import httpx
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Commercial Bank Knowledge Assistant", page_icon="🏦", layout="wide")

ss = st.session_state
ss.setdefault("token", None)
ss.setdefault("user", None)
ss.setdefault("tools", [])
ss.setdefault("session_id", uuid.uuid4().hex[:12])
ss.setdefault("messages", [])      # [{role, content, run_id?, sources?, degraded?}]
ss.setdefault("events", [])        # activity events of the latest turn
ss.setdefault("pending_approval", None)
ss.setdefault("queued_prompt", None)
ss.setdefault("feedback_sent", set())

CSS = """
<style>
#MainMenu, [data-testid="stAppDeployButton"], footer {display: none !important;}
/* Streamlit fades elements from the previous run while a long turn streams; keep the page crisp. */
.stale-element, [data-stale="true"] {opacity: 1 !important; filter: none !important; transition: none !important;}[data-testid="stHeader"] {background: transparent;}
.block-container, [data-testid="stMainBlockContainer"] {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1500px;}
.cb-header {display:flex; align-items:center; justify-content:space-between; background:#0B2545; color:#fff;
  border-radius:12px; padding:14px 20px; margin-bottom:14px;}
.cb-brand {display:flex; align-items:center; gap:12px;}
.cb-mark {width:38px; height:38px; border-radius:9px; background:linear-gradient(135deg,#5FA8D3,#1B6CA8);
  display:flex; align-items:center; justify-content:center; font-weight:700; letter-spacing:.5px;}
.cb-title {font-size:1.15rem; font-weight:650; line-height:1.2;}
.cb-sub {font-size:.78rem; opacity:.75;}
.cb-right {display:flex; gap:8px; align-items:center; font-size:.8rem;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600;
  line-height:1.6; white-space:nowrap;}
.pill-viewer {background:#E6F4EA; color:#1E6B34;} .pill-analyst {background:#E3F0FB; color:#0F4C81;}
.pill-admin {background:#FDECEA; color:#A1281B;}
.pill-ghost {background:rgba(255,255,255,.12); color:#E6EDF6;}
.acc-public {background:#E6F4EA; color:#1E6B34;} .acc-internal {background:#E3F0FB; color:#0F4C81;}
.acc-confidential {background:#FFF3DC; color:#8A5300;} .acc-restricted {background:#FDECEA; color:#A1281B;}
.st-ok {background:#E6F4EA; color:#1E6B34;} .st-warn {background:#FFF3DC; color:#8A5300;}
.st-bad {background:#FDECEA; color:#A1281B;} .st-info {background:#E3F0FB; color:#0F4C81;}
.st-idle {background:#EEF1F5; color:#5B6675;}
.cite {display:inline-block; font-size:.7rem; font-weight:600; color:#0F4C81; background:#E3F0FB;
  border-radius:6px; padding:0 6px; margin:0 1px; vertical-align:1px; font-family:ui-monospace,monospace;}
.steps {display:flex; flex-wrap:wrap; gap:6px 4px; align-items:center; margin:4px 0 10px;}
.step {display:flex; align-items:center; gap:5px; padding:4px 9px; border-radius:8px; font-size:.74rem;
  border:1px solid #DDE3EC; background:#fff; color:#5B6675;}
.step .ms {opacity:.65; font-size:.68rem;}
.step.done {border-color:#B7DFC2; background:#F2FAF4; color:#1E6B34;}
.step.active {border-color:#7FB3E0; background:#EAF3FC; color:#0F4C81; font-weight:600;
  animation:pulse 1.2s ease-in-out infinite;}
.step.paused {border-color:#F0C674; background:#FFF8E8; color:#8A5300; font-weight:600;}
.step.error {border-color:#EBA59D; background:#FDF1EF; color:#A1281B;}
.step.skipped {opacity:.45; border-style:dashed;}
.arrow {color:#A9B4C2; font-size:.7rem;}
@keyframes pulse {0%,100% {box-shadow:0 0 0 0 rgba(15,76,129,.25);} 50% {box-shadow:0 0 0 4px rgba(15,76,129,0);}}
.sec {font-size:.72rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:#5B6675;
  margin:14px 0 6px;}
.card {background:#fff; border:1px solid #DDE3EC; border-radius:10px; padding:10px 12px; margin-bottom:8px;
  font-size:.84rem;}
.card .k {color:#5B6675; font-size:.74rem;}
.row {display:flex; gap:8px; flex-wrap:wrap; align-items:center;}
.batches {display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:6px; margin-bottom:8px;}
.batch {border:1px solid #DDE3EC; border-radius:8px; padding:6px 8px; background:#fff; font-size:.74rem;}
.batch b {font-size:.8rem;}
.theme {margin:4px 0;}
.theme .bar {height:8px; border-radius:4px; background:linear-gradient(90deg,#1B6CA8,#5FA8D3);}
.theme .lbl {display:flex; justify-content:space-between; font-size:.78rem; margin-bottom:2px;}
.tool {border-left:4px solid #DDE3EC;} .tool.ok {border-left-color:#3C9D5D;}
.tool.error, .tool.denied {border-left-color:#C8473A;} .tool.rejected {border-left-color:#D99A22;}
.mono {font-family:ui-monospace,monospace; font-size:.76rem;}
.st-key-approval {background:#FFFBF0; border:1px solid #F0C674 !important; border-radius:12px;}
[data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2, [data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {font-size:1.02rem !important; font-weight:700; margin:.7rem 0 .3rem !important;
  padding:0 !important;}
.now {font-size:.8rem; color:#0F4C81; background:#EAF3FC; border-radius:8px; padding:6px 10px; margin-bottom:6px;}
.panel {font-size:.84rem; color:#1B2430;}
/* Keep the activity panel in view while the chat scrolls. */
[data-testid="stHorizontalBlock"]:has(.st-key-activity) {align-items: flex-start;}
[data-testid="stColumn"]:has(.st-key-activity) {position: sticky; top: 12px; align-self: flex-start;}
details.code {background:#fff; border:1px solid #DDE3EC; border-radius:10px; padding:8px 12px; margin-bottom:8px;}
details.code summary {cursor:pointer; color:#0F4C81; font-size:.8rem;}
details.code pre {background:#F6F8FB; border-radius:6px; padding:8px; margin:8px 0 0; font-size:.72rem;
  white-space:pre-wrap; max-height:320px; overflow:auto;}
details.more summary {cursor:pointer; color:#0F4C81; font-size:.76rem; margin-top:6px;}
details.more div {margin-top:4px;}
.q {padding:5px 0; border-top:1px solid #EEF1F5;} .q:first-of-type {border-top:none;}
.qt {white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom:2px;}
table.scores {width:100%; border-collapse:collapse; font-size:.74rem;}
table.scores th {text-align:left; color:#5B6675; font-weight:600; padding:3px 4px; border-bottom:1px solid #DDE3EC;}
table.scores td {padding:3px 4px; border-bottom:1px solid #F0F3F7;}
table.scores td.c {font-family:ui-monospace,monospace; white-space:nowrap;}
table.scores td.n {text-align:right; color:#5B6675;}
table.scores td.b {white-space:nowrap;} table.scores td.b span {font-size:.7rem; margin-left:4px;}
.sb {display:inline-block; width:46px; height:6px; background:#EEF1F5; border-radius:3px; vertical-align:middle;}
.sb i {display:block; height:6px; border-radius:3px; background:linear-gradient(90deg,#1B6CA8,#5FA8D3);}

/* ---- Responsive: keep everything readable when the window is not maximised ---- */
[data-testid="InputInstructions"] {display:none !important;}
input[type="password"]::-ms-reveal, input[type="password"]::-ms-clear {display:none;}
.cb-title {white-space:nowrap;}
.cb-brand > div {min-width:0;}
.cb-sub {white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.card {overflow:hidden;}
.card .pill {max-width:100%; overflow:hidden; text-overflow:ellipsis; vertical-align:middle;}
/* The panel adapts to its own width, whatever the window and sidebar do. */
.st-key-activity {container-type:inline-size; height:min(920px, calc(100vh - 28px)) !important;}
@container (max-width: 470px) {
  .sb {width:30px;}
  table.scores td.b span {margin-left:3px;}
}
@container (max-width: 400px) {
  table.scores th:nth-child(2), table.scores th:nth-child(3),
  table.scores td:nth-child(2), table.scores td:nth-child(3) {display:none;}  /* keep chunk, hybrid, rerank */
  .step {font-size:.68rem; padding:3px 7px;} .step .ms {display:none;}
  .batches {grid-template-columns:1fr 1fr;}
  .card {padding:8px 10px;}
  .pill {font-size:.7rem; padding:1px 8px;}
}
@container (max-width: 300px) {
  .sb {display:none;}
  .batches {grid-template-columns:1fr;}
}
@media (max-width: 1280px) {
  .cb-right .pill-ghost {display:none;}          /* tech badges go first; the role badge stays */
  section[data-testid="stSidebar"] {width:250px !important; min-width:250px !important;}
}
@media (max-width: 1100px) {
  .cb-header {padding:12px 14px;}
  .cb-title {font-size:1rem;} .cb-sub {font-size:.72rem;}
  .cb-mark {width:32px; height:32px; font-size:.85rem;}
}
/* Narrow windows: put the activity panel under the chat instead of squeezing both. */
@media (max-width: 980px) {
  [data-testid="stHorizontalBlock"]:has(.st-key-activity) {flex-direction:column !important;}
  [data-testid="stHorizontalBlock"]:has(.st-key-activity) > [data-testid="stColumn"] {width:100% !important; flex:1 1 100% !important;}
  [data-testid="stColumn"]:has(.st-key-activity) {position:static;}
  .st-key-activity {height:auto !important; max-height:none !important;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ROLE_LABEL = {"viewer": "Viewer", "analyst": "Analyst", "admin": "Administrator"}
NODE_LABELS = {
    "input_guard": "Guardrails", "memory_load": "Memory", "supervisor": "Supervisor",
    "retrieval": "Retrieval", "research": "Research (RLM)", "tools": "Tools",
    "tool_planner": "Tools: plan", "tool_executor": "Tools: execute", "response": "Response",
    "validator": "Validation", "memory_save": "Save",
}
TOOL_LABELS = {
    "knowledge_search": "Knowledge search", "python_analysis": "Python analysis",
    "mcp_employee_lookup": "Employee directory (MCP)", "mcp_service_catalog": "Service catalog (MCP)",
    "mcp_incident_records": "Incident records (MCP)", "mcp_update_incident_status": "Update incident status (MCP)",
    "deep_research": "Deep research", "admin_chaos_control": "Failure injection",
}
SUGGESTIONS = {
    "viewer": ["What caused the Mobile Payments API outage on 23 May 2026?", "Who is on call?",
               "What is the password policy for internal systems?"],
    "analyst": ["What does the runbook say to do when payment switch certificates are close to expiry, "
                "and who is the SRE lead?",
                "Summarize all outage reports related to payment failures during the last year and identify "
                "recurring root causes.",
                "What was agreed in the payment gateway vendor meeting?"],
    "admin": ["Resolve INC-2026-016, the fix is deployed",
              "What does the runbook say to do when payment switch certificates are close to expiry, "
              "and who is the SRE lead?"],
}
CITE_RE = re.compile(r"\[((?:[A-Z]{2,5}-\d{4}-\d{3}|[A-Z]{2,5}-\d{3})#\d+|tool:[a-z_]+)\]")


def api(method: str, path: str, **kwargs) -> httpx.Response:
    headers = {"Authorization": f"Bearer {ss.token}"} if ss.token else {}
    return httpx.request(method, f"{BACKEND}{path}", headers=headers, timeout=30, **kwargs)


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def pill(text: str, cls: str) -> str:
    return f'<span class="pill {cls}">{esc(text)}</span>'


# --- answer rendering ------------------------------------------------------------------------------
def format_answer(text: str) -> str:
    """Escape model output (it is untrusted), then style citations as small badges."""
    safe = html.escape(text, quote=False)

    def badge(m: re.Match) -> str:
        ref = m.group(1)
        label = TOOL_LABELS.get(ref[5:], ref[5:]) if ref.startswith("tool:") else ref
        return f'<span class="cite">{esc(label)}</span>'

    return CITE_RE.sub(badge, safe)


def humanise_degraded(flag: str) -> str:
    source, _, reason = flag.partition(":")
    names = {"vector_db": "Vector database", "llm": "AI model", "mcp": "Enterprise data service",
             "research": "Research agent"}
    source = source.strip()
    if source.startswith("tool "):
        name = TOOL_LABELS.get(source[5:], source[5:])
    else:
        name = names.get(source, source.capitalize())
    reason = re.sub(r"^\w+Error:\s*", "", reason.strip()).rstrip(".")
    if "chaos" in reason or "injected" in reason:
        reason = "simulated outage"
    reason = reason.replace("(", "").replace(")", "")
    return f"{name} unavailable{(' (' + reason + ')') if reason else ''}"


def plain(text: str) -> str:
    """Passages are stored as Markdown; show them as readable plain text in the source cards."""
    text = re.sub(r"\*\*|__|`", "", text)
    return re.sub(r"\s*\|\s*", " · ", text).strip(" ·")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    docs: dict[str, list[dict]] = {}
    for s in sources:
        docs.setdefault(s["doc_id"], []).append(s)
    with st.expander(f"Sources · {len(docs)} document{'s' if len(docs) != 1 else ''}"):
        for doc_id, chunks in docs.items():
            first = chunks[0]
            meta = " · ".join(x for x in (str(first.get("document_type") or "").replace("_", " "),
                                          str(first.get("created_date") or "")) if x)
            st.markdown(
                f'<div class="row"><b>{esc(doc_id)}</b> {esc(first.get("title") or "")} '
                f'{pill(first.get("access_level") or "", "acc-" + str(first.get("access_level")))}'
                f'<span class="k" style="color:#5B6675;font-size:.75rem">{esc(meta)}</span></div>',
                unsafe_allow_html=True)
            for c in chunks:
                st.markdown(f'<div class="card"><span class="cite">{esc(c["chunk_id"])}</span> '
                            f'{esc(plain(c.get("passage") or ""))}</div>', unsafe_allow_html=True)


def render_message(i: int, msg: dict) -> None:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🏦"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            return
        if msg.get("degraded"):
            notes = list(dict.fromkeys(humanise_degraded(d) for d in msg["degraded"]))
            st.warning("Degraded mode: " + "; ".join(notes) + ". The answer uses fallbacks.", icon="⚠️")
        st.markdown(format_answer(msg["content"]), unsafe_allow_html=True)
        render_sources(msg.get("sources") or [])
        if msg.get("run_id"):
            choice = st.feedback("thumbs", key=f"fb-{msg['run_id']}")
            if choice is not None and msg["run_id"] not in ss.feedback_sent:
                api("POST", "/feedback", json={"run_id": msg["run_id"], "score": int(choice)})
                ss.feedback_sent.add(msg["run_id"])
                st.toast("Feedback sent to LangSmith", icon="✅")


# --- activity panel --------------------------------------------------------------------------------
def node_status(events: list[dict]) -> dict[str, dict]:
    status: dict[str, dict] = {}
    for e in events:
        n = e.get("node")
        if not n or n.startswith("research."):
            continue
        key = "tools" if n in ("tool_planner", "tool_executor") else n
        cur = status.setdefault(key, {"state": "pending", "ms": 0})
        if e["type"] == "node_start":
            cur["state"] = "active"
        elif e["type"] == "node_end":
            cur["state"], cur["ms"] = "done", cur["ms"] + int(e.get("elapsed_ms") or 0)
        elif e["type"] == "node_paused":
            cur["state"] = "paused"
        elif e["type"] == "node_error":
            cur["state"] = "error"
    return status


def stepper_html(events: list[dict], running: bool) -> str:
    sup = next((e for e in events if e["type"] == "agent_state" and e.get("agent") == "supervisor"), None)
    plan = [("tools" if p == "tools" else p) for p in (sup or {}).get("plan", [])]
    steps = ["input_guard", "memory_load", "supervisor", *plan, "response", "validator", "memory_save"]
    status = node_status(events)
    blocked = any(e["type"] == "validation" and e.get("stage") == "input" and not e.get("allowed", True)
                  for e in events)
    parts = []
    for i, s in enumerate(steps):
        st_ = status.get(s, {"state": "pending", "ms": 0})
        state = st_["state"]
        if state == "active" and not running:
            state = "done"
        if blocked and s not in ("input_guard", "memory_save"):
            state = "skipped"
        icon = {"done": "✓", "active": "●", "paused": "⏸", "error": "✗", "skipped": "–"}.get(state, "○")
        ms = f'<span class="ms">{st_["ms"]} ms</span>' if st_["ms"] and state == "done" else ""
        parts.append(f'<span class="step {state}">{icon} {esc(NODE_LABELS.get(s, s))} {ms}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="arrow">›</span>')
    return '<div class="steps">' + "".join(parts) + "</div>"


def access_from_filter(flt: dict) -> list[str]:
    for clause in (flt or {}).get("$and", [flt or {}]):
        if "access_level" in clause:
            return clause["access_level"].get("$in", [])
    return []


_PANEL: list[str] = []


def put(html_text: str, **_ignored) -> None:
    """Collect panel HTML; render_activity outputs it in one go."""
    _PANEL.append(html_text)


def section(title: str) -> None:
    put(f'<div class="sec">{esc(title)}</div>', unsafe_allow_html=True)


def render_supervisor(events: list[dict]) -> None:
    sup = next((e for e in events if e["type"] == "agent_state" and e.get("agent") == "supervisor"), None)
    if not sup:
        return
    section("Supervisor plan")
    plan = " → ".join(NODE_LABELS.get(p, p) for p in sup.get("plan", [])) or "answer directly"
    filters = sup.get("filters") or {}
    flt = " ".join(pill(f"{k}: {', '.join(v) if isinstance(v, list) else v}", "st-info") for k, v in filters.items())
    put(
        f'<div class="card"><div class="row">{pill(sup.get("intent", "").replace("_", " "), "st-info")} '
        f'<b>{esc(plan)}</b></div>'
        + (f'<div class="row" style="margin-top:6px">{flt}</div>' if flt else "")
        + f'<details class="more"><summary>Why this plan · rewritten query</summary>'
        f'<div>{esc(sup.get("reason", ""))}</div><div class="k" style="margin-top:4px">Rewritten query</div>'
        f'<div>{esc(sup.get("standalone_query", ""))}</div></details></div>',
        unsafe_allow_html=True)


def render_retrieval(events: list[dict]) -> None:
    done = [e for e in events if e["type"] == "retrieval" and e.get("status") == "done"]
    if not done:
        return
    section("Retrieval · hybrid search")
    access = access_from_filter(done[0].get("filter") or {})
    rows = []
    for e in done:
        flag = pill("BM25 fallback", "st-warn") if e.get("degraded") else pill(
            "reranked" if e.get("reranked") else "fused", "st-ok")
        rows.append(f'<div class="q"><div class="qt">{esc(e["query"])}</div>'
                    f'<div class="row">{flag}<span class="k">{e["hits"]} hits · {e.get("elapsed_ms", 0)} ms · '
                    f'{esc(", ".join(e.get("namespaces") or []))}</span></div></div>')
    acc = " ".join(pill(a, "acc-" + a) for a in access)
    put(f'<div class="card"><div class="row"><span class="k">Role access filter</span>{acc}</div>'
                + "".join(rows) + "</div>", unsafe_allow_html=True)
    best: dict[str, dict] = {}
    for e in done:
        for r in e.get("top") or []:
            if r["id"] not in best or (r.get("rerank") or 0) > (best[r["id"]].get("rerank") or 0):
                best[r["id"]] = r
    table = sorted(best.values(), key=lambda r: (r.get("rerank") is not None, r.get("rerank") or r["hybrid"]),
                   reverse=True)[:8]
    if table:
        def num(v) -> str:
            return "–" if v is None else f"{v:.2f}"

        def bar(v) -> str:
            if v is None:
                return '<td class="n">–</td>'
            w = max(2, min(100, int(round(float(v) * 100))))
            return f'<td class="b"><div class="sb"><i style="width:{w}%"></i></div><span>{v:.2f}</span></td>'

        body = "".join(
            f'<tr title="{esc(r["title"])}"><td class="c">{esc(r["id"])}</td><td class="n">{num(r["dense"])}</td>'
            f'<td class="n">{num(r["sparse"])}</td>{bar(r["hybrid"])}{bar(r.get("rerank"))}</tr>' for r in table)
        put('<div class="card" style="padding:6px 8px"><table class="scores"><thead><tr><th>Chunk</th>'
                    '<th>Dense</th><th>Sparse</th><th>Hybrid</th><th>Rerank</th></tr></thead><tbody>'
                    f'{body}</tbody></table><div class="k" style="margin-top:4px">hybrid = 0.6 · dense + 0.4 · '
                    'sparse, then cross-encoder rerank · hover a row for the document</div></div>',
                    unsafe_allow_html=True)


def render_research(events: list[dict]) -> None:
    rs = [e for e in events if e["type"] == "research"]
    if not rs:
        return
    section("Research agent (RLM)")
    for depth in sorted({e.get("depth", 0) for e in rs if "depth" in e}):
        at = [e for e in rs if e.get("depth", 0) == depth]
        explore = next((e for e in at if e["phase"] == "explore"), None)
        plan = next((e for e in at if e["phase"] == "plan"), None)
        execute = next((e for e in at if e["phase"] == "execute"), None)
        batches = [e for e in at if e["phase"] == "analyze_batch"]
        agg = next((e for e in at if e["phase"] == "aggregate"), None)
        title = "Top-level question" if depth == 0 else f"Recursion · depth {depth}"
        q = esc((explore or {}).get("question", ""))
        stats = []
        if explore:
            stats.append(f"{explore.get('documents_visible', 0)} documents visible")
        if execute:
            stats.append(f"{execute['chunks']} sections pulled from {len(execute['documents'])} documents")
            stats.append(f"{execute['batches']} parallel batches")
        put(f'<div class="card"><div class="row"><b>{title}</b>'
                    + (pill("LLM plan" if plan and plan.get("source") == "llm" else "default plan", "st-info")
                       if plan else "")
                    + f'</div><div style="margin:4px 0">{q}</div><div class="k">{esc(" · ".join(stats))}</div>'
                    + (f'<div style="margin-top:4px">{esc(plan.get("rationale", ""))}</div>' if plan else "")
                    + "</div>", unsafe_allow_html=True)
        if plan and plan.get("code"):
            put(f'<details class="code"><summary>Generated Python search plan (depth {depth})</summary>'
                f'<pre>{esc(plan["code"])}</pre></details>')
        if batches:
            cells = []
            for b in sorted(batches, key=lambda x: x.get("index", 0)):
                failed = b.get("status") == "failed"
                cls = "st-bad" if failed else ("st-ok" if b.get("source") == "llm" else "st-warn")
                cells.append(f'<div class="batch"><b>Batch {b.get("index", 0)}</b> '
                             f'{pill("failed" if failed else b.get("source", "done"), cls)}<br>'
                             f'{esc(", ".join(b.get("documents") or []))}<br>'
                             f'<span class="k">{b.get("findings", 0)} findings</span></div>')
            put('<div class="batches">' + "".join(cells) + "</div>", unsafe_allow_html=True)
        if agg and agg.get("themes"):
            top = max(t["count"] for t in agg["themes"]) or 1
            bars = "".join(
                f'<div class="theme"><div class="lbl"><span>{esc(t["label"])}</span>'
                f'<b>{t["count"]}</b></div><div class="bar" style="width:{max(6, int(100 * t["count"] / top))}%">'
                f"</div></div>" for t in agg["themes"])
            fu = ("<div class='k' style='margin-top:6px'>Follow-up: " + esc("; ".join(agg["follow_ups"])) + "</div>"
                  if agg.get("follow_ups") else "")
            put(f'<div class="card"><div class="k">Recurring themes (counted from batch findings)</div>'
                        f"{bars}{fu}</div>", unsafe_allow_html=True)


def fmt_args(args: dict) -> str:
    return " · ".join(f"{k.replace('_', ' ')}: {v}" for k, v in (args or {}).items() if v not in (None, "", []))[:180]


def tool_summary(tool: str, preview: str) -> str:
    """One readable line from the (truncated JSON) tool output preview."""
    if not preview:
        return ""
    if tool == "mcp_employee_lookup":
        people = re.findall(r'"name":\s*"([^"]+)"[^}]*?"(?:title|role)":\s*"([^"]+)"', preview)
        return "Found: " + "; ".join(f"{n} ({t})" for n, t in people[:3]) if people else ""
    if tool == "mcp_incident_records":
        ids = re.findall(r'"incident_id":\s*"([^"]+)"', preview)
        return f"{len(ids)}+ incident record(s): " + ", ".join(ids[:4]) if ids else ""
    if tool == "mcp_update_incident_status":
        new = re.search(r'"status":\s*"([^"]+)"', preview)
        return f"Status is now {new.group(1)}" if new else ""
    if tool == "mcp_service_catalog":
        svc = re.findall(r'"service":\s*"([^"]+)"', preview)
        owner = re.search(r'"owner(?:_team)?":\s*"([^"]+)"', preview)
        return ", ".join(svc[:3]) + (f" · owner {owner.group(1)}" if owner else "") if svc else ""
    if tool == "knowledge_search":
        ids = re.findall(r'"chunk_id":\s*"([^"]+)"', preview)
        return f"{len(ids)}+ passages: " + ", ".join(ids[:3]) if ids else ""
    return ""


def render_tools(events: list[dict]) -> None:
    planned = next((e for e in reversed(events) if e["type"] == "agent_state" and e.get("agent") == "tools"), None)
    results = [e for e in events if e["type"] == "tool_call" and e.get("status") != "running"]
    if not planned and not results:
        return
    section("Tools")
    shown = set()
    for r in results:
        shown.add(r["tool"])
        cls = {"ok": "st-ok", "rejected": "st-warn"}.get(r["status"], "st-bad")
        detail = r.get("error") or tool_summary(r["tool"], r.get("output_preview") or "")
        put(f'<div class="card tool {esc(r["status"])}"><div class="row">'
                    f'<b>{esc(TOOL_LABELS.get(r["tool"], r["tool"]))}</b>{pill(r["status"], cls)}'
                    f'<span class="k">{r.get("elapsed_ms", 0)} ms</span></div>'
                    f'<div class="k" style="margin-top:3px">{esc(fmt_args(r.get("args", {})))}</div>'
                    + (f'<div style="margin-top:3px">{esc(detail)}</div>' if detail else "")
                    + "</div>", unsafe_allow_html=True)
    for c in (planned or {}).get("planned_calls", []):
        if c["name"] not in shown:
            put(f'<div class="card tool"><div class="row"><b>{esc(TOOL_LABELS.get(c["name"], c["name"]))}'
                        f'</b>{pill("planned", "st-idle")}</div><div class="k" style="margin-top:3px">'
                        f'{esc(fmt_args(c.get("args", {})))}</div></div>', unsafe_allow_html=True)
    decision = next((e for e in reversed(events) if e["type"] == "approval_result"), None)
    if decision:
        ok = (decision.get("decision") or {}).get("approved")
        put(f'<div class="row">{pill("human decision: " + ("approved" if ok else "rejected"), "st-ok" if ok else "st-warn")}</div>',
                    unsafe_allow_html=True)


def render_safety(events: list[dict]) -> None:
    vals = [e for e in events if e["type"] == "validation"]
    if not vals:
        return
    section("Safety & validation")
    chips = []
    for v in vals:
        stage = v.get("stage")
        if stage == "input":
            ok = v.get("allowed", True)
            chips.append(pill("input: allowed" if ok else "input: blocked", "st-ok" if ok else "st-bad"))
            if not ok:
                chips += [pill(f, "st-bad") for f in v.get("findings") or []]
        elif stage == "retrieved_content":
            chips.append(pill(f"injection neutralised in {len(v.get('findings') or [])} chunk(s)", "st-warn"))
        elif stage == "output":
            action = v.get("action", "")
            cls = {"accepted": "st-ok", "retry": "st-warn", "repaired": "st-warn"}.get(action, "st-info")
            chips.append(pill(f"output: {action} · {len(v.get('citations') or [])} citations verified", cls))
            for bad in v.get("invalid_citations") or []:
                chips.append(pill(f"invalid citation {bad}", "st-bad"))
            for issue in v.get("issues") or []:
                chips.append(pill(issue, "st-warn"))
    put('<div class="row">' + " ".join(chips) + "</div>", unsafe_allow_html=True)


def render_memory(events: list[dict]) -> None:
    mem = [e for e in events if e["type"] == "memory"]
    if not mem:
        return
    section("Memory")
    lines = []
    for m in mem:
        if m["action"] == "recalled":
            lines.append(f"Recalled {m.get('session_messages', 0)} message(s) from this session, "
                         f"{m.get('long_term_interactions', 0)} past interaction(s) for this user, "
                         f"{len(m.get('related') or [])} related.")
        elif m["action"] == "summarised":
            lines.append(f"Summarised {m.get('folded_messages', 0)} older messages to keep the context short.")
        elif m["action"] == "saved":
            lines.append("Saved this answer to long-term memory.")
        elif m["action"] == "skipped":
            lines.append("Not remembered: " + m.get("reason", ""))
    put('<div class="card">' + "<br>".join(esc(x) for x in lines) + "</div>", unsafe_allow_html=True)


def now_line(events: list[dict]) -> str:
    """Plain-language description of the latest step, shown while the turn is running."""
    for e in reversed(events):
        t = e["type"]
        if t == "research":
            p, d = e.get("phase"), e.get("depth", 0)
            where = "" if not d else f" (recursion depth {d})"
            return {
                "explore": f"exploring the catalog: {e.get('documents_visible', 0)} documents visible{where}",
                "plan": f"writing a Python search plan{where}",
                "execute": f"plan ran: {e.get('chunks', 0)} sections in {e.get('batches', 0)} batches{where}",
                "analyze_batch": f"analysing batches in parallel: batch {e.get('index', 0)} done{where}",
                "aggregate": f"aggregating themes across batches{where}",
                "recurse": "investigating a follow-up question recursively",
            }.get(p, "researching")
        if t == "retrieval":
            return "searching Pinecone (dense + sparse) and reranking" if e.get("status") == "started" \
                else f"retrieved {e.get('hits', 0)} passages"
        if t == "tool_call":
            return f"calling {TOOL_LABELS.get(e['tool'], e['tool'])}"
        if t == "agent_state" and e.get("agent") == "tools":
            return "choosing tools"
        if t == "answer_reset":
            return "writing the answer"
        if t == "validation" and e.get("stage") == "output":
            return "checking citations and brand rules"
        if t in ("node_paused", "approval_required"):
            return "waiting for your approval (see the card in the chat)"
        if t == "approval_result":
            return "running the approved action" if (e.get("decision") or {}).get("approved") else "recording the rejection"
        if t == "node_start":
            return {"input_guard": "checking the request", "memory_load": "loading memory",
                    "supervisor": "planning", "response": "writing the answer", "validator": "validating",
                    "memory_save": "saving to memory", "research": "starting the research agent",
                    "research.explore": "exploring the document catalog",
                    "research.plan": "writing a Python search plan",
                    "research.execute_plan": "running the search plan in the sandbox",
                    "research.analyze_batch": "analysing batches in parallel",
                    "research.aggregate": "aggregating themes across batches",
                    "research.recurse": "investigating a follow-up question recursively",
                    "tool_planner": "choosing tools", "tool_executor": "running tools",
                    "retrieval": "searching Pinecone (dense + sparse) and reranking",
                    }.get(e["node"], NODE_LABELS.get(e["node"], e["node"]))
    return "starting"


def render_activity(container, events: list[dict]) -> None:
    """Draw the whole panel as ONE html element, so a redraw never leaves stale cards behind."""
    _PANEL.clear()
    _build_activity(events)
    container.html('<div class="panel">' + "".join(_PANEL) + "</div>")


def _build_activity(events: list[dict]) -> None:
    done = any(e["type"] == "run_finished" for e in events)
    last = {e["type"]: i for i, e in enumerate(events)}
    # The stream ends (run_finished) when the graph pauses, so "waiting" means the latest
    # approval request has no decision after it yet.
    waiting = last.get("approval_required", -1) > last.get("approval_result", -1)
    errored = any(e["type"] in ("error", "node_error") for e in events)
    final = next((e for e in reversed(events) if e["type"] == "final"), None)
    degraded = bool(final and final.get("degraded"))
    blocked = any(e["type"] == "validation" and e.get("stage") == "input" and not e.get("allowed", True)
                  for e in events)
    if blocked:
        state = pill("⛔ blocked by guardrails", "st-bad")
    elif waiting:
        state = pill("⏸ waiting for approval", "st-warn")
    elif errored:
        state = pill("error", "st-bad")
    elif done:
        state = pill("finished · degraded" if degraded else "finished", "st-warn" if degraded else "st-ok")
    elif events:
        state = pill("● running", "st-info")
    else:
        state = pill("idle", "st-idle")
    total = sum(int(e.get("elapsed_ms") or 0) for e in events
                if e["type"] == "node_end" and not str(e.get("node", "")).startswith("research."))
    put(f'<div class="row">{state}<span class="k" style="color:#5B6675;font-size:.78rem">'
                f'{f"{total / 1000:.1f} s in agents" if total else ""}</span></div>', unsafe_allow_html=True)
    if not events:
        put('<div class="k" style="margin-top:6px">Ask a question to see the agents work: the plan, retrieval '
            'scores, research batches, tool calls and safety checks appear here live.</div>')
        return
    put(stepper_html(events, running=not done), unsafe_allow_html=True)
    if not done:
        put(f'<div class="now">Now: {esc(now_line(events))}</div>', unsafe_allow_html=True)
    for err in (e for e in events if e["type"] in ("error", "node_error")):
        put(f'<div class="card tool error">{esc((str(err.get("node", "")) + " " + str(err.get("message") or err.get("error") or "")).strip())}</div>')
    render_supervisor(events)
    render_research(events)
    render_retrieval(events)
    render_tools(events)
    render_safety(events)
    render_memory(events)


# --- streaming a turn ------------------------------------------------------------------------------
def stream_turn(path: str, payload: dict, answer_box, activity_box, resume: bool = False) -> None:
    # A resumed turn continues the paused one, so keep its events (minus the old end marker).
    ss.events = [e for e in ss.events if e["type"] != "run_finished"] if resume else []
    answer, run_id, final = "", None, {}
    activity_box.empty()  # drop the previous turn's cards; a redraw alone keeps extra old children
    render_activity(activity_box, ss.events)
    try:
        with httpx.stream("POST", f"{BACKEND}{path}", json=payload, timeout=300,
                          headers={"Authorization": f"Bearer {ss.token}"}) as resp:
            if resp.status_code != 200:
                body = json.loads(resp.read() or b"{}")
                answer = f"⚠️ {body.get('message') or body.get('detail') or resp.status_code}"
            else:
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    ev = json.loads(line[6:])
                    if ev["type"] == "token":
                        answer += ev["text"]
                        answer_box.markdown(format_answer(answer) + "▌", unsafe_allow_html=True)
                        continue
                    if ev["type"] == "answer_reset":
                        answer = ""
                        answer_box.markdown("_Writing the answer…_")
                    elif ev["type"] == "final":
                        answer, final = ev["answer"], ev
                    elif ev["type"] == "run_started":
                        run_id = ev["run_id"]
                    elif ev["type"] == "approval_required":
                        ss.pending_approval = ev
                    elif ev["type"] == "error":
                        answer = f"⚠️ {ev['message']}"
                    ss.events.append(ev)
                    render_activity(activity_box, ss.events)
    except httpx.HTTPError as exc:
        answer = f"⚠️ Backend unreachable: {type(exc).__name__}"
    if answer:
        ss.messages.append({"role": "assistant", "content": answer, "run_id": run_id if final else None,
                            "sources": final.get("sources") or [], "degraded": final.get("degraded") or []})


# --- login -----------------------------------------------------------------------------------------
def login_page() -> None:
    st.markdown('<div class="cb-header"><div class="cb-brand"><div class="cb-mark">CB</div><div>'
                '<div class="cb-title">Commercial Bank · Knowledge Assistant</div>'
                '<div class="cb-sub">Internal AI assistant for policies, architecture, runbooks and incidents</div>'
                '</div></div></div>', unsafe_allow_html=True)
    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        with st.container(border=True):
            st.markdown("#### Sign in")
            with st.form("login", border=False):
                username = st.text_input("Username", placeholder="viewer, analyst or admin")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                    try:
                        r = api("POST", "/auth/login", json={"username": username.strip(), "password": password})
                        if r.status_code == 200:
                            data = r.json()
                            ss.token, ss.user, ss.tools = data["token"], data["user"], data["tools"]
                            st.rerun()
                        st.error("Invalid username or password")
                    except httpx.HTTPError:
                        st.error("Backend unreachable")
            st.caption("Demo users: viewer, analyst and admin (passwords in the README). "
                       "All documents and data are synthetic.")


if not ss.token:
    login_page()
    st.stop()

role = ss.user["role"]

# --- sidebar ---------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {esc(ss.user['display_name'])}")
    st.markdown(pill(ROLE_LABEL.get(role, role), f"pill-{role}"), unsafe_allow_html=True)
    st.caption("Tools available to this role")
    st.markdown(" ".join(pill(TOOL_LABELS.get(t, t), "pill-ghost") for t in ss.tools), unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    if c1.button("New chat", use_container_width=True):
        ss.session_id, ss.messages, ss.events, ss.pending_approval = uuid.uuid4().hex[:12], [], [], None
        st.rerun()
    if c2.button("Sign out", use_container_width=True):
        for k in ("token", "user", "pending_approval"):
            ss[k] = None
        ss.messages, ss.events = [], []
        st.rerun()
    try:
        rl = api("GET", "/me").json().get("rate_limit", {})
        st.progress(min(1.0, rl.get("tokens", 0) / max(rl.get("capacity", 1), 1)),
                    text=f"Rate limit · {rl.get('tokens', 0):.0f} of {rl.get('capacity', 0):.0f} requests left")
    except (httpx.HTTPError, ValueError):
        pass
    st.caption(f"Session `{ss.session_id}`")

    if role == "admin":
        st.divider()
        st.markdown("#### Failure injection")
        st.caption("Simulate outages to show graceful degradation.")
        current = api("GET", "/admin/chaos").json()
        labels = {"llm_fail": "AI model down", "vector_db_fail": "Vector database down",
                  "mcp_fail": "Enterprise data (MCP) down", "tool_slow": "Slow tools (timeouts)"}
        flags = {k: st.toggle(v, value=current.get(k, False)) for k, v in labels.items()}
        if flags != current:
            api("POST", "/admin/chaos", json=flags)
            st.rerun()

# --- header ----------------------------------------------------------------------------------------
st.markdown(
    '<div class="cb-header"><div class="cb-brand"><div class="cb-mark">CB</div><div>'
    '<div class="cb-title">Commercial Bank · Knowledge Assistant</div>'
    '<div class="cb-sub">Answers from internal documents and enterprise systems, with citations</div></div></div>'
    f'<div class="cb-right">{pill("LangGraph multi-agent", "pill-ghost")}{pill("Claude Sonnet 5.5", "pill-ghost")}'
    f'{pill(ROLE_LABEL.get(role, role), "pill-" + role)}</div></div>', unsafe_allow_html=True)

# --- main: chat + activity -------------------------------------------------------------------------
chat_col, activity_col = st.columns([1.45, 1], gap="medium")
with activity_col:
    with st.container(border=True, height=760, key="activity"):
        st.markdown("**Agent activity**")
        activity_box = st.empty()
        render_activity(activity_box, ss.events)

with chat_col:
    intro = st.empty()
    if not ss.messages and not ss.pending_approval and not ss.queued_prompt:
        with intro.container():
            st.markdown("##### How can I help?")
            st.caption("Try one of these, or ask your own question.")
            for j, s in enumerate(SUGGESTIONS.get(role, [])):
                if st.button(s, key=f"sugg{j}", use_container_width=True):
                    ss.queued_prompt = s
                    st.rerun()

    for i, msg in enumerate(ss.messages):
        render_message(i, msg)

    # The card lives in its own slot: when it's gone, the empty slot replaces the whole old card
    # immediately instead of leaving stale pieces on screen while the next turn streams.
    approval_slot = st.empty()
    if ss.pending_approval:
        with approval_slot.container(border=True, key="approval"):
            st.markdown("**✋ Approval needed** · this action changes live data")
            for call in ss.pending_approval["calls"]:
                args = call.get("args", {})
                if call["name"] == "mcp_update_incident_status":
                    st.markdown(f"Change incident **{esc(args.get('incident_id', ''))}** to status "
                                f"**{esc(args.get('status', ''))}**", unsafe_allow_html=True)
                    if args.get("note"):
                        st.markdown(f"> {esc(args['note'])}", unsafe_allow_html=True)
                else:
                    st.markdown(f"Run **{esc(TOOL_LABELS.get(call['name'], call['name']))}**")
                    st.code(json.dumps(args, indent=2), language="json")
                st.caption(f"Tool: {TOOL_LABELS.get(call['name'], call['name'])} · requested by "
                           f"{ss.user['display_name']} · only administrators can approve")
            a1, a2, _ = st.columns([1, 1, 2])
            decision = True if a1.button("Approve", type="primary", use_container_width=True) else (
                False if a2.button("Reject", use_container_width=True) else None)
            if decision is not None:
                # Rerun first so the card disappears, then stream the result as a normal chat bubble.
                ss.pending_approval, ss.queued_decision = None, decision
                st.rerun()

    turn_area = st.container()  # the new turn streams here, above the input box
    prompt = st.chat_input("Ask about policies, incidents, architecture, runbooks...") or ss.queued_prompt
    if ss.get("queued_decision") is not None:
        decision, ss.queued_decision = ss.queued_decision, None
        with turn_area, st.chat_message("assistant", avatar="🏦"):
            box = st.empty()
            box.markdown("_Approved, running the action…_" if decision else "_Rejected, recording the decision…_")
            stream_turn("/chat/resume", {"session_id": ss.session_id, "approved": decision}, box, activity_box,
                        resume=True)
        st.rerun()
    if prompt:
        intro.empty()
        ss.queued_prompt = None
        ss.messages.append({"role": "user", "content": prompt})
        with turn_area:
            render_message(len(ss.messages) - 1, ss.messages[-1])
            with st.chat_message("assistant", avatar="🏦"):
                box = st.empty()
                box.markdown("_Thinking…_")
                stream_turn("/chat/stream", {"message": prompt, "session_id": ss.session_id}, box, activity_box)
        st.rerun()
