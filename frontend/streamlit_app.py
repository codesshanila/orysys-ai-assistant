"""Streamlit frontend: chat on the left, live agent activity on the right.

Deliberately thin: all logic lives in the backend. The UI only
  * logs in and keeps the JWT,
  * streams /chat/stream (Server-Sent Events) and renders two things from it:
      - answer tokens into the chat bubble,
      - every other event into the Agent Activity panel (active node, plan, retrieval
        scores, research batches, tool calls, memory, validation),
  * shows Approve / Reject when a state-changing tool is waiting for a human,
  * sends thumbs up/down feedback for the turn's LangSmith run,
  * lets administrators toggle failure injection to demonstrate graceful degradation.

Run:  streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import uuid

import httpx
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Commercial Bank Knowledge Assistant", layout="wide")

ss = st.session_state
ss.setdefault("token", None)
ss.setdefault("user", None)
ss.setdefault("tools", [])
ss.setdefault("session_id", uuid.uuid4().hex[:12])
ss.setdefault("messages", [])      # [{role, content, run_id?}]
ss.setdefault("events", [])        # activity events of the latest turn
ss.setdefault("pending_approval", None)


def api(method: str, path: str, **kwargs) -> httpx.Response:
    headers = {"Authorization": f"Bearer {ss.token}"} if ss.token else {}
    return httpx.request(method, f"{BACKEND}{path}", headers=headers, timeout=30, **kwargs)


# --- activity panel rendering --------------------------------------------------------------
NODE_LABELS = {
    "input_guard": "Input guardrails", "memory_load": "Memory: load", "supervisor": "Supervisor",
    "retrieval": "Retrieval agent", "research": "Research agent (RLM)", "tool_planner": "Tools agent: plan",
    "tool_executor": "Tools agent: execute", "response": "Response agent", "validator": "Output validation",
    "memory_save": "Memory: save",
}


def describe(ev: dict) -> str | None:
    t = ev["type"]
    if t == "node_start":
        return f"▶ **{NODE_LABELS.get(ev['node'], ev['node'])}**"
    if t == "node_end" and not ev["node"].startswith("research."):
        return f"✓ {NODE_LABELS.get(ev['node'], ev['node'])} ({ev['elapsed_ms']} ms)"
    if t == "node_paused":
        return f"⏸ {ev['node']} paused for human approval"
    if t == "node_error":
        return f"✗ {ev['node']} failed: {ev['error']}"
    if t == "agent_state" and ev.get("agent") == "supervisor":
        return (f"🧭 intent=`{ev['intent']}` plan=`{ev['plan']}`  \n_{ev['reason']}_  \n"
                f"query: {ev['standalone_query']}" + (f"  \nfilters: `{ev['filters']}`" if ev.get("filters") else ""))
    if t == "agent_state" and ev.get("agent") == "tools":
        return f"🛠 planned calls: `{json.dumps(ev['planned_calls'])}`"
    if t == "retrieval" and ev.get("status") == "done":
        flag = " ⚠️ degraded: " + ev["degraded_reason"] if ev.get("degraded") else ""
        return f"🔎 '{ev['query'][:60]}' → {ev['hits']} hits in {ev['elapsed_ms']} ms, reranked={ev['reranked']}{flag}"
    if t == "research":
        phase = ev["phase"]
        d = f"[depth {ev.get('depth', 0)}] "
        if phase == "explore":
            return f"🧪 {d}explore: {ev['documents_visible']} docs visible, types {ev['by_type']}"
        if phase == "plan":
            return f"🧪 {d}search plan ({ev['source']}): {ev['rationale']}"
        if phase == "plan_rejected":
            return f"🧪 plan rejected: {ev['reason']}"
        if phase == "execute":
            return f"🧪 {d}plan executed: {ev['chunks']} chunks from {len(ev['documents'])} docs → {ev['batches']} batches"
        if phase == "analyze_batch":
            return f"🧪 {d}batch {ev['index']} ({ev.get('source', ev.get('status'))}): {ev.get('findings', 0)} findings"
        if phase == "aggregate":
            themes = ", ".join(f"{x['label']} ({x['count']})" for x in ev["themes"][:5])
            return f"🧪 {d}aggregate: {themes}" + (f"  \nfollow-ups: {ev['follow_ups']}" if ev["follow_ups"] else "")
        if phase == "recurse":
            return f"🧪 recursing to depth {ev['depth']} on {ev['follow_ups']}"
    if t == "tool_call" and ev["status"] != "running":
        return f"🛠 `{ev['tool']}` → **{ev['status']}** {ev.get('error') or ''} ({ev.get('elapsed_ms', 0)} ms)"
    if t == "approval_required":
        return "✋ approval required for: " + ", ".join(c["name"] for c in ev["calls"])
    if t == "memory":
        return f"🧠 memory {ev['action']}: " + json.dumps({k: v for k, v in ev.items() if k not in ('type', 'ts', 'action')})[:200]
    if t == "validation":
        ok = "✅" if ev.get("valid", ev.get("allowed", True)) else "⛔"
        return f"{ok} validation [{ev['stage']}] {ev.get('action', '')} issues={ev.get('issues') or ev.get('findings') or []}"
    if t == "error":
        return f"✗ {ev['message']}"
    return None


def render_activity(container, events: list[dict]) -> None:
    with container.container():
        active = next((e for e in reversed(events) if e["type"] == "node_start"), None)
        done = any(e["type"] == "run_finished" for e in events)
        waiting = any(e["type"] == "approval_required" for e in events) and not done
        state = "⏸ waiting for approval" if waiting else ("finished" if done else "running")
        st.markdown(f"**State:** {state}  \n**Active node:** "
                    f"`{NODE_LABELS.get(active['node'], active['node']) if active else '-'}`")
        lines = [line for line in (describe(e) for e in events) if line]
        st.markdown("\n\n".join(lines[-60:]) or "_No activity yet._")
        retrieval = [e for e in events if e["type"] == "retrieval" and e.get("status") == "done"]
        if retrieval and retrieval[-1].get("top"):
            st.caption("Top hits (dense / sparse / hybrid / rerank scores)")
            st.dataframe(retrieval[-1]["top"], hide_index=True, use_container_width=True)
        plan = next((e for e in events if e["type"] == "research" and e.get("phase") == "plan"), None)
        if plan:
            with st.expander("Generated search plan (Python)"):
                st.code(plan["code"], language="python")


# --- streaming a turn ------------------------------------------------------------------------
def stream_turn(path: str, payload: dict, answer_box, activity_box) -> None:
    ss.events, answer, run_id = [], "", None
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
                        answer_box.markdown(answer + "▌")
                        continue
                    if ev["type"] == "answer_reset":
                        answer = ""
                    elif ev["type"] == "final":
                        answer = ev["answer"]
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
        answer_box.markdown(answer)
        ss.messages.append({"role": "assistant", "content": answer, "run_id": run_id})


# --- sidebar: login, session, admin ------------------------------------------------------------
with st.sidebar:
    st.title("🏦 Knowledge Assistant")
    if not ss.token:
        with st.form("login"):
            username = st.text_input("Username", value="analyst")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in"):
                try:
                    r = api("POST", "/auth/login", json={"username": username, "password": password})
                    if r.status_code == 200:
                        data = r.json()
                        ss.token, ss.user, ss.tools = data["token"], data["user"], data["tools"]
                        st.rerun()
                    st.error("Invalid credentials")
                except httpx.HTTPError:
                    st.error("Backend unreachable")
        st.caption("Demo users: viewer / analyst / admin (passwords in README)")
        st.stop()

    st.markdown(f"**{ss.user['display_name']}**  \nrole: `{ss.user['role']}`")
    st.caption("Tools: " + ", ".join(ss.tools))
    st.caption(f"Session: `{ss.session_id}`")
    c1, c2 = st.columns(2)
    if c1.button("New session"):
        ss.session_id, ss.messages, ss.events, ss.pending_approval = uuid.uuid4().hex[:12], [], [], None
        st.rerun()
    if c2.button("Sign out"):
        for k in ("token", "user", "messages", "events", "pending_approval"):
            ss[k] = None if k in ("token", "user", "pending_approval") else []
        st.rerun()
    try:
        rl = api("GET", "/me").json().get("rate_limit", {})
        st.progress(min(1.0, rl.get("tokens", 0) / max(rl.get("capacity", 1), 1)),
                    text=f"Rate limit: {rl.get('tokens')}/{rl.get('capacity')} requests")
    except (httpx.HTTPError, ValueError):
        pass

    if ss.user["role"] == "admin":
        st.divider()
        st.subheader("Failure injection")
        current = api("GET", "/admin/chaos").json()
        flags = {k: st.toggle(k.replace("_", " "), value=current.get(k, False)) for k in
                 ("llm_fail", "vector_db_fail", "mcp_fail", "tool_slow")}
        if flags != current:
            api("POST", "/admin/chaos", json=flags)
            st.rerun()


# --- main: chat + activity ---------------------------------------------------------------------
chat_col, activity_col = st.columns([3, 2])
with activity_col:
    st.subheader("Agent activity")
    activity_box = st.empty()
    render_activity(activity_box, ss.events)

with chat_col:
    st.subheader("Chat")
    for i, msg in enumerate(ss.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("run_id"):
                f1, f2, _ = st.columns([1, 1, 10])
                for col, score, icon in ((f1, 1, "👍"), (f2, 0, "👎")):
                    if col.button(icon, key=f"fb{score}-{i}"):
                        api("POST", "/feedback", json={"run_id": msg["run_id"], "score": score})
                        st.toast("Thanks for the feedback")

    if ss.pending_approval:
        with st.container(border=True):
            st.warning("A state-changing action needs your approval:")
            st.json(ss.pending_approval["calls"])
            a1, a2 = st.columns(2)
            decision = True if a1.button("✅ Approve") else (False if a2.button("❌ Reject") else None)
            if decision is not None:
                ss.pending_approval = None
                with st.chat_message("assistant"):
                    stream_turn("/chat/resume", {"session_id": ss.session_id, "approved": decision},
                                st.empty(), activity_box)
                st.rerun()

    if prompt := st.chat_input("Ask about policies, incidents, architecture, runbooks..."):
        ss.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            stream_turn("/chat/stream", {"message": prompt, "session_id": ss.session_id}, st.empty(), activity_box)
        st.rerun()
