"""Memory design.

Three layers, each answering a different question:

1. Short-term (within a session) - "what did we just talk about?"
   The LangGraph checkpointer (SQLite) persists the full graph state per thread, where
   thread_id = session id. `messages` therefore survives every turn and process restarts.

2. Rolling summary (long sessions) - "keep the context window bounded"
   When the message list grows past `memory_window_messages`, older messages are folded
   into `summary` and removed from state. Recent turns stay verbatim (best for follow-ups
   like "and what about the second one?"); older turns survive in compressed form.

3. Long-term (across sessions, per user) - "what do we know about this user?"
   A small SQLite table keyed by username: the user's profile (role, department), their
   recent questions, and past question/answer pairs. On each turn the most relevant past
   interactions are recalled by keyword overlap and given to the supervisor.
   Keyword overlap is deliberate for the POC: it's deterministic, explainable in the
   activity panel, and needs no extra embedding call. The swap-in is the same table plus
   a vector column.

Long-term memory is written only *after* the answer passes validation, so a blocked or
invalid answer never becomes "memory".
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import aiosqlite

from app.config import get_settings

_STOPWORDS = frozenset(
    "the a an and or of to in on for is are was were what which who how why when did do does "
    "with about from by at it this that these those me my our we you your be been".split()
)


def keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS}


class LongTermMemory:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or get_settings().memory_db_path
        self._ready = False

    async def _db(self) -> aiosqlite.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(self.db_path)
        if not self._ready:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS user_memory ("
                " username TEXT NOT NULL, kind TEXT NOT NULL, content TEXT NOT NULL, ts REAL NOT NULL)"
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS ix_user ON user_memory(username, kind, ts)")
            await conn.commit()
            self._ready = True
        return conn

    async def remember_interaction(self, username: str, question: str, answer: str,
                                   citations: list[str]) -> None:
        record = json.dumps({"q": question, "a": answer[:600], "citations": citations})
        conn = await self._db()
        try:
            await conn.execute("INSERT INTO user_memory VALUES (?, 'interaction', ?, ?)",
                               (username, record, time.time()))
            await conn.commit()
        finally:
            await conn.close()

    async def recall(self, username: str, question: str, limit: int = 3) -> dict:
        """Recent questions plus the past interactions most related to `question`."""
        conn = await self._db()
        try:
            cursor = await conn.execute(
                "SELECT content FROM user_memory WHERE username=? AND kind='interaction' "
                "ORDER BY ts DESC LIMIT 50", (username,))
            rows = [json.loads(r[0]) for r in await cursor.fetchall()]
        finally:
            await conn.close()

        query_terms = keywords(question)
        scored = []
        for row in rows:
            overlap = len(query_terms & keywords(row["q"]))
            if overlap >= 2:
                scored.append((overlap, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return {
            "recent_questions": [r["q"] for r in rows[:5]],
            "related": [{"q": r["q"], "a": r["a"], "overlap": o} for o, r in scored[:limit]],
            "total_interactions": len(rows),
        }

    async def forget(self, username: str) -> None:
        conn = await self._db()
        try:
            await conn.execute("DELETE FROM user_memory WHERE username=?", (username,))
            await conn.commit()
        finally:
            await conn.close()


def format_memory_context(user: dict, recalled: dict, summary: str) -> str:
    """Plain-text block given to the supervisor and response agent."""
    lines = [f"User: {user['display_name']} (role={user['role']}, department={user['department']})"]
    if summary:
        lines.append(f"Earlier in this session (summary): {summary}")
    if recalled.get("recent_questions"):
        lines.append("Recent questions from this user: " + " | ".join(recalled["recent_questions"][:3]))
    for item in recalled.get("related", []):
        lines.append(f"Related past answer - Q: {item['q']} A: {item['a'][:300]}")
    return "\n".join(lines)
