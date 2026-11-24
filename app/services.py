"""Process-wide singletons (retriever, long-term memory).

Built lazily on first use so importing modules never triggers I/O, and tests can
replace them with `set_retriever` / `set_memory`.
"""
from __future__ import annotations

from app.agents.memory import LongTermMemory
from app.retrieval.hybrid import HybridRetriever

_retriever: HybridRetriever | None = None
_memory: LongTermMemory | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever.from_settings()
    return _retriever


def set_retriever(retriever: HybridRetriever) -> None:
    global _retriever
    _retriever = retriever


def get_memory() -> LongTermMemory:
    global _memory
    if _memory is None:
        _memory = LongTermMemory()
    return _memory


def set_memory(memory: LongTermMemory) -> None:
    global _memory
    _memory = memory
