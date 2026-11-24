"""Hybrid retriever: dense + BM25, role-filtered, reranked, with graceful degradation.

Pipeline for one query
  1. Build the metadata filter from the *authenticated user's role* (access levels) plus
     any structured filters the supervisor extracted (document type, department, dates).
  2. Embed the query (dense) and BM25-encode it (sparse).
  3. Query the relevant namespaces concurrently (asyncio.gather). A failing namespace is
     logged and skipped; the others still answer.
  4. For every match compute the dense and sparse components separately so the UI and
     traces can show *why* a chunk ranked where it did:
         hybrid = alpha * dense + (1 - alpha) * sparse
  5. Rerank the top candidates with a cross-encoder (bge-reranker-v2-m3). If reranking
     fails, the fused order is kept.
  6. If Pinecone is down (error, timeout, circuit open, or chaos flag) fall back to the
     local BM25 index with the *same* access filter, and mark the result `degraded`.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field

from langsmith import traceable

from app.config import Settings, get_settings
from app.errors import ToolTimeoutError, VectorStoreError
from app.logging_setup import get_logger
from app.resilience import BREAKERS, chaos, with_timeout
from app.retrieval.bm25 import BM25Encoder, LocalBM25Index, sparse_dot
from app.retrieval.documents import ALL_NAMESPACES, NAMESPACE_BY_TYPE, Chunk, load_chunks
from app.retrieval.pinecone_store import PineconeStore
from app.security.rbac import allowed_access_levels

log = get_logger(__name__)


@dataclass
class SearchFilters:
    document_types: list[str] | None = None
    departments: list[str] | None = None
    date_from: str | None = None  # YYYY-MM-DD
    date_to: str | None = None
    doc_ids: list[str] | None = None

    def namespaces(self) -> list[str]:
        if self.document_types:
            return sorted({NAMESPACE_BY_TYPE.get(t, "general") for t in self.document_types})
        return list(ALL_NAMESPACES)


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    document_type: str
    department: str
    access_level: str
    created_date: str
    namespace: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None
    source: str = "pinecone"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    degraded: bool = False
    degraded_reason: str = ""
    namespaces: list[str] = field(default_factory=list)
    filter: dict = field(default_factory=dict)
    reranked: bool = False
    elapsed_ms: int = 0

    def summary(self) -> dict:
        """Compact view for the activity panel and traces."""
        return {
            "query": self.query,
            "hits": len(self.hits),
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "namespaces": self.namespaces,
            "filter": self.filter,
            "reranked": self.reranked,
            "elapsed_ms": self.elapsed_ms,
            "top": [
                {"id": h.chunk_id, "title": h.title, "dense": round(h.dense_score, 3),
                 "sparse": round(h.sparse_score, 3), "hybrid": round(h.hybrid_score, 3),
                 "rerank": None if h.rerank_score is None else round(h.rerank_score, 3)}
                for h in self.hits[:5]
            ],
        }


def build_metadata_filter(access_levels: tuple[str, ...], f: SearchFilters) -> dict:
    clauses: list[dict] = [{"access_level": {"$in": list(access_levels)}}]
    if f.document_types:
        clauses.append({"document_type": {"$in": f.document_types}})
    if f.departments:
        clauses.append({"department": {"$in": f.departments}})
    if f.doc_ids:
        clauses.append({"doc_id": {"$in": f.doc_ids}})
    if f.date_from:
        clauses.append({"created_ts": {"$gte": int(f.date_from.replace("-", ""))}})
    if f.date_to:
        clauses.append({"created_ts": {"$lte": int(f.date_to.replace("-", ""))}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def chunk_matches(c: Chunk, access_levels: tuple[str, ...], f: SearchFilters) -> bool:
    """The same filter as `build_metadata_filter`, evaluated locally (fallback path)."""
    if c.access_level not in access_levels:
        return False
    if f.document_types and c.document_type not in f.document_types:
        return False
    if f.departments and c.department not in f.departments:
        return False
    if f.doc_ids and c.doc_id not in f.doc_ids:
        return False
    if f.date_from and c.created_date < f.date_from:
        return False
    if f.date_to and c.created_date > f.date_to:
        return False
    return True


def _trace_outputs(result: RetrievalResult) -> dict:
    # LangSmith renders retriever runs nicely when outputs look like documents.
    return {
        "documents": [
            {"page_content": h.text, "type": "Document",
             "metadata": {k: v for k, v in h.to_dict().items() if k != "text"}}
            for h in result.hits
        ],
        "degraded": result.degraded,
        "filter": result.filter,
    }


class HybridRetriever:
    def __init__(self, settings: Settings, chunks: list[Chunk], encoder: BM25Encoder,
                 store: PineconeStore | None):
        self.settings = settings
        self.chunks = {c.chunk_id: c for c in chunks}
        self.encoder = encoder
        self.store = store
        self.local = LocalBM25Index(encoder, [(c.chunk_id, c.text) for c in chunks])

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> HybridRetriever:
        settings = settings or get_settings()
        chunks = load_chunks(settings.chunk_store_path)
        encoder = BM25Encoder.load(settings.bm25_stats_path)
        store = None
        if settings.pinecone_api_key:
            store = PineconeStore(settings)
        else:
            log.warning("pinecone_disabled", reason="PINECONE_API_KEY not set; local BM25 only")
        return cls(settings, chunks, encoder, store)

    # ------------------------------------------------------------------------------
    @traceable(run_type="retriever", name="hybrid_search", process_outputs=_trace_outputs)
    async def search(self, query: str, role: str, filters: SearchFilters | None = None,
                     top_k: int | None = None) -> RetrievalResult:
        filters = filters or SearchFilters()
        top_k = top_k or self.settings.retrieval_top_k
        access = allowed_access_levels(role)
        started = time.perf_counter()
        breaker = BREAKERS["vector_db"]

        reason = ""
        if self.store is None:
            reason = "vector store not configured"
        elif chaos.vector_db_fail:
            reason = "vector DB failure injected (chaos mode)"
        elif breaker.is_open:
            reason = "vector DB circuit open after repeated failures"

        if not reason:
            try:
                result = await with_timeout(
                    self._pinecone_search(query, access, filters, top_k),
                    self.settings.retrieval_timeout_s, "pinecone_search",
                )
                breaker.record_success()
                result.elapsed_ms = int((time.perf_counter() - started) * 1000)
                return result
            except (VectorStoreError, ToolTimeoutError) as exc:
                breaker.record_failure()
                reason = f"vector DB error: {exc}"
                log.warning("retrieval_degraded", error=str(exc))

        result = self._local_search(query, access, filters, top_k)
        result.degraded, result.degraded_reason = True, reason
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result

    async def _pinecone_search(self, query: str, access: tuple[str, ...], filters: SearchFilters,
                               top_k: int) -> RetrievalResult:
        assert self.store is not None
        alpha = self.settings.hybrid_alpha
        metadata_filter = build_metadata_filter(access, filters)
        namespaces = filters.namespaces()
        dense = (await self.store.embed([query], input_type="query"))[0]
        sparse = self.encoder.encode_query(query)

        responses = await asyncio.gather(
            *[self.store.query(dense, sparse, ns, metadata_filter, self.settings.retrieval_candidates)
              for ns in namespaces],
            return_exceptions=True,
        )
        matches: list[dict] = []
        failed = []
        for ns, resp in zip(namespaces, responses):
            if isinstance(resp, Exception):
                failed.append(ns)
                log.warning("namespace_query_failed", namespace=ns, error=str(resp))
            else:
                matches.extend(resp)
        if failed and len(failed) == len(namespaces):
            raise VectorStoreError(f"all namespaces failed: {failed}")

        hits: list[Hit] = []
        for m in matches:
            md = m["metadata"]
            d = sum(a * b for a, b in zip(dense, m["values"])) if m["values"] else m["score"]
            s = sparse_dot(sparse, m["sparse_values"])
            hits.append(Hit(
                chunk_id=m["id"], doc_id=md.get("doc_id", ""), title=md.get("title", ""),
                section=md.get("section", ""), text=md.get("text", ""),
                document_type=md.get("document_type", ""), department=md.get("department", ""),
                access_level=md.get("access_level", ""), created_date=md.get("created_date", ""),
                namespace=md.get("namespace", ""), dense_score=d, sparse_score=s,
                hybrid_score=alpha * d + (1 - alpha) * s,
            ))
        hits.sort(key=lambda h: h.hybrid_score, reverse=True)
        hits = hits[: self.settings.retrieval_candidates]

        reranked = False
        if self.settings.rerank_enabled and len(hits) > 1:
            try:
                scores = await self.store.rerank(query, [(h.chunk_id, h.text) for h in hits], top_n=len(hits))
                for h in hits:
                    h.rerank_score = scores.get(h.chunk_id, 0.0)
                hits.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
                reranked = True
            except VectorStoreError as exc:
                log.warning("rerank_skipped", error=str(exc))

        return RetrievalResult(
            query=query, hits=hits[:top_k], namespaces=namespaces, filter=metadata_filter,
            reranked=reranked, degraded=bool(failed),
            degraded_reason=f"namespaces unavailable: {failed}" if failed else "",
        )

    def _local_search(self, query: str, access: tuple[str, ...], filters: SearchFilters,
                      top_k: int) -> RetrievalResult:
        allowed = {cid for cid, c in self.chunks.items() if chunk_matches(c, access, filters)}
        hits = []
        for cid, score in self.local.search(query, allowed, top_k):
            c = self.chunks[cid]
            hits.append(Hit(
                chunk_id=cid, doc_id=c.doc_id, title=c.title, section=c.section, text=c.text,
                document_type=c.document_type, department=c.department,
                access_level=c.access_level, created_date=c.created_date, namespace=c.namespace,
                sparse_score=score, hybrid_score=score, source="local_bm25",
            ))
        return RetrievalResult(query=query, hits=hits, namespaces=filters.namespaces(),
                               filter=build_metadata_filter(access, filters))
