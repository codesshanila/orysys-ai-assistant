"""Async Pinecone access: index setup, embeddings, hybrid queries and reranking.

Design choices
- One serverless index, metric=dotproduct, holding a dense vector (Pinecone-hosted
  `llama-text-embed-v2`) and a BM25 sparse vector per chunk. dotproduct is required for
  hybrid queries.
- Namespaces partition the index by domain (policies / engineering / operations / product).
- Hybrid weighting is applied on the query side (Pinecone's recommended convex
  combination): dense * alpha, sparse * (1 - alpha).
- Embeddings and reranking use Pinecone Inference, so the solution needs only one
  vector-platform key and the LLM provider stays swappable.
"""
from __future__ import annotations

from pinecone import PineconeAsyncio, ServerlessSpec

from app.config import Settings
from app.errors import VectorStoreError
from app.logging_setup import get_logger

log = get_logger(__name__)


def scale_hybrid(dense: list[float], sparse: dict, alpha: float) -> tuple[list[float], dict]:
    return (
        [v * alpha for v in dense],
        {"indices": sparse["indices"], "values": [v * (1 - alpha) for v in sparse["values"]]},
    )


class PineconeStore:
    def __init__(self, settings: Settings):
        if not settings.pinecone_api_key:
            raise VectorStoreError("PINECONE_API_KEY is not set")
        self.settings = settings
        self._pc: PineconeAsyncio | None = None
        self._host: str | None = None

    async def _client(self) -> PineconeAsyncio:
        if self._pc is None:
            self._pc = PineconeAsyncio(api_key=self.settings.pinecone_api_key)
        return self._pc

    async def close(self) -> None:
        if self._pc is not None:
            await self._pc.close()
            self._pc = None

    async def ensure_index(self) -> None:
        pc = await self._client()
        name = self.settings.pinecone_index
        if not await pc.has_index(name):
            log.info("pinecone_create_index", index=name)
            await pc.create_index(
                name=name,
                dimension=self.settings.embed_dim,
                metric="dotproduct",
                spec=ServerlessSpec(cloud=self.settings.pinecone_cloud, region=self.settings.pinecone_region),
            )
        desc = await pc.describe_index(name)
        self._host = desc.host

    async def _index(self):
        if self._host is None:
            await self.ensure_index()
        pc = await self._client()
        return pc.IndexAsyncio(host=self._host)

    async def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        pc = await self._client()
        try:
            out: list[list[float]] = []
            for start in range(0, len(texts), 90):  # inference batch limit
                res = await pc.inference.embed(
                    model=self.settings.embed_model,
                    inputs=texts[start : start + 90],
                    parameters={"input_type": input_type, "truncate": "END"},
                )
                out.extend([list(item.values) for item in res.data])
            return out
        except Exception as exc:  # SDK raises several types; normalise
            raise VectorStoreError(f"embedding failed: {exc}") from exc

    async def upsert(self, vectors: list[dict], namespace: str) -> None:
        index = await self._index()
        async with index:
            await index.upsert(vectors=vectors, namespace=namespace, batch_size=100, show_progress=False)

    async def query(
        self,
        dense: list[float],
        sparse: dict,
        namespace: str,
        metadata_filter: dict,
        top_k: int,
    ) -> list[dict]:
        alpha = self.settings.hybrid_alpha
        d, s = scale_hybrid(dense, sparse, alpha)
        try:
            index = await self._index()
            async with index:
                kwargs = dict(
                    vector=d,
                    top_k=top_k,
                    namespace=namespace,
                    filter=metadata_filter,
                    include_metadata=True,
                    include_values=True,
                )
                if s["indices"]:
                    kwargs["sparse_vector"] = s
                res = await index.query(**kwargs)
        except Exception as exc:
            raise VectorStoreError(f"query failed in namespace={namespace}: {exc}") from exc
        matches = []
        for m in res.matches:
            matches.append(
                {
                    "id": m.id,
                    "score": float(m.score),
                    "metadata": dict(m.metadata or {}),
                    "values": list(m.values or []),
                    "sparse_values": (
                        {"indices": list(m.sparse_values.indices), "values": list(m.sparse_values.values)}
                        if getattr(m, "sparse_values", None)
                        else {}
                    ),
                }
            )
        return matches

    async def rerank(self, query: str, docs: list[tuple[str, str]], top_n: int) -> dict[str, float]:
        pc = await self._client()
        try:
            res = await pc.inference.rerank(
                model=self.settings.rerank_model,
                query=query,
                documents=[{"id": i, "text": t} for i, t in docs],
                top_n=top_n,
                return_documents=False,
            )
        except Exception as exc:
            raise VectorStoreError(f"rerank failed: {exc}") from exc
        return {docs[row.index][0]: float(row.score) for row in res.data}
