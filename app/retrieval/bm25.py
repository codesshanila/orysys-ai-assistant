"""BM25 for the sparse half of hybrid search.

Pinecone stores sparse vectors, not an inverted index with BM25 scoring, so we use the
standard decomposition (the same one Pinecone's BM25Encoder uses):

    document vector  value(t) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * len/avg_len))
    query vector     value(t) = idf(t) / sum(idf)

The dot product of the two is a (normalised) BM25 score, so Pinecone can compute it at
query time. Token ids are CRC32 hashes - deterministic, so the corpus statistics file
plus the code fully reproduces every vector.

`LocalBM25Index` scores the same way in memory. It is the degraded-mode fallback when
Pinecone is unavailable, and it powers offline tests.
"""
from __future__ import annotations

import json
import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the this to was
    were will with which what when where who why how all any can could did do does during
    into last our over should than then there these they those we you your about after""".split()
)
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*[a-z0-9]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def token_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF


@dataclass
class BM25Encoder:
    k1: float = 1.2
    b: float = 0.75
    n_docs: int = 0
    avg_len: float = 0.0
    doc_freq: dict[str, int] = field(default_factory=dict)

    def fit(self, texts: list[str]) -> BM25Encoder:
        df: Counter[str] = Counter()
        total = 0
        for text in texts:
            tokens = tokenize(text)
            total += len(tokens)
            df.update(set(tokens))
        self.n_docs = len(texts)
        self.avg_len = total / max(len(texts), 1)
        self.doc_freq = dict(df)
        return self

    def idf(self, token: str) -> float:
        df = self.doc_freq.get(token, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def encode_document(self, text: str) -> dict:
        tokens = tokenize(text)
        tf = Counter(tokens)
        norm = 1 - self.b + self.b * len(tokens) / max(self.avg_len, 1e-9)
        weights: dict[int, float] = {}
        for tok, count in tf.items():
            weights[token_id(tok)] = count * (self.k1 + 1) / (count + self.k1 * norm)
        return _to_sparse(weights)

    def encode_query(self, text: str) -> dict:
        tokens = set(tokenize(text))
        idfs = {token_id(t): self.idf(t) for t in tokens}
        total = sum(idfs.values()) or 1.0
        return _to_sparse({k: v / total for k, v in idfs.items() if v > 0})

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"k1": self.k1, "b": self.b, "n_docs": self.n_docs,
                        "avg_len": self.avg_len, "doc_freq": self.doc_freq}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> BM25Encoder:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _to_sparse(weights: dict[int, float]) -> dict:
    items = sorted(weights.items())
    return {"indices": [k for k, _ in items], "values": [round(v, 6) for _, v in items]}


def sparse_dot(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    lookup = dict(zip(a.get("indices", []), a.get("values", [])))
    return sum(lookup.get(i, 0.0) * v for i, v in zip(b.get("indices", []), b.get("values", [])))


class LocalBM25Index:
    """In-memory BM25 over chunk texts (fallback + offline mode)."""

    def __init__(self, encoder: BM25Encoder, items: list[tuple[str, str]]):
        self.encoder = encoder
        self._ids = [i for i, _ in items]
        self._vectors = [encoder.encode_document(t) for _, t in items]

    def search(self, query: str, allowed_ids: set[str] | None, top_k: int) -> list[tuple[str, float]]:
        q = self.encoder.encode_query(query)
        scored = [
            (cid, sparse_dot(q, vec))
            for cid, vec in zip(self._ids, self._vectors)
            if allowed_ids is None or cid in allowed_ids
        ]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: s[1], reverse=True)
        return scored[:top_k]
