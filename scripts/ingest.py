"""Ingest the knowledge base.

    python scripts/ingest.py             # chunk, fit BM25, build catalog, upsert to Pinecone
    python scripts/ingest.py --local     # local artefacts only (no Pinecone; offline mode)

Steps: parse markdown + metadata -> section-aware chunks -> fit BM25 statistics on the
chunk corpus -> write chunk store / catalog / BM25 stats -> embed chunks with Pinecone
Inference -> upsert dense + sparse vectors per namespace.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.retrieval.bm25 import BM25Encoder  # noqa: E402
from app.retrieval.documents import chunk_document, load_documents, save_catalog, save_chunks  # noqa: E402


async def main(local_only: bool) -> None:
    settings = get_settings()
    docs = load_documents(settings.documents_dir)
    chunks = [c for d in docs for c in chunk_document(d)]
    encoder = BM25Encoder().fit([c.text for c in chunks])

    save_chunks(chunks, settings.chunk_store_path)
    save_catalog(docs, settings.catalog_path)
    encoder.save(settings.bm25_stats_path)
    print(f"Parsed {len(docs)} documents into {len(chunks)} chunks; local artefacts written.")
    if local_only:
        return

    from app.retrieval.pinecone_store import PineconeStore

    store = PineconeStore(settings)
    try:
        await store.ensure_index()
        by_ns = defaultdict(list)
        for c in chunks:
            by_ns[c.namespace].append(c)
        for ns, ns_chunks in by_ns.items():
            dense = await store.embed([c.text for c in ns_chunks], input_type="passage")
            vectors = [
                {"id": c.chunk_id, "values": vec,
                 "sparse_values": encoder.encode_document(c.text), "metadata": c.metadata()}
                for c, vec in zip(ns_chunks, dense)
            ]
            await store.upsert(vectors, namespace=ns)
            print(f"  upserted {len(vectors):3d} chunks into namespace '{ns}'")
    finally:
        await store.close()
    print(f"Done. Index: {settings.pinecone_index}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="skip Pinecone upload")
    asyncio.run(main(parser.parse_args().local))
