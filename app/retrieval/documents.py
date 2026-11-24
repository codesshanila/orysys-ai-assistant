"""Document loading, chunking and the document catalog.

Chunking is section-aware: documents are split on markdown headings first and long
sections are then windowed with overlap. Each chunk keeps its document metadata plus the
section heading, which is what the UI shows as attribution ("INC-2026-008 > Root cause").

The catalog (one row per document, no text) is what the research agent explores
*before* retrieving anything - the RLM idea of inspecting the collection instead of
stuffing whole documents into the context window.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Document type -> Pinecone namespace. Namespaces partition the index by domain so a
# query can be scoped (e.g. only operations data) and queried concurrently.
NAMESPACE_BY_TYPE = {
    "policy": "policies",
    "architecture": "engineering",
    "runbook": "engineering",
    "incident": "operations",
    "meeting_notes": "operations",
    "product_spec": "product",
}
ALL_NAMESPACES = sorted(set(NAMESPACE_BY_TYPE.values()))

CHUNK_WORDS = 140
CHUNK_OVERLAP = 30


@dataclass
class Document:
    doc_id: str
    title: str
    document_type: str
    department: str
    access_level: str
    created_date: str
    text: str
    extra: dict = field(default_factory=dict)

    @property
    def namespace(self) -> str:
        return NAMESPACE_BY_TYPE.get(self.document_type, "general")

    def catalog_row(self) -> dict:
        row = {k: v for k, v in asdict(self).items() if k not in ("text", "extra")}
        row.update({k: v for k, v in self.extra.items() if k in ("severity", "service", "tags")})
        row["namespace"] = self.namespace
        row["word_count"] = len(self.text.split())
        return row


@dataclass
class Chunk:
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
    severity: str = ""
    service: str = ""

    @property
    def created_ts(self) -> int:
        """Dates as YYYYMMDD integers: Pinecone range filters need numbers."""
        return int(self.created_date.replace("-", ""))

    def metadata(self) -> dict:
        meta = asdict(self)
        meta.pop("chunk_id")
        meta["created_ts"] = self.created_ts
        meta["text"] = self.text[:3000]
        return meta


def parse_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not match:
        raise ValueError(f"{path.name}: missing front matter")
    meta: dict = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key.strip()] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key.strip()] = value
    core = ("doc_id", "title", "document_type", "department", "access_level", "created_date")
    missing = [k for k in core if k not in meta]
    if missing:
        raise ValueError(f"{path.name}: missing metadata {missing}")
    return Document(
        **{k: meta[k] for k in core},
        text=match.group(2).strip(),
        extra={k: v for k, v in meta.items() if k not in core},
    )


def load_documents(directory: str | Path) -> list[Document]:
    return [parse_markdown(p) for p in sorted(Path(directory).glob("*.md"))]


def chunk_document(doc: Document) -> list[Chunk]:
    sections: list[tuple[str, str]] = []
    current_heading, buffer = doc.title, []
    for line in doc.text.splitlines():
        if line.startswith("#"):
            if buffer:
                sections.append((current_heading, "\n".join(buffer).strip()))
            current_heading, buffer = line.lstrip("# ").strip(), []
        elif not line.startswith("> SYNTHETIC"):
            buffer.append(line)
    if buffer:
        sections.append((current_heading, "\n".join(buffer).strip()))

    chunks: list[Chunk] = []
    for heading, body in sections:
        words = body.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            window = " ".join(words[start : start + CHUNK_WORDS])
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks):02d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=heading,
                    # The title is prepended so every chunk is self-describing for both
                    # embedding and keyword search.
                    text=f"{doc.title} - {heading}: {window}",
                    document_type=doc.document_type,
                    department=doc.department,
                    access_level=doc.access_level,
                    created_date=doc.created_date,
                    namespace=doc.namespace,
                    severity=str(doc.extra.get("severity", "")),
                    service=str(doc.extra.get("service", "")),
                )
            )
            if start + CHUNK_WORDS >= len(words):
                break
            start += CHUNK_WORDS - CHUNK_OVERLAP
    return chunks


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c)) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]


def save_catalog(docs: list[Document], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([d.catalog_row() for d in docs], indent=2), encoding="utf-8")


def load_catalog(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
