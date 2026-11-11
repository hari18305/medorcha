"""Ingestion pipeline: chunk a small starter corpus, embed it with
BioBERT (embed_text), and upsert it into a Qdrant collection.

Corpus: real, citable passages loaded from data/knowledge_base/ (see
load_corpus() below and backend/rag/build_knowledge_base.py, which
produces those files from MedlinePlus and PubMed). Each passage's
"source" is its actual citation; anything not real must be labeled
source="synthetic", never mislabeled either way.

Qdrant connection: reads QDRANT_URL / QDRANT_API_KEY from the
environment (via .env). If QDRANT_URL is set, connects to that server
-- this covers both a local docker-compose Qdrant (QDRANT_URL=
http://localhost:6333, no API key) and Qdrant Cloud (QDRANT_URL=
https://<cluster>.cloud.qdrant.io, QDRANT_API_KEY=<key>). If QDRANT_URL
is not set, falls back to an embedded, on-disk local Qdrant instance
(no server required) at data/qdrant_local, so this pipeline is runnable
out of the box for local dev/demo.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.rag.embeddings import EMBEDDING_DIM, embed_text

load_dotenv()

COLLECTION_NAME = "medorchestrate_kb"

# Word-count is used as a cheap approximation of token count for
# chunking purposes (real subword tokenization would require loading
# the model's tokenizer just to size chunks, which is unnecessary
# precision for a prototype-scoped corpus this small).
# 350 (was 300): the KB builder now keeps up to 340 words per passage so a
# MedlinePlus article's Symptoms AND Causes both survive; with a 300-word window the
# last sentences would spill into a tiny second chunk.
CHUNK_SIZE_TOKENS = 350
MIN_CHUNK_TOKENS = 200
MAX_CHUNK_TOKENS = 400

DEFAULT_LOCAL_QDRANT_PATH = "data/qdrant_local"

# --------------------------------------------------------------------------
# Knowledge base: real, citable passages loaded from checked-in JSON.
#
# The corpus lives in data/knowledge_base/{guidelines,symptom_disease}/
# *.json -- verbatim MedlinePlus (U.S. NLM, public domain) text and short
# PubMed abstract excerpts, each carrying its real citation in "source"
# and a "url". It is produced by `python -m backend.rag.build_knowledge_base`
# (a developer tool, never run at ingest time), so every passage's
# provenance can be re-verified. A record may still be marked
# source="synthetic" for clearly-labeled filler, but the build spec
# requires >=80% real content, and tests/test_ingest.py enforces it.
# --------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"
KB_SUBDIRS = ("guidelines", "symptom_disease")
REQUIRED_KB_FIELDS = ("doc_id", "source", "text")


def load_corpus(kb_dir: Path | None = None) -> list[dict]:
    """Load every record from data/knowledge_base/<subdir>/*.json.

    Fails loudly (ValueError) on a missing field, a duplicate doc_id, or
    an empty knowledge base -- ingesting nothing and reporting success
    would leave RAG silently returning no evidence.
    """
    kb_dir = Path(kb_dir) if kb_dir else KNOWLEDGE_BASE_DIR
    records: list[dict] = []
    seen: set[str] = set()
    for subdir in KB_SUBDIRS:
        for path in sorted((kb_dir / subdir).glob("*.json")):
            for record in json.loads(path.read_text(encoding="utf-8")):
                missing = [f for f in REQUIRED_KB_FIELDS if not str(record.get(f, "")).strip()]
                if missing:
                    raise ValueError(f"{path.name}: record {record.get('doc_id')!r} is missing {missing}")
                if record["doc_id"] in seen:
                    raise ValueError(f"{path.name}: duplicate doc_id {record['doc_id']!r}")
                seen.add(record["doc_id"])
                records.append(record)
    if not records:
        raise ValueError(f"No knowledge-base records found under {kb_dir}/{{{','.join(KB_SUBDIRS)}}}")
    return records


def provenance_summary(corpus: list[dict]) -> dict:
    """Real vs. synthetic breakdown (the spec asks for this to be reported)."""
    synthetic = sum(1 for doc in corpus if doc["source"] == "synthetic")
    total = len(corpus)
    return {
        "total": total,
        "real": total - synthetic,
        "synthetic": synthetic,
        "real_fraction": (total - synthetic) / total,
    }


def _approx_token_count(text: str) -> int:
    """Cheap word-count approximation of token count for chunk sizing."""
    return len(text.split())


def chunk_document(doc: dict, chunk_size_tokens: int = CHUNK_SIZE_TOKENS) -> list[dict]:
    """Split one corpus document into ~200-400 token passages.

    Splits on whitespace-delimited words into windows of
    `chunk_size_tokens`. For documents already within the ~200-400
    token band (true for most of this small starter corpus) this
    produces a single chunk; longer documents are split into multiple
    sequential, non-overlapping chunks with incrementing `page` numbers.
    """
    words = doc["text"].split()
    chunks = []
    for start in range(0, len(words), chunk_size_tokens):
        chunk_words = words[start : start + chunk_size_tokens]
        chunk_text = " ".join(chunk_words)
        chunks.append(
            {
                "source": doc["source"],
                "doc_id": doc["doc_id"],
                "page": len(chunks) + 1,
                "chunk_length": len(chunk_words),
                "text": chunk_text,
                # Extra provenance, stored in the Qdrant payload only.
                "url": doc.get("url"),
                "title": doc.get("title"),
            }
        )
    return chunks


_shared_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Build a Qdrant client from QDRANT_URL / QDRANT_API_KEY, with a
    local on-disk fallback when no URL is configured (see module docstring).

    Cached as a single shared instance at module level and reused by
    EVERY caller across this process -- retriever.py
    (medorchestrate_kb + medorchestrate_case_memory), case_memory.py,
    visual_memory.py (medorchestrate_visual_memory), and this module's
    own ingest(). This is required, not just an efficiency nicety: the
    embedded local Qdrant fallback takes an exclusive file lock on
    data/qdrant_local (see qdrant_client.local.qdrant_local), so opening
    a SECOND client instance against that same path from within the
    same process raises a hard RuntimeError ("already accessed by
    another instance") -- confirmed live when retriever.py and
    case_memory.py each cached their own independent client instead of
    sharing this one. A real Qdrant server (QDRANT_URL set) has no such
    constraint, but sharing one client is still simply more efficient.
    """
    global _shared_client
    if _shared_client is not None:
        return _shared_client

    qdrant_url = os.getenv("QDRANT_URL") or None
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or None

    if qdrant_url:
        _shared_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        _shared_client = QdrantClient(path=DEFAULT_LOCAL_QDRANT_PATH)
    return _shared_client


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def build_chunks() -> list[dict]:
    chunks = []
    for doc in load_corpus():
        chunks.extend(chunk_document(doc))
    return chunks


def ingest(client: QdrantClient | None = None) -> int:
    """Chunk the knowledge-base corpus, embed each chunk, and load it into
    Qdrant.

    The "medorchestrate_kb" collection is rebuilt from scratch each time
    (it is fully regenerated from data/knowledge_base/, so upserting over
    an older, larger corpus would leave stale passages behind). Only this
    collection is touched -- case memory and visual memory are separate
    collections and are never dropped here.

    Returns the number of chunks upserted.
    """
    client = client or get_qdrant_client()
    chunks = build_chunks()  # validate/load first: a bad KB must not wipe a good collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    ensure_collection(client)


    points = [
        PointStruct(id=i, vector=embed_text(chunk["text"]), payload=chunk)
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


if __name__ == "__main__":
    count = ingest()
    summary = provenance_summary(load_corpus())
    print(f"Ingested {count} chunks from {summary['total']} documents into '{COLLECTION_NAME}'.")
    print(
        f"Provenance: {summary['real']} real / {summary['synthetic']} synthetic "
        f"({summary['real_fraction']:.0%} real)."
    )
