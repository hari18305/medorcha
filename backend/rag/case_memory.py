"""Text case memory (Phase 13): a Qdrant collection of clinician-verified
diagnosis corrections, retrievable as evidence for future similar cases
by symptom-text similarity.

Embedding space: BioBERT (backend/rag/embeddings.py), 768-dim, same
space as the curated knowledge base -- but a SEPARATE Qdrant collection
("medorchestrate_case_memory", not "medorchestrate_kb"). Case-memory
entries are clinician corrections, not curated reference material; they
must never be silently blended into the KB collection, and downstream
consumers (fusion_agent.py) must be able to tell the two apart, which is
why every entry's `source` field is tagged "clinician_verified:<reviewer>"
rather than left as a generic source string.

Written to only when a clinician overrides a diagnosis (see
backend/api/routes.py's POST /review/{case_id} handler) -- this
collection grows unboundedly and is not curated. A future phase should
add staleness/expiry or periodic re-review; not implemented here.
"""

import logging
import re
import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.models.schemas import ClinicalFindings, ImagingFindings
from backend.rag.embeddings import EMBEDDING_DIM, embed_text
from backend.rag.ingest import get_qdrant_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "medorchestrate_case_memory"

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    """Lazily create and cache a single Qdrant client for this process --
    same rationale as retriever.py's _get_client(): the local on-disk
    Qdrant fallback takes an exclusive file lock, so a single reused
    client avoids both that conflict and reconnect overhead."""
    global _client
    if _client is None:
        _client = get_qdrant_client()
    return _client


def ensure_collection(client: QdrantClient | None = None) -> None:
    client = client or _get_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


# The review UI sends a replacement diagnosis as "Override diagnosis: <name>"
# in ReviewRequest.annotation (there is no separate field for it -- see
# backend/api/schemas.py). That prefix is transport framing, not part of
# the diagnosis, so it is stripped before the diagnosis is shown to a model.
_OVERRIDE_PREFIX_RE = re.compile(r"^\s*override diagnosis\s*:\s*", re.IGNORECASE)


def clean_diagnosis(diagnosis_text: str) -> str:
    return _OVERRIDE_PREFIX_RE.sub("", diagnosis_text or "").strip()


def format_text_precedent(case_text: str, diagnosis_text: str) -> str:
    """The evidence text a text-memory hit is shown as. It MUST carry the
    clinician's diagnosis, not just the past case's symptoms: a precedent
    that says only "a similar case existed" tells the fusion model nothing
    about what the clinician actually concluded, which is the entire point
    of the feedback loop."""
    return f"Similar past case: {case_text}. Clinician-verified diagnosis: {clean_diagnosis(diagnosis_text)}."


def build_case_text(clinical: ClinicalFindings, imaging: ImagingFindings | None) -> str:
    """Same query-shape convention as rag_agent._build_query() (symptoms
    + history + supported-modality imaging findings), so a future case's
    RAG query text lines up well against what got embedded here."""
    parts = []
    if clinical.symptoms:
        parts.append(", ".join(clinical.symptoms))
    if clinical.history:
        parts.append("history of " + ", ".join(clinical.history))
    if imaging is not None and imaging.findings and not imaging.unsupported_modality:
        parts.append("imaging findings: " + ", ".join(imaging.findings))
    return "; ".join(parts)


def add_clinician_verified_case(
    case_id: str,
    clinical: ClinicalFindings,
    imaging: ImagingFindings | None,
    diagnosis_text: str,
    reviewer: str,
    review_id: int | None = None,
) -> None:
    """Embeds the combined case text and upserts it into
    "medorchestrate_case_memory", tagged source="clinician_verified" (the
    reviewer identity travels in its own `reviewer` payload field, and
    retriever.retrieve_case_memory() folds it into the returned
    source string as "clinician_verified:<reviewer>" for RAGEvidence).
    Logs the write via storage.log_case_memory_embedding(memory_type="text")
    -- imported lazily to avoid a module-level import cycle with
    backend.services.storage (which does not import this module, but
    keeping the import local here matches the "only import what a call
    actually needs" style already used for optional/heavy deps
    elsewhere in this codebase).
    """
    from backend.services import storage

    client = _get_client()
    ensure_collection(client)

    text = build_case_text(clinical, imaging)
    vector = embed_text(text)
    point_id = str(uuid.uuid4())
    payload = {
        "text": text,
        "diagnosis": diagnosis_text,
        "source": "clinician_verified",
        "reviewer": reviewer,
        "case_id": case_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    client.upsert(collection_name=COLLECTION_NAME, points=[PointStruct(id=point_id, vector=vector, payload=payload)])
    logger.info("case_memory: added clinician-verified text entry for case_id=%s", case_id)
    storage.log_case_memory_embedding(
        case_id=case_id, review_id=review_id, qdrant_point_id=point_id, memory_type="text"
    )
