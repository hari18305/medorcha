"""Visual case memory (Phase 13, NEW in v5): a Qdrant collection of
clinician-verified diagnosis corrections for cases that had an image,
retrievable as evidence for future similar cases by VISUAL similarity
of the image itself -- not by symptom-text similarity (that's text
case_memory.py's job).

Embedding space: BiomedCLIP's image encoder (the same loaded model
imaging_agent.py already uses for zero-shot classification -- reused
via imaging_agent.embed_image()/get_biomedclip_bundle(), never a second
copy loaded here). This is a COMPLETELY DIFFERENT space from BioBERT's
768-dim text embeddings: different dimension, different meaning. Mixing
BiomedCLIP and BioBERT vectors in the same collection, or assuming they
share a dimension, would silently produce meaningless similarity scores
or a hard Qdrant dimension-mismatch error -- which is exactly why this
lives in its own collection ("medorchestrate_visual_memory"), separate
from both "medorchestrate_kb" and "medorchestrate_case_memory", with its
vector size determined at runtime from the actual loaded model (see
imaging_agent.get_image_embedding_dim()) rather than assumed.

Written to only when a clinician overrides a diagnosis on a case that
had an image in a SUPPORTED modality (see backend/api/routes.py's
POST /review/{case_id} handler) -- skipped entirely for text-only
overrides, since there's no image to embed. Like case_memory.py, this
collection grows unboundedly and is not curated; a future phase should
add staleness/expiry or periodic re-review, not implemented here.
"""

import logging
import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.agents.imaging_agent import embed_image, get_image_embedding_dim
from backend.rag.ingest import get_qdrant_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "medorchestrate_visual_memory"

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = get_qdrant_client()
    return _client


def ensure_collection(client: QdrantClient | None = None) -> None:
    client = client or _get_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=get_image_embedding_dim(), distance=Distance.COSINE),
        )


async def add_visual_case_memory(
    case_id: str,
    image_path: str,
    modality: str,
    diagnosis_text: str,
    reviewer: str,
    review_id: int | None = None,
) -> None:
    """Embeds `image_path` via the SAME cached BiomedCLIP model
    imaging_agent.py uses, and upserts it into
    "medorchestrate_visual_memory", tagged source="clinician_verified_visual".
    Only call this when the override's case actually had an image in a
    supported modality -- callers (POST /review/{case_id}) are
    responsible for that check; this function does not re-verify it.
    """
    from backend.services import storage

    client = _get_client()
    ensure_collection(client)

    vector = await embed_image(image_path)
    point_id = str(uuid.uuid4())
    payload = {
        "diagnosis": diagnosis_text,
        "source": "clinician_verified_visual",
        "reviewer": reviewer,
        "case_id": case_id,
        "modality": modality,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    client.upsert(collection_name=COLLECTION_NAME, points=[PointStruct(id=point_id, vector=vector, payload=payload)])
    logger.info("visual_memory: added clinician-verified visual entry for case_id=%s", case_id)
    storage.log_case_memory_embedding(
        case_id=case_id, review_id=review_id, qdrant_point_id=point_id, memory_type="visual"
    )


async def retrieve_visual_memory(image_path: str, top_k: int = 3) -> list[dict]:
    """Embeds the query image and searches "medorchestrate_visual_memory".
    Only call this when the CURRENT case has an image in a supported
    modality -- callers (rag_agent.py) are responsible for that check;
    there is no meaningful visual query to run otherwise. Returns []
    (never raises) if the collection doesn't exist yet -- e.g. no
    clinician has overridden an image case yet in this deployment.
    """
    client = _get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return []

    vector = await embed_image(image_path)
    hits = client.query_points(collection_name=COLLECTION_NAME, query=vector, limit=top_k).points

    return [
        {
            "text": hit.payload.get("diagnosis"),
            # Kept visibly distinct from text case_memory's
            # "clinician_verified:<reviewer>" tag -- this is a visually
            # similar past IMAGE, not a similar-sounding past case.
            "source": f"visual_precedent:{hit.payload.get('reviewer', '')}",
            "score": hit.score,
            "case_id": hit.payload.get("case_id"),
            "modality": hit.payload.get("modality"),
        }
        for hit in hits
    ]
