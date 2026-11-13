"""RAG agent: turns ClinicalFindings (+ ImagingFindings) into a query,
retrieves and reranks evidence from the local knowledge base, and
returns RAGEvidence.

Retrieval (retriever.py) and reranking (reranker.py) are both
synchronous, CPU/IO-bound calls, so the pipeline runs inside asyncio's
default executor -- same pattern as imaging_agent.py -- keeping this a
proper async function without blocking the event loop.

Case memory (Phase 13): run_rag() also folds in clinician-verified text
precedents (retriever.retrieve_case_memory(), tagged
"clinician_verified:<reviewer>") and, when the current case has an
image in a supported modality, visually-similar past-case precedents
(visual_memory.retrieve_visual_memory(), tagged
"visual_precedent:<reviewer>") -- both merged into the same returned
RAGEvidence.evidence list, distinguishable from ordinary KB evidence (and
from each other) by their tagged `source` string. Neither ever
participates in the reranker's cross-encoder scoring below (that's only
for the raw KB candidates) -- they're appended with their own retrieval
score after reranking, since "similarity to a past clinician
correction" and "cross-encoder relevance to the KB" aren't the same
signal and shouldn't be blended.
"""

import asyncio
import logging

from backend.models.schemas import ClinicalFindings, EvidenceItem, ImagingFindings, RAGEvidence
from backend.rag.reranker import rerank
from backend.rag.retriever import retrieve, retrieve_case_memory
from backend.rag.visual_memory import retrieve_visual_memory

logger = logging.getLogger(__name__)

RETRIEVE_TOP_K = 20  # was 10 for the 42-passage KB; at 67 passages new related passages pushed a relevant one (CASE009 subdural) out of the top 10
RERANK_TOP_K = 5
CASE_MEMORY_TOP_K = 3
VISUAL_MEMORY_TOP_K = 3

# If every reranked result scores below this, the first query is
# treated as having failed to find relevant evidence and a broadened
# fallback query is tried once before giving up.
LOW_SCORE_THRESHOLD = 0.3

# Fallback query strategy: broaden to just the top N symptoms, dropping
# history/imaging detail that may have over-specified the first query.
FALLBACK_MAX_SYMPTOMS = 2


def _build_query(findings: ClinicalFindings, imaging: ImagingFindings | None) -> str:
    """Join symptoms + history + imaging findings into a natural clinical query."""
    parts = []
    if findings.symptoms:
        parts.append(", ".join(findings.symptoms))
    if findings.history:
        parts.append("history of " + ", ".join(findings.history))
    if imaging is not None and imaging.drives_retrieval:
        parts.append("imaging findings: " + ", ".join(imaging.findings))
    return "; ".join(parts)


def _build_fallback_query(findings: ClinicalFindings) -> str:
    """Broaden to just the top symptoms, dropping history/imaging terms."""
    return ", ".join(findings.symptoms[:FALLBACK_MAX_SYMPTOMS])


def _run_pipeline(query: str) -> list[dict]:
    candidates = retrieve(query, top_k=RETRIEVE_TOP_K)
    if not candidates:
        return []
    return rerank(query, candidates, top_k=RERANK_TOP_K)


def _all_below_threshold(results: list[dict]) -> bool:
    return not results or all(r["score"] < LOW_SCORE_THRESHOLD for r in results)


async def run_rag(
    findings: ClinicalFindings,
    imaging: ImagingFindings | None,
    query_override: str | None = None,
    image_path: str | None = None,
) -> RAGEvidence:
    """query_override lets a caller (e.g. optimizer_agent.py, re-running
    retrieval with a Claude-refined query) skip the normal
    findings/imaging-derived query and supply one directly. When given,
    the low-score fallback below still applies -- a refined query that
    still scores poorly falls back to just the top symptoms, same as
    the default path.

    image_path (Phase 13, NEW in v5): the case's raw image, if any --
    used ONLY to query the visual case-memory collection (image
    similarity, not text similarity). Passing it has no effect on the
    text query built above. Visual-memory retrieval is skipped entirely
    (no meaningful query to run) when image_path is None, imaging is
    None, or imaging.unsupported_modality is True.
    """
    query = query_override or _build_query(findings, imaging)
    if not query:
        return RAGEvidence(evidence=[])

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, _run_pipeline, query)

    if _all_below_threshold(results):
        fallback_query = _build_fallback_query(findings)
        if fallback_query and fallback_query != query:
            fallback_results = await loop.run_in_executor(None, _run_pipeline, fallback_query)
            if fallback_results:
                results = fallback_results

    evidence = [
        EvidenceItem(text=r["text"], source=r["source"], score=r["score"]) for r in results
    ]

    # Phase 13: clinician-verified text precedents, merged in alongside
    # the KB results above -- tagged "clinician_verified:<reviewer>" so
    # they're distinguishable at a glance (see retrieve_case_memory()).
    try:
        case_memory_hits = await loop.run_in_executor(None, retrieve_case_memory, query, CASE_MEMORY_TOP_K)
    except Exception as exc:  # Qdrant down, etc. -- degrade, don't fail the whole RAG step
        logger.warning("run_rag(): case-memory retrieval failed, skipping: %s", exc)
        case_memory_hits = []
    evidence.extend(
        EvidenceItem(text=hit["text"] or "", source=hit["source"], score=hit["score"]) for hit in case_memory_hits
    )

    # Phase 13: visually-similar past-case precedents -- only when the
    # CURRENT case actually has an image in a supported modality; there
    # is no meaningful visual query to run otherwise.
    if image_path and imaging is not None and not imaging.unsupported_modality:
        try:
            visual_hits = await retrieve_visual_memory(
                image_path, top_k=VISUAL_MEMORY_TOP_K, modality=imaging.modality
            )
        except Exception as exc:  # Qdrant down, BiomedCLIP failure, etc.
            logger.warning("run_rag(): visual-memory retrieval failed, skipping: %s", exc)
            visual_hits = []
        evidence.extend(
            EvidenceItem(text=hit["text"] or "", source=hit["source"], score=hit["score"]) for hit in visual_hits
        )

    return RAGEvidence(evidence=evidence)
