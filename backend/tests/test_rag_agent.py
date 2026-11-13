"""Tests for backend.agents.rag_agent (+ retriever.py / reranker.py).

Qdrant and the cross-encoder reranker model are both mocked -- no real
network/Qdrant server and no real model download/inference. Retrieval
and reranking still run their real code paths against fakes, so
query-building, the retrieve -> rerank pipeline, and the low-score
fallback logic in rag_agent.py are all genuinely exercised.
"""

from unittest.mock import AsyncMock

import pytest

import backend.agents.rag_agent as rag_agent_module
import backend.rag.reranker as reranker_module
import backend.rag.retriever as retriever_module
from backend.agents.rag_agent import (
    LOW_SCORE_THRESHOLD,
    _build_fallback_query,
    _build_query,
    run_rag,
)
from backend.models.schemas import ClinicalFindings, ImagingFindings, RAGEvidence

CASE001_FINDINGS = ClinicalFindings(
    demographics={},
    symptoms=["headache", "dizziness"],
    history=["hypertension"],
    labs={},
    imaging_text_findings=[],
)
CASE001_IMAGING = ImagingFindings(
    modality="MRI brain scan",
    findings=["white matter hyperintensity"],
    confidence=0.98,
)


class _FakeHit:
    def __init__(self, score: float, payload: dict):
        self.score = score
        self.payload = payload


class _FakeQueryResult:
    def __init__(self, points):
        self.points = points


class _FakeQdrantClient:
    """Returns the same canned candidate set regardless of query text --
    realistic enough for exercising the pipeline/fallback logic, since
    what actually changes between the primary and fallback query in
    these tests is the reranker's relevance scoring, not retrieval."""

    def __init__(self, payloads: list[dict]):
        self._payloads = payloads
        self.query_calls: list[list[float]] = []

    def query_points(self, collection_name, query, limit):
        self.query_calls.append(query)
        hits = [_FakeHit(score=0.5, payload=p) for p in self._payloads[:limit]]
        return _FakeQueryResult(hits)


class _FakeCrossEncoder:
    """Returns a preconfigured sequence of raw-logit lists, one per call
    to .predict() -- lets tests script "low scores first call, high
    scores second call" to drive the fallback path deterministically."""

    def __init__(self, score_sequence: list[list[float]]):
        self._score_sequence = score_sequence
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        idx = min(len(self.calls) - 1, len(self._score_sequence) - 1)
        scores = self._score_sequence[idx]
        return scores[: len(pairs)]


CANDIDATE_PAYLOADS = [
    {"text": "Brain MRI findings overview...", "source": "synthetic", "doc_id": "brain_mri_findings_overview", "page": 1, "chunk_length": 163},
    {"text": "Stroke and TIA red flags...", "source": "synthetic", "doc_id": "stroke_tia_red_flags", "page": 1, "chunk_length": 168},
    {"text": "Dizziness and vertigo overview...", "source": "synthetic", "doc_id": "dizziness_vertigo_overview", "page": 1, "chunk_length": 181},
]


@pytest.fixture
def fake_qdrant(monkeypatch):
    client = _FakeQdrantClient(CANDIDATE_PAYLOADS)
    monkeypatch.setattr(retriever_module, "_get_client", lambda: client)
    monkeypatch.setattr(retriever_module, "embed_text", lambda text: [0.0] * 768)
    return client


def _patch_cross_encoder(monkeypatch, score_sequence: list[list[float]]) -> _FakeCrossEncoder:
    fake_model = _FakeCrossEncoder(score_sequence)
    monkeypatch.setattr(reranker_module, "_get_model", lambda: fake_model)
    return fake_model


# --------------------------------------------------------------------------
# Query-building logic
# --------------------------------------------------------------------------


def test_build_query_includes_symptoms_history_and_imaging():
    query = _build_query(CASE001_FINDINGS, CASE001_IMAGING)

    assert "headache" in query
    assert "dizziness" in query
    assert "hypertension" in query
    assert "white matter hyperintensity" in query


def test_build_query_without_imaging_omits_imaging_section():
    query = _build_query(CASE001_FINDINGS, None)

    assert "headache" in query
    assert "imaging findings" not in query


def test_build_query_empty_findings_returns_empty_string():
    empty = ClinicalFindings(demographics={}, symptoms=[], history=[], labs={}, imaging_text_findings=[])

    assert _build_query(empty, None) == ""


def test_build_query_omits_unsupported_modality_findings():
    # A placeholder "no findings model available..." entry must never
    # reach the retrieval query -- it's not a real clinical finding, and
    # embedding/searching it produces misleading evidence (this is
    # exactly the mechanism that surfaced brain-related evidence for a
    # real uploaded chest X-ray before imaging.unsupported_modality
    # existed).
    unsupported = ImagingFindings(
        modality="CT scan",
        findings=["no findings model available for this modality (supported: MRI brain scan, chest X-ray)"],
        confidence=0.0,
        unsupported_modality=True,
    )

    query = _build_query(CASE001_FINDINGS, unsupported)

    assert "headache" in query
    assert "imaging findings" not in query
    assert "no findings model" not in query


def test_build_fallback_query_uses_top_symptoms_only():
    fallback = _build_fallback_query(CASE001_FINDINGS)

    assert fallback == "headache, dizziness"
    assert "hypertension" not in fallback


# --------------------------------------------------------------------------
# run_rag: happy path (no fallback needed)
# --------------------------------------------------------------------------


async def test_run_rag_happy_path_returns_ragevidence(fake_qdrant, monkeypatch):
    # raw logit 4.0 -> calibrated score ~0.996, comfortably above LOW_SCORE_THRESHOLD
    fake_model = _patch_cross_encoder(monkeypatch, [[4.0, 3.0, 2.0]])

    result = await run_rag(CASE001_FINDINGS, CASE001_IMAGING)

    assert isinstance(result, RAGEvidence)
    assert len(result.evidence) == 3
    assert all(0.0 <= item.score <= 1.0 for item in result.evidence)
    # sorted descending by (reranked) score
    scores = [item.score for item in result.evidence]
    assert scores == sorted(scores, reverse=True)
    assert all(item.source == "synthetic" for item in result.evidence)
    # only one retrieve + one rerank call -- fallback should not trigger
    assert len(fake_qdrant.query_calls) == 1
    assert len(fake_model.calls) == 1


async def test_run_rag_empty_findings_short_circuits_without_calling_pipeline(monkeypatch):
    client = _FakeQdrantClient(CANDIDATE_PAYLOADS)
    monkeypatch.setattr(retriever_module, "_get_client", lambda: client)
    fake_model = _patch_cross_encoder(monkeypatch, [[4.0, 3.0, 2.0]])

    empty = ClinicalFindings(demographics={}, symptoms=[], history=[], labs={}, imaging_text_findings=[])
    result = await run_rag(empty, None)

    assert result == RAGEvidence(evidence=[])
    assert client.query_calls == []
    assert fake_model.calls == []


# --------------------------------------------------------------------------
# run_rag: fallback path
# --------------------------------------------------------------------------


async def test_run_rag_retries_with_fallback_query_when_all_scores_low(fake_qdrant, monkeypatch):
    # First call: all raw logits very negative (-20: low under either model's calibration),
    # all below LOW_SCORE_THRESHOLD. Second call (fallback): high scores.
    fake_model = _patch_cross_encoder(
        monkeypatch,
        [[-20.0, -20.0, -20.0], [4.0, 3.0, 2.0]],
    )

    result = await run_rag(CASE001_FINDINGS, CASE001_IMAGING)

    assert len(fake_model.calls) == 2
    # second rerank call should have used the simplified fallback query
    first_call_query = fake_model.calls[0][0][0]
    second_call_query = fake_model.calls[1][0][0]
    assert first_call_query != second_call_query
    assert second_call_query == _build_fallback_query(CASE001_FINDINGS)

    # final result reflects the fallback (high-score) attempt
    assert all(item.score >= LOW_SCORE_THRESHOLD for item in result.evidence)


async def test_run_rag_gives_up_after_one_retry_if_still_low(fake_qdrant, monkeypatch):
    # Both attempts score low -- should NOT retry a third time.
    fake_model = _patch_cross_encoder(
        monkeypatch,
        [[-20.0, -20.0, -20.0], [-19.0, -19.0, -19.0]],
    )

    result = await run_rag(CASE001_FINDINGS, CASE001_IMAGING)

    assert len(fake_model.calls) == 2  # exactly one retry, not more
    assert all(item.score < LOW_SCORE_THRESHOLD for item in result.evidence)


# --------------------------------------------------------------------------
# Phase 13: case memory + visual memory merged into RAGEvidence
# --------------------------------------------------------------------------


def test_run_rag_merges_clinician_verified_text_precedent(fake_qdrant, monkeypatch):
    _patch_cross_encoder(monkeypatch, [[4.0, 3.0, 2.0]])
    monkeypatch.setattr(
        rag_agent_module,
        "retrieve_case_memory",
        lambda query, top_k=3: [{"text": "prior similar case", "source": "clinician_verified:dr_lee", "score": 0.9}],
    )

    import asyncio

    result = asyncio.run(run_rag(CASE001_FINDINGS, CASE001_IMAGING))

    tagged = [e for e in result.evidence if e.source.startswith("clinician_verified:")]
    assert len(tagged) == 1
    assert tagged[0].text == "prior similar case"
    assert tagged[0].source == "clinician_verified:dr_lee"


def test_run_rag_case_memory_failure_degrades_gracefully(fake_qdrant, monkeypatch):
    _patch_cross_encoder(monkeypatch, [[4.0, 3.0, 2.0]])

    def _boom(query, top_k=3):
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(rag_agent_module, "retrieve_case_memory", _boom)

    import asyncio

    result = asyncio.run(run_rag(CASE001_FINDINGS, CASE001_IMAGING))

    assert not any(e.source.startswith("clinician_verified:") for e in result.evidence)
    assert len(result.evidence) == 3  # the 3 real KB hits are still returned


async def test_run_rag_merges_visual_precedent_when_image_present_and_supported(fake_qdrant, monkeypatch):
    _patch_cross_encoder(monkeypatch, [[4.0, 3.0, 2.0]])
    monkeypatch.setattr(rag_agent_module, "retrieve_case_memory", lambda query, top_k=3: [])
    fake_retrieve_visual = AsyncMock(
        return_value=[{"text": "visually similar prior scan", "source": "visual_precedent:dr_lee", "score": 0.8}]
    )
    monkeypatch.setattr(rag_agent_module, "retrieve_visual_memory", fake_retrieve_visual)

    result = await run_rag(CASE001_FINDINGS, CASE001_IMAGING, image_path="data/images/synthetic_demo_brain_mri_001.png")

    fake_retrieve_visual.assert_awaited_once()
    # the current image's modality is passed so only same-modality
    # precedents can come back
    assert fake_retrieve_visual.await_args.kwargs["modality"] == CASE001_IMAGING.modality
    tagged = [e for e in result.evidence if e.source.startswith("visual_precedent:")]
    assert len(tagged) == 1
    assert tagged[0].text == "visually similar prior scan"


async def test_run_rag_skips_visual_precedent_when_no_image_path(fake_qdrant, monkeypatch):
    _patch_cross_encoder(monkeypatch, [[4.0, 3.0, 2.0]])
    monkeypatch.setattr(rag_agent_module, "retrieve_case_memory", lambda query, top_k=3: [])
    fake_retrieve_visual = AsyncMock(return_value=[])
    monkeypatch.setattr(rag_agent_module, "retrieve_visual_memory", fake_retrieve_visual)

    await run_rag(CASE001_FINDINGS, CASE001_IMAGING, image_path=None)

    fake_retrieve_visual.assert_not_called()


async def test_run_rag_skips_visual_precedent_for_unsupported_modality(fake_qdrant, monkeypatch):
    _patch_cross_encoder(monkeypatch, [[4.0, 3.0, 2.0]])
    monkeypatch.setattr(rag_agent_module, "retrieve_case_memory", lambda query, top_k=3: [])
    fake_retrieve_visual = AsyncMock(return_value=[])
    monkeypatch.setattr(rag_agent_module, "retrieve_visual_memory", fake_retrieve_visual)

    unsupported = ImagingFindings(
        modality="chest X-ray",
        findings=["no findings model available for this modality (supported: MRI brain scan, chest X-ray)"],
        confidence=0.0,
        unsupported_modality=True,
    )

    await run_rag(CASE001_FINDINGS, unsupported, image_path="data/images/some_chest_xray.png")

    fake_retrieve_visual.assert_not_called()


def test_an_inconclusive_image_read_does_not_enter_the_query():
    # CASE004: three near-equal labels (edema / atrophy / chronic infarct) used to be appended and steered retrieval.
    flat = ImagingFindings(modality="MRI brain scan", findings=["edema", "cerebral atrophy", "chronic infarct"],
                           confidence=0.34, inconclusive=True)

    query = _build_query(CASE001_FINDINGS, flat)

    assert "headache" in query and "hypertension" in query
    assert "imaging findings" not in query
    assert "edema" not in query
