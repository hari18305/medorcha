"""Tests for backend.rag.case_memory (text) and backend.rag.visual_memory
(image, NEW in v5).

Qdrant, BiomedCLIP, and the BioBERT embedding function are all mocked --
no real network/Qdrant server, no real model weights, no real inference.
The fake Qdrant client below implements real cosine-similarity ranking
over small hand-picked vectors, so retrieval genuinely exercises
"a similar-symptom query retrieves the text entry" / "a visually-similar
image retrieves the visual entry", not just a canned return value.
"""

import math

import pytest

import backend.rag.case_memory as case_memory
import backend.rag.visual_memory as visual_memory
import backend.services.storage as storage
from backend.models.schemas import ClinicalFindings, ImagingFindings


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test_case_memory.db"))
    storage._init_tables()


class _FakePoint:
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload


class _FakeHit:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class _FakeQueryResult:
    def __init__(self, points):
        self.points = points


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


class _FakeQdrantClient:
    """A tiny in-memory Qdrant stand-in with real cosine-similarity
    ranking, shared across both case_memory.py and visual_memory.py's
    tests (and their two, deliberately separate, collections)."""

    def __init__(self):
        self.collections: dict[str, dict] = {}

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections.setdefault(collection_name, {"points": {}})

    def upsert(self, collection_name, points):
        col = self.collections.setdefault(collection_name, {"points": {}})
        for p in points:
            col["points"][p.id] = _FakePoint(p.id, p.vector, p.payload)

    def query_points(self, collection_name, query, limit, query_filter=None, score_threshold=None):
        """Honours the two real-Qdrant options retrieval now relies on:
        `query_filter` (exact-match payload conditions, applied BEFORE the
        top-k cut like the real server) and `score_threshold` (drop hits
        scoring below it)."""
        col = self.collections.get(collection_name, {"points": {}})
        candidates = list(col["points"].values())
        if query_filter is not None:
            for condition in query_filter.must:
                candidates = [p for p in candidates if p.payload.get(condition.key) == condition.match.value]
        ranked = sorted(candidates, key=lambda p: _cosine(p.vector, query), reverse=True)
        hits = [_FakeHit(score=_cosine(p.vector, query), payload=p.payload) for p in ranked]
        if score_threshold is not None:
            hits = [h for h in hits if h.score >= score_threshold]
        return _FakeQueryResult(hits[:limit])


CLINICAL = ClinicalFindings(
    demographics={}, symptoms=["headache", "dizziness"], history=["hypertension"], labs={}, imaging_text_findings=[]
)
IMAGING = ImagingFindings(modality="MRI brain scan", findings=["white matter hyperintensity"], confidence=0.9)


# --------------------------------------------------------------------------
# Text case memory
# --------------------------------------------------------------------------


@pytest.fixture
def fake_text_client(monkeypatch):
    client = _FakeQdrantClient()
    monkeypatch.setattr(case_memory, "_get_client", lambda: client)
    return client


def test_add_clinician_verified_case_writes_a_single_point_and_logs_text_memory_type(fake_text_client, monkeypatch):
    monkeypatch.setattr(case_memory, "embed_text", lambda text: [1.0, 0.0])

    case_memory.add_clinician_verified_case(
        case_id="CASE001", clinical=CLINICAL, imaging=IMAGING, diagnosis_text="Vestibular migraine",
        reviewer="dr_lee", review_id=1,
    )

    col = fake_text_client.collections[case_memory.COLLECTION_NAME]
    assert len(col["points"]) == 1
    point = next(iter(col["points"].values()))
    assert point.payload["diagnosis"] == "Vestibular migraine"
    assert point.payload["source"] == "clinician_verified"
    assert point.payload["reviewer"] == "dr_lee"

    logged = storage.get_case_memory_log("CASE001")
    assert len(logged) == 1
    assert logged[0]["memory_type"] == "text"


def test_build_case_text_includes_symptoms_history_and_supported_imaging():
    text = case_memory.build_case_text(CLINICAL, IMAGING)
    assert "headache" in text
    assert "hypertension" in text
    assert "white matter hyperintensity" in text


def test_build_case_text_omits_unsupported_modality_findings():
    unsupported = ImagingFindings(
        modality="chest X-ray",
        findings=["no findings model available for this modality (supported: MRI brain scan, chest X-ray)"],
        confidence=0.0,
        unsupported_modality=True,
    )
    text = case_memory.build_case_text(CLINICAL, unsupported)
    assert "no findings model" not in text


def test_retrieve_case_memory_finds_similar_symptom_entry(fake_text_client, monkeypatch):
    import backend.rag.retriever as retriever

    monkeypatch.setattr(retriever, "_get_client", lambda: fake_text_client)
    monkeypatch.setattr(case_memory, "embed_text", lambda text: [1.0, 0.0] if "headache" in text else [0.0, 1.0])
    monkeypatch.setattr(retriever, "embed_text", lambda text: [1.0, 0.0])  # the new query also mentions headache

    case_memory.add_clinician_verified_case(
        case_id="CASE001", clinical=CLINICAL, imaging=IMAGING, diagnosis_text="Vestibular migraine", reviewer="dr_lee"
    )

    hits = retriever.retrieve_case_memory("headache dizziness", top_k=3)

    assert len(hits) == 1
    assert hits[0]["source"] == "clinician_verified:dr_lee"
    # The precedent shows the past case AND the clinician's diagnosis --
    # the diagnosis is the whole point of the feedback loop.
    assert hits[0]["text"] == (
        f"Similar past case: {case_memory.build_case_text(CLINICAL, IMAGING)}. "
        "Clinician-verified diagnosis: Vestibular migraine."
    )
    assert hits[0]["diagnosis"] == "Vestibular migraine"


def test_text_precedent_strips_the_override_diagnosis_transport_prefix(fake_text_client, monkeypatch):
    # The review UI sends the replacement as "Override diagnosis: <name>".
    import backend.rag.retriever as retriever

    monkeypatch.setattr(retriever, "_get_client", lambda: fake_text_client)
    monkeypatch.setattr(case_memory, "embed_text", lambda text: [1.0, 0.0])
    monkeypatch.setattr(retriever, "embed_text", lambda text: [1.0, 0.0])

    case_memory.add_clinician_verified_case(
        case_id="CASE001", clinical=CLINICAL, imaging=None,
        diagnosis_text="Override diagnosis: Chronic subdural hematoma", reviewer="dr_lee",
    )

    text = retriever.retrieve_case_memory("headache dizziness", top_k=3)[0]["text"]

    assert text.endswith("Clinician-verified diagnosis: Chronic subdural hematoma.")
    assert "Override diagnosis" not in text


def test_clean_diagnosis_handles_prefix_case_and_plain_text():
    assert case_memory.clean_diagnosis("Override diagnosis: Migraine") == "Migraine"
    assert case_memory.clean_diagnosis("  override DIAGNOSIS :   Migraine  ") == "Migraine"
    assert case_memory.clean_diagnosis("Migraine") == "Migraine"
    assert case_memory.clean_diagnosis("") == ""


def test_retrieve_case_memory_drops_unrelated_past_cases_below_the_min_score(fake_text_client, monkeypatch):
    """Without a cutoff every query got its 3 nearest past cases however
    unrelated. A chest-pain query must not surface a headache precedent."""
    import backend.rag.retriever as retriever

    monkeypatch.setattr(retriever, "_get_client", lambda: fake_text_client)
    monkeypatch.setattr(case_memory, "embed_text", lambda text: [1.0, 0.0])
    case_memory.add_clinician_verified_case(
        case_id="CASE001", clinical=CLINICAL, imaging=None, diagnosis_text="Tension headache", reviewer="dr_lee"
    )

    monkeypatch.setattr(retriever, "embed_text", lambda text: [0.0, 1.0])  # orthogonal: cosine 0.0
    assert retriever.retrieve_case_memory("chest pain, shortness of breath", top_k=3) == []

    monkeypatch.setattr(retriever, "embed_text", lambda text: [0.9, 0.3])  # cosine ~0.95
    assert len(retriever.retrieve_case_memory("headache and dizziness", top_k=3)) == 1


def test_case_memory_min_score_is_overridable_per_call(fake_text_client, monkeypatch):
    import backend.rag.retriever as retriever

    monkeypatch.setattr(retriever, "_get_client", lambda: fake_text_client)
    monkeypatch.setattr(case_memory, "embed_text", lambda text: [1.0, 0.0])
    case_memory.add_clinician_verified_case(
        case_id="CASE001", clinical=CLINICAL, imaging=None, diagnosis_text="Tension headache", reviewer="dr_lee"
    )
    monkeypatch.setattr(retriever, "embed_text", lambda text: [0.5, 0.866])  # cosine 0.5

    assert retriever.retrieve_case_memory("q", min_score=0.6) == []
    assert len(retriever.retrieve_case_memory("q", min_score=0.4)) == 1


def test_retrieve_case_memory_returns_empty_when_collection_never_created(fake_text_client, monkeypatch):
    import backend.rag.retriever as retriever

    monkeypatch.setattr(retriever, "_get_client", lambda: fake_text_client)
    assert retriever.retrieve_case_memory("anything") == []


# --------------------------------------------------------------------------
# Visual case memory (NEW in v5)
# --------------------------------------------------------------------------


@pytest.fixture
def fake_visual_client(monkeypatch):
    client = _FakeQdrantClient()
    monkeypatch.setattr(visual_memory, "_get_client", lambda: client)
    monkeypatch.setattr(visual_memory, "get_image_embedding_dim", lambda: 2)
    return client


async def test_add_visual_case_memory_writes_a_point_in_a_separate_collection(fake_visual_client, monkeypatch):
    async def fake_embed_image(path):
        return [1.0, 0.0]

    monkeypatch.setattr(visual_memory, "embed_image", fake_embed_image)

    await visual_memory.add_visual_case_memory(
        case_id="CASE002", image_path="data/images/synthetic_demo_brain_mri_001.png", modality="MRI brain scan",
        diagnosis_text="Chronic small vessel disease", reviewer="dr_patel", review_id=2,
    )

    assert visual_memory.COLLECTION_NAME != case_memory.COLLECTION_NAME
    col = fake_visual_client.collections[visual_memory.COLLECTION_NAME]
    assert len(col["points"]) == 1
    point = next(iter(col["points"].values()))
    assert point.payload["source"] == "clinician_verified_visual"
    assert point.payload["modality"] == "MRI brain scan"

    logged = storage.get_case_memory_log("CASE002")
    assert len(logged) == 1
    assert logged[0]["memory_type"] == "visual"


async def test_retrieve_visual_memory_finds_visually_similar_entry(fake_visual_client, monkeypatch):
    vectors = {"prior.png": [1.0, 0.0], "new_query.png": [0.9, 0.1], "unrelated.png": [0.0, 1.0]}

    async def fake_embed_image(path):
        return vectors[path]

    monkeypatch.setattr(visual_memory, "embed_image", fake_embed_image)

    await visual_memory.add_visual_case_memory(
        case_id="CASE002", image_path="prior.png", modality="MRI brain scan",
        diagnosis_text="Chronic small vessel disease", reviewer="dr_patel",
    )

    hits = await visual_memory.retrieve_visual_memory("new_query.png", top_k=3)

    assert len(hits) == 1
    assert hits[0]["source"] == "visual_precedent:dr_patel"
    assert hits[0]["text"] == (
        "Past case with a visually similar MRI brain scan image. "
        "Clinician-verified diagnosis: Chronic small vessel disease."
    )
    assert hits[0]["modality"] == "MRI brain scan"


async def _seed_visual(monkeypatch, vectors, entries):
    """entries: [(path, modality, diagnosis)] written via the real writer."""

    async def fake_embed_image(path):
        return vectors[path]

    monkeypatch.setattr(visual_memory, "embed_image", fake_embed_image)
    for i, (path, modality, diagnosis) in enumerate(entries):
        await visual_memory.add_visual_case_memory(
            case_id=f"CASE{i}", image_path=path, modality=modality, diagnosis_text=diagnosis, reviewer="dr_x"
        )


async def test_visual_memory_never_returns_a_precedent_from_a_different_modality(fake_visual_client, monkeypatch):
    """A brain MRI override must not come back as a 'visually similar'
    precedent for a chest X-ray, even if the embeddings land close."""
    vectors = {"mri.png": [1.0, 0.0], "cxr_query.png": [0.99, 0.05], "mri_query.png": [0.99, 0.05]}
    await _seed_visual(monkeypatch, vectors, [("mri.png", "MRI brain scan", "Chronic infarct")])

    cxr_hits = await visual_memory.retrieve_visual_memory("cxr_query.png", modality="chest X-ray")
    mri_hits = await visual_memory.retrieve_visual_memory("mri_query.png", modality="MRI brain scan")

    assert cxr_hits == []
    assert len(mri_hits) == 1


async def test_visual_memory_modality_filter_applies_before_the_top_k_cut(fake_visual_client, monkeypatch):
    """Two closer wrong-modality images must not push the right-modality
    one out of the top_k -- the filter has to be applied first."""
    vectors = {
        "ct1.png": [1.0, 0.0], "ct2.png": [0.99, 0.1], "mri.png": [0.9, 0.2], "q.png": [1.0, 0.0],
    }
    await _seed_visual(
        monkeypatch, vectors,
        [("ct1.png", "CT scan", "A"), ("ct2.png", "CT scan", "B"), ("mri.png", "MRI brain scan", "Right one")],
    )

    hits = await visual_memory.retrieve_visual_memory("q.png", top_k=1, modality="MRI brain scan", min_score=0.5)

    assert [h["modality"] for h in hits] == ["MRI brain scan"]
    assert "Right one" in hits[0]["text"]


async def test_visual_memory_drops_dissimilar_images_below_the_min_score(fake_visual_client, monkeypatch):
    vectors = {"prior.png": [1.0, 0.0], "different.png": [0.6, 0.8], "same.png": [0.99, 0.05]}
    await _seed_visual(monkeypatch, vectors, [("prior.png", "MRI brain scan", "Chronic infarct")])

    assert await visual_memory.retrieve_visual_memory("different.png", modality="MRI brain scan") == []  # cosine 0.6
    assert len(await visual_memory.retrieve_visual_memory("same.png", modality="MRI brain scan")) == 1


async def test_retrieve_visual_memory_returns_empty_when_collection_never_created(fake_visual_client, monkeypatch):
    async def fake_embed_image(path):
        return [1.0, 0.0]

    monkeypatch.setattr(visual_memory, "embed_image", fake_embed_image)

    assert await visual_memory.retrieve_visual_memory("anything.png") == []


async def test_visual_memory_uses_biomedclip_dimension_not_biobert_768(fake_visual_client, monkeypatch):
    """Regression test for the exact bug the spec calls out: the visual
    collection's vector size must come from BiomedCLIP's real output
    dimension, never assumed to be BioBERT's 768."""
    monkeypatch.setattr(visual_memory, "get_image_embedding_dim", lambda: 512)

    async def fake_embed_image(path):
        return [0.1] * 512

    monkeypatch.setattr(visual_memory, "embed_image", fake_embed_image)

    captured = {}
    original_create = fake_visual_client.create_collection

    def _spy_create_collection(collection_name, vectors_config):
        captured["size"] = vectors_config.size
        original_create(collection_name, vectors_config)

    monkeypatch.setattr(fake_visual_client, "create_collection", _spy_create_collection)

    await visual_memory.add_visual_case_memory(
        case_id="CASE003", image_path="x.png", modality="MRI brain scan", diagnosis_text="dx", reviewer="r"
    )

    assert captured["size"] == 512
    assert captured["size"] != 768


# --------------------------------------------------------------------------
# Full round-trip (Phase 13 spec's own acceptance scenario): override an
# image case, then a new case with a similarly-worded complaint AND a
# visually similar image should retrieve BOTH a clinician_verified: and a
# visual_precedent: entry, correctly tagged and distinguishable.
# --------------------------------------------------------------------------


async def test_text_only_override_creates_only_a_text_entry(fake_text_client, fake_visual_client, monkeypatch):
    monkeypatch.setattr(case_memory, "embed_text", lambda text: [1.0, 0.0])

    case_memory.add_clinician_verified_case(
        case_id="CASE_TEXT_ONLY", clinical=CLINICAL, imaging=None, diagnosis_text="Tension headache", reviewer="dr_a"
    )
    # visual_memory.add_visual_case_memory is deliberately NOT called for
    # a text-only override -- see routes.py's POST /review/{case_id}.

    logged = storage.get_case_memory_log("CASE_TEXT_ONLY")
    assert len(logged) == 1
    assert logged[0]["memory_type"] == "text"


async def test_image_case_override_creates_both_text_and_visual_entries_and_both_are_retrievable(
    fake_text_client, fake_visual_client, monkeypatch
):
    import backend.rag.retriever as retriever

    monkeypatch.setattr(retriever, "_get_client", lambda: fake_text_client)
    monkeypatch.setattr(case_memory, "embed_text", lambda text: [1.0, 0.0])
    monkeypatch.setattr(retriever, "embed_text", lambda text: [1.0, 0.0])

    async def fake_embed_image(path):
        return {"original.png": [1.0, 0.0], "similar_new.png": [0.95, 0.05]}[path]

    monkeypatch.setattr(visual_memory, "embed_image", fake_embed_image)

    # Simulate the override: both memory types get written (see routes.py).
    case_memory.add_clinician_verified_case(
        case_id="CASE_IMG", clinical=CLINICAL, imaging=IMAGING,
        diagnosis_text="Chronic small vessel disease", reviewer="dr_b", review_id=5,
    )
    await visual_memory.add_visual_case_memory(
        case_id="CASE_IMG", image_path="original.png", modality="MRI brain scan",
        diagnosis_text="Chronic small vessel disease", reviewer="dr_b", review_id=5,
    )

    logged = storage.get_case_memory_log("CASE_IMG")
    assert {row["memory_type"] for row in logged} == {"text", "visual"}
    assert len(logged) == 2

    # A subsequent similar-symptom query retrieves the TEXT entry.
    text_hits = retriever.retrieve_case_memory("headache dizziness hypertension", top_k=3)
    assert len(text_hits) == 1
    assert text_hits[0]["source"] == "clinician_verified:dr_b"

    # A subsequent visually-similar image retrieves the VISUAL entry.
    visual_hits = await visual_memory.retrieve_visual_memory("similar_new.png", top_k=3)
    assert len(visual_hits) == 1
    assert visual_hits[0]["source"] == "visual_precedent:dr_b"

    # Both correctly tagged and distinguishable from each other.
    assert text_hits[0]["source"] != visual_hits[0]["source"]
