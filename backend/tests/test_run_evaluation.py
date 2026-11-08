"""Tests for backend.tests.run_evaluation's Phase 17 additions (Mode 2,
Mode 3, and the required text-case-memory ablation).

run_evaluation.py itself is NOT auto-collected by pytest (see its own
docstring -- it does real, slow work by design when run directly), but
this file imports its functions and exercises them against mocks/fakes,
same "mock at the client boundary" convention as the rest of this
suite, so these run fast and offline in the normal test suite.
"""

import json
import sys
import types

import pytest
import torch

import backend.agents.imaging_agent as imaging_agent
import backend.services.storage as storage
import backend.tests.run_evaluation as ev
from backend.agents.imaging_labels import FINDING_LABELS_BY_MODALITY, MODALITY_LABELS
from backend.models.schemas import CaseInput, Patient


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test_run_evaluation.db"))
    storage._init_tables()


# --------------------------------------------------------------------------
# Mode 2: MedQA/PubMedQA -- graceful "unavailable" path
# --------------------------------------------------------------------------


def test_load_medqa_pubmedqa_unavailable_when_datasets_not_installed(monkeypatch):
    # Setting the sys.modules entry to None makes `import datasets` raise
    # ImportError even when the real package IS installed.
    monkeypatch.setitem(sys.modules, "datasets", None)
    result = ev._load_medqa_pubmedqa(10)
    assert result["available"] is False
    assert "datasets" in result["reason"]


def test_run_medqa_pubmedqa_benchmark_unavailable_propagates_cleanly(monkeypatch):
    # Simulate the `datasets` package being absent -- must degrade to a
    # clear, honest result, never raise.
    monkeypatch.setitem(sys.modules, "datasets", None)
    result = ev.run_medqa_pubmedqa_benchmark(n_per_benchmark=5)
    assert result["available"] is False
    assert "datasets" in result["reason"]


def test_print_mode2_result_handles_unavailable(capsys):
    ev.print_mode2_result({"available": False, "reason": "the `datasets` package is not installed"})
    captured = capsys.readouterr()
    assert "UNAVAILABLE" in captured.out


# --------------------------------------------------------------------------
# Mode 2: normalization + scoring logic, against a FAKE `datasets` module
# (so this suite doesn't need the real package or network access).
# --------------------------------------------------------------------------


@pytest.fixture
def fake_datasets_module(monkeypatch):
    medqa_rows = [
        {"question": "Q1?", "choices": ["A opt", "B opt", "C opt"], "answer": ["B opt"]},
        {"question": "Q2?", "choices": ["X opt", "Y opt"], "answer": ["X opt"]},
    ]
    pubmedqa_rows = [
        {"question": "Does X cause Y?", "final_decision": "yes", "context": {"contexts": ["Ctx one.", "Ctx two."]}},
        {"question": "Does Z help W?", "final_decision": "no", "context": {"contexts": ["Only ctx."]}},
    ]

    class _FakeDataset(list):
        pass

    def _fake_load_dataset(path, *args, **kwargs):
        if path == "bigbio/med_qa":
            return _FakeDataset(medqa_rows)
        if path == "qiaojin/PubMedQA":
            return _FakeDataset(pubmedqa_rows)
        raise ValueError(f"unexpected dataset path in test: {path}")

    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = _fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    return {"medqa_rows": medqa_rows, "pubmedqa_rows": pubmedqa_rows}


def test_load_medqa_pubmedqa_normalizes_both_benchmarks(fake_datasets_module):
    medqa_items, pubmedqa_items = ev._load_medqa_pubmedqa(10)

    assert medqa_items == [
        {"question": "Q1?", "options": ["A opt", "B opt", "C opt"], "correct_letter": "B"},
        {"question": "Q2?", "options": ["X opt", "Y opt"], "correct_letter": "A"},
    ]
    assert pubmedqa_items == [
        {
            "question": "Does X cause Y?", "options": ["yes", "no", "maybe"],
            "correct_letter": "A", "context": "Ctx one. Ctx two.",
        },
        {
            "question": "Does Z help W?", "options": ["yes", "no", "maybe"],
            "correct_letter": "B", "context": "Only ctx.",
        },
    ]


async def test_answer_mcq_returns_none_on_client_failure(monkeypatch):
    def _boom():
        raise RuntimeError("no credentials configured")

    monkeypatch.setattr("backend.services.llm_client.get_llm_client", _boom)

    result = await ev._answer_mcq("A question?", ["opt1", "opt2"])

    assert result is None


async def test_answer_mcq_parses_a_valid_letter(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="B")]
    fake_client.messages.create = AsyncMock(return_value=fake_response)
    monkeypatch.setattr("backend.services.llm_client.get_llm_client", lambda: fake_client)

    result = await ev._answer_mcq("A question?", ["opt1", "opt2"])

    assert result == "B"


async def test_answer_mcq_rejects_an_invalid_letter(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="Z")]  # out of range for a 2-option question
    fake_client.messages.create = AsyncMock(return_value=fake_response)
    monkeypatch.setattr("backend.services.llm_client.get_llm_client", lambda: fake_client)

    result = await ev._answer_mcq("A question?", ["opt1", "opt2"])

    assert result is None


def test_run_medqa_pubmedqa_benchmark_end_to_end_with_fakes(fake_datasets_module, monkeypatch):
    """Full Mode 2 harness, with the dataset faked (above) and the LLM
    MCQ-answering call replaced by a deterministic "always correct"
    stand-in -- proves the accuracy/calibration/recalibration plumbing
    itself works, independent of whether a real credentialed provider
    or real HF network access is available in this environment.
    """

    async def _always_correct(question, options, *, _items={}):
        return None  # overwritten per-call below via monkeypatch of _score_items instead

    async def _fake_score_items(items):
        return [{**item, "predicted_letter": item["correct_letter"], "correct": True} for item in items]

    monkeypatch.setattr(ev, "_score_items", _fake_score_items)

    # The calibration half calls real RAG + live-evidence retrieval per
    # item; replace both and the breakdown with deterministic stand-ins so
    # this stays offline (no Qdrant, no PubMed).
    async def _fake_rag(*args, **kwargs):
        return None

    async def _fake_live(*args, **kwargs):
        return None

    monkeypatch.setattr(ev, "run_rag", _fake_rag)
    monkeypatch.setattr(ev, "gather_live_evidence", _fake_live)
    monkeypatch.setattr(
        "backend.services.confidence.compute_confidence_breakdown",
        lambda rag, live, fusion: {"s_rag": 0.8, "s_web": 0.6, "agreement": 0.7, "confidence": 0.7},
    )

    result = ev.run_medqa_pubmedqa_benchmark(n_per_benchmark=10)

    assert result["available"] is True
    assert result["medqa_accuracy"] == 1.0
    assert result["pubmedqa_accuracy"] == 1.0
    assert result["medqa_errors"] == 0
    assert "before_weights" in result and "after_weights" in result


# --------------------------------------------------------------------------
# Mode 3: imaging mini-benchmark (BiomedCLIP mocked -- same convention as
# test_imaging_agent.py, no real weights downloaded).
# --------------------------------------------------------------------------


class _RecordingTokenizer:
    def __call__(self, labels):
        return list(labels)


class _DeterministicModel:
    """encode_image always returns the SAME embedding regardless of the
    (fake, solid-color) input image -- encode_text returns a
    near-perfect match for whichever label the test wants to "win"."""

    def __init__(self, dominant_label: str):
        self.dominant_label = dominant_label

    def encode_image(self, image_input):
        return torch.tensor([[1.0, 0.0]])

    def encode_text(self, labels):
        vectors = [
            torch.tensor([1.0, 0.0]) if label == self.dominant_label else torch.tensor([0.0, 1.0])
            for label in labels
        ]
        return torch.stack(vectors)


def _fake_preprocess(image):
    return torch.zeros(3, 224, 224)


@pytest.fixture
def imaging_eval_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "IMAGING_EVAL_DIR", tmp_path / "imaging_eval")
    monkeypatch.setattr(ev, "IMAGING_EVAL_MANIFEST", tmp_path / "imaging_eval" / "manifest.json")
    return tmp_path / "imaging_eval"


def test_generate_imaging_eval_set_covers_every_supported_label(imaging_eval_dir):
    manifest = ev.generate_imaging_eval_set()

    total_labels = sum(len(labels) for labels in FINDING_LABELS_BY_MODALITY.values())
    assert len(manifest) == total_labels
    assert set(FINDING_LABELS_BY_MODALITY) <= {entry["modality"] for entry in manifest}
    for entry in manifest:
        from pathlib import Path

        assert Path(entry["image_path"]).exists()
    assert json.loads((imaging_eval_dir / "manifest.json").read_text())


def test_run_imaging_benchmark_scores_top1_accuracy(imaging_eval_dir, monkeypatch):
    manifest = ev.generate_imaging_eval_set(images_per_label=1)
    # Force every MRI image to classify as the FIRST MRI label correctly,
    # and every chest X-ray image to also land on the correct label --
    # i.e. a deterministic, per-image "correct" model bundle.
    mri_first_label = FINDING_LABELS_BY_MODALITY["MRI brain scan"][0]

    call_count = {"n": 0}

    def _bundle_for_call():
        # Every entry in this test's manifest is scored against a model
        # that always picks the label the image was actually generated
        # for -- proven per-entry below via a fresh bundle per image.
        return None

    def _per_image_bundle(entry):
        model = _DeterministicModel(entry["intended_label"])
        return model, _fake_preprocess, _RecordingTokenizer()

    # analyze_image() calls _get_model_bundle() once per call -- patch it
    # to return a bundle keyed to whichever image is being processed by
    # tracking call order against the manifest.
    manifest_iter = iter(manifest)
    current = {"entry": None}

    def _get_model_bundle():
        return _per_image_bundle(current["entry"])

    monkeypatch.setattr(imaging_agent, "_get_model_bundle", _get_model_bundle)

    correct = 0
    for entry in manifest:
        current["entry"] = entry
        import asyncio

        result = asyncio.run(imaging_agent.analyze_image(entry["image_path"], modality_hint=entry["modality"]))
        if result.findings[0] == entry["intended_label"]:
            correct += 1

    assert correct == len(manifest)  # every image correctly classified by construction

    # Now exercise run_imaging_benchmark() itself with a bundle that
    # always gets the MRI modality's first label right (deterministic,
    # single bundle -- some images "wrong" by construction, which is
    # fine, this just checks the accuracy bookkeeping is correct).
    monkeypatch.setattr(imaging_agent, "_get_model_bundle", lambda: (_DeterministicModel(mri_first_label), _fake_preprocess, _RecordingTokenizer()))

    benchmark = ev.run_imaging_benchmark(manifest)

    assert set(benchmark["accuracy_by_modality"]) == set(FINDING_LABELS_BY_MODALITY)
    mri_labels = FINDING_LABELS_BY_MODALITY["MRI brain scan"]
    expected_mri_accuracy = sum(1 for e in manifest if e["modality"] == "MRI brain scan" and e["intended_label"] == mri_first_label) / len(mri_labels)
    assert benchmark["accuracy_by_modality"]["MRI brain scan"] == pytest.approx(expected_mri_accuracy, abs=1e-3)


def test_run_image_fusion_comparison_demonstrates_the_code_path_differs(imaging_eval_dir, monkeypatch):
    manifest = ev.generate_imaging_eval_set(images_per_label=1)
    monkeypatch.setattr(
        imaging_agent, "_get_model_bundle",
        lambda: (_DeterministicModel(manifest[0]["intended_label"]), _fake_preprocess, _RecordingTokenizer()),
    )

    rows = ev.run_image_fusion_comparison(manifest, sample_size=2)

    assert len(rows) == 2
    for row in rows:
        assert row["differs"] is True  # with-image and without-image mocks are deliberately different
        assert row["with_image_confidence"] != row["without_image_confidence"]


# --------------------------------------------------------------------------
# Required ablation: with vs. without text case-memory retrieval
# --------------------------------------------------------------------------


class _FakePoint:
    def __init__(self, id, vector, payload):
        self.id, self.vector, self.payload = id, vector, payload


class _FakeHit:
    def __init__(self, score, payload):
        self.score, self.payload = score, payload


class _FakeQueryResult:
    def __init__(self, points):
        self.points = points


class _FakeQdrantClient:
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

    def query_points(self, collection_name, query, limit):
        col = self.collections.get(collection_name, {"points": {}})
        hits = [_FakeHit(score=0.9, payload=p.payload) for p in list(col["points"].values())[:limit]]
        return _FakeQueryResult(hits)


def test_run_case_memory_ablation_shows_a_confidence_delta(monkeypatch):
    import backend.agents.clinical_agent as clinical_agent
    import backend.agents.fusion_agent as fusion_agent
    import backend.rag.case_memory as case_memory
    import backend.rag.retriever as retriever

    fake_qdrant = _FakeQdrantClient()
    monkeypatch.setattr(case_memory, "_get_client", lambda: fake_qdrant)
    monkeypatch.setattr(retriever, "_get_client", lambda: fake_qdrant)
    monkeypatch.setattr(case_memory, "embed_text", lambda text: [1.0, 0.0])
    monkeypatch.setattr(retriever, "embed_text", lambda text: [1.0, 0.0])

    from unittest.mock import AsyncMock, MagicMock

    clinical_mock_client = MagicMock()
    clinical_mock_client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[
                MagicMock(
                    text=json.dumps(
                        {"demographics": {}, "symptoms": ["headache", "dizziness"], "history": ["hypertension"], "labs": {}, "imaging_text_findings": []}
                    )
                )
            ]
        )
    )
    monkeypatch.setattr(clinical_agent, "_get_client", lambda: clinical_mock_client)

    case = CaseInput(
        case_id="EVAL001", patient=Patient(age=45, sex="male"),
        clinical_text="recurrent headache and dizziness", history=["hypertension"],
    )

    result = ev.run_case_memory_ablation(case)

    assert result["case_memory_entries_present"] >= 1
    assert "confidence_with_case_memory" in result
    assert "confidence_without_case_memory" in result
    assert isinstance(result["delta"], float)
