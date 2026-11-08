"""End-to-end HITL review flow: submit a review, then read it back and
confirm it round-trips.

Covers both the API layer (POST then GET /review/{case_id}, mocking
orchestrator.run_case exactly like test_api.py so this stays offline)
and the storage layer directly (services/storage.py's real SQLite
persistence), each pointed at an isolated temp database file so this
suite never touches the real dev database and stays repeatable.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import backend.api.routes as routes
from backend.main import app
from backend.models.schemas import ClinicalFindings, FinalReport, ImagingFindings
from backend.services import storage

client = TestClient(app)

SAMPLE_REPORT = FinalReport(
    case_id="CASE001",
    diagnoses=[
        {"name": "Chronic hypertensive cerebral small vessel disease", "confidence": 0.58, "supporting_evidence": [], "rank": 1},
        {"name": "Vestibular migraine", "confidence": 0.3, "supporting_evidence": [], "rank": 2},
    ],
    confidence=0.58,
    evidence=[{"type": "rag", "text": "evidence text", "source": "synthetic", "score": 0.9}],
    review_required=True,
    conflicts=["Offline and live evidence point toward different underlying causes."],
)


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test_review_flow.db"))
    storage._init_tables()


@pytest.fixture
def diagnosed_case(monkeypatch):
    """Puts a report in storage the way a real /diagnose call would,
    since /review requires the case to have been diagnosed first."""
    monkeypatch.setattr(routes, "run_case", MagicMock(return_value=SAMPLE_REPORT))
    response = client.post("/diagnose", data={"case_id": "CASE001", "age": 45, "sex": "male"})
    assert response.status_code == 200
    return "CASE001"


@pytest.fixture(autouse=True)
def _mock_case_memory_boundary(monkeypatch):
    """Phase 13's override handler re-derives clinical/imaging findings
    and writes to case_memory.py/visual_memory.py -- all mocked here at
    the client boundary (same convention as the rest of this suite) so
    an override test never makes a real LLM/BiomedCLIP call or writes to
    the real on-disk Qdrant store. Individual tests can still override
    these mocks' behavior/assert on them via routes.<name>.
    """
    monkeypatch.setattr(
        routes,
        "extract_clinical_findings",
        AsyncMock(return_value=ClinicalFindings(demographics={}, symptoms=[], history=[], labs={}, imaging_text_findings=[])),
    )
    monkeypatch.setattr(routes, "analyze_image", AsyncMock(return_value=None))
    monkeypatch.setattr(routes.case_memory, "add_clinician_verified_case", MagicMock())
    monkeypatch.setattr(routes.visual_memory, "add_visual_case_memory", AsyncMock())


# --------------------------------------------------------------------------
# API round-trip: POST /review/{case_id} then GET /review/{case_id}
# --------------------------------------------------------------------------


def test_submit_then_get_review_round_trips_via_api(diagnosed_case):
    post_response = client.post(
        "/review/CASE001",
        json={"decision": "accept", "annotation": None, "reviewer": "Dr. Smith"},
    )
    assert post_response.status_code == 200
    posted = post_response.json()

    get_response = client.get("/review/CASE001")
    assert get_response.status_code == 200
    fetched = get_response.json()

    assert fetched["case_id"] == "CASE001"
    assert fetched["decision"] == "accept"
    assert fetched["reviewer"] == "Dr. Smith"
    assert fetched["annotation"] is None
    assert fetched["timestamp"] == posted["timestamp"]  # same row, not just same shape


def test_override_round_trips_with_annotation_carrying_the_new_diagnosis(diagnosed_case):
    override_text = "Override diagnosis: Vestibular migraine (reviewer judgment based on episodic pattern)"

    client.post(
        "/review/CASE001",
        json={"decision": "override", "annotation": override_text, "reviewer": "Dr. Lee"},
    )

    response = client.get("/review/CASE001")

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "override"
    assert body["annotation"] == override_text
    assert body["reviewer"] == "Dr. Lee"


def test_annotate_round_trips(diagnosed_case):
    client.post(
        "/review/CASE001",
        json={"decision": "annotate", "annotation": "Recommend follow-up MRI in 3 months.", "reviewer": "Dr. Patel"},
    )

    response = client.get("/review/CASE001")

    assert response.status_code == 200
    assert response.json() == {
        "case_id": "CASE001",
        "decision": "annotate",
        "reviewer": "Dr. Patel",
        "annotation": "Recommend follow-up MRI in 3 months.",
        "rating": None,
        "timestamp": response.json()["timestamp"],
        "recorded": True,
    }


def test_get_review_not_found_for_case_never_reviewed(diagnosed_case):
    response = client.get("/review/CASE001")
    assert response.status_code == 404


def test_re_review_returns_the_most_recent_one(diagnosed_case):
    client.post("/review/CASE001", json={"decision": "annotate", "annotation": "First pass note.", "reviewer": "Dr. Smith"})
    client.post("/review/CASE001", json={"decision": "accept", "reviewer": "Dr. Lee"})

    response = client.get("/review/CASE001")

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "accept"
    assert body["reviewer"] == "Dr. Lee"


# --------------------------------------------------------------------------
# Storage layer round-trip, direct (no HTTP)
# --------------------------------------------------------------------------


def test_storage_save_review_then_get_reviews_round_trips():
    stored = storage.save_review(
        "CASE002", {"decision": "accept", "annotation": None, "reviewer": "Dr. Nguyen"}
    )

    fetched = storage.get_reviews("CASE002")

    assert len(fetched) == 1
    assert fetched[0]["case_id"] == "CASE002"
    assert fetched[0]["decision"] == "accept"
    assert fetched[0]["reviewer"] == "Dr. Nguyen"
    assert fetched[0]["annotation"] is None
    assert fetched[0]["timestamp"] == stored["timestamp"]
    assert fetched[0]["id"] == stored["id"]


def test_storage_get_reviews_returns_full_history_most_recent_first():
    storage.save_review("CASE003", {"decision": "annotate", "annotation": "note 1", "reviewer": "Dr. A"})
    storage.save_review("CASE003", {"decision": "override", "annotation": "note 2", "reviewer": "Dr. B"})

    history = storage.get_reviews("CASE003")

    assert len(history) == 2
    assert history[0]["reviewer"] == "Dr. B"  # most recent first
    assert history[1]["reviewer"] == "Dr. A"


def test_storage_get_reviews_returns_empty_list_when_never_reviewed():
    assert storage.get_reviews("never-reviewed-case") == []


# --------------------------------------------------------------------------
# Phase 13: override triggers case-memory writes; accept/annotate don't.
# --------------------------------------------------------------------------


def test_override_triggers_text_case_memory_write(diagnosed_case):
    response = client.post(
        "/review/CASE001",
        json={"decision": "override", "annotation": "Vestibular migraine", "reviewer": "Dr. Lee"},
    )

    assert response.status_code == 200
    routes.case_memory.add_clinician_verified_case.assert_called_once()
    call_kwargs = routes.case_memory.add_clinician_verified_case.call_args.kwargs
    assert call_kwargs["case_id"] == "CASE001"
    assert call_kwargs["diagnosis_text"] == "Vestibular migraine"
    assert call_kwargs["reviewer"] == "Dr. Lee"
    # no image on this case -- visual memory must NOT be written
    routes.visual_memory.add_visual_case_memory.assert_not_called()


def test_accept_does_not_trigger_case_memory_write(diagnosed_case):
    client.post("/review/CASE001", json={"decision": "accept", "reviewer": "Dr. Smith"})

    routes.case_memory.add_clinician_verified_case.assert_not_called()
    routes.visual_memory.add_visual_case_memory.assert_not_called()


def test_annotate_does_not_trigger_case_memory_write(diagnosed_case):
    client.post(
        "/review/CASE001",
        json={"decision": "annotate", "annotation": "Follow up in 3 months.", "reviewer": "Dr. Patel"},
    )

    routes.case_memory.add_clinician_verified_case.assert_not_called()
    routes.visual_memory.add_visual_case_memory.assert_not_called()


def test_override_on_image_case_writes_both_text_and_visual_memory(monkeypatch):
    monkeypatch.setattr(routes, "run_case", MagicMock(return_value=SAMPLE_REPORT))
    monkeypatch.setattr(
        routes,
        "analyze_image",
        AsyncMock(return_value=ImagingFindings(modality="MRI brain scan", findings=["white matter hyperintensity"], confidence=0.9)),
    )
    response = client.post(
        "/diagnose",
        data={"case_id": "CASE_IMG", "age": 60, "sex": "female"},
        files={"image": ("scan.png", b"\x89PNG\r\n\x1a\n fake png bytes", "image/png")},
    )
    assert response.status_code == 200

    override_response = client.post(
        "/review/CASE_IMG",
        json={"decision": "override", "annotation": "Chronic small vessel disease", "reviewer": "Dr. Kim"},
    )

    assert override_response.status_code == 200
    routes.case_memory.add_clinician_verified_case.assert_called_once()
    routes.visual_memory.add_visual_case_memory.assert_awaited_once()
    visual_kwargs = routes.visual_memory.add_visual_case_memory.call_args.kwargs
    assert visual_kwargs["modality"] == "MRI brain scan"
    assert visual_kwargs["diagnosis_text"] == "Chronic small vessel disease"


def test_override_on_unsupported_modality_image_writes_only_text_memory(monkeypatch):
    monkeypatch.setattr(routes, "run_case", MagicMock(return_value=SAMPLE_REPORT))
    monkeypatch.setattr(
        routes,
        "analyze_image",
        AsyncMock(
            return_value=ImagingFindings(
                modality="unspecified medical image",
                findings=["no findings model available for this modality (supported: MRI brain scan, chest X-ray)"],
                confidence=0.0,
                unsupported_modality=True,
            )
        ),
    )
    response = client.post(
        "/diagnose",
        data={"case_id": "CASE_UNSUPPORTED", "age": 60, "sex": "female"},
        files={"image": ("scan.png", b"\x89PNG\r\n\x1a\n fake png bytes", "image/png")},
    )
    assert response.status_code == 200

    override_response = client.post(
        "/review/CASE_UNSUPPORTED",
        json={"decision": "override", "annotation": "Some replacement diagnosis", "reviewer": "Dr. Kim"},
    )

    assert override_response.status_code == 200
    routes.case_memory.add_clinician_verified_case.assert_called_once()
    routes.visual_memory.add_visual_case_memory.assert_not_called()


def test_review_with_rating_round_trips_and_case_closes():
    """Phase 11/13: a rating alongside a decision round-trips, and the
    case reaches "closed" status on successful submission."""
    from backend.models.schemas import CaseInput, Patient

    storage.save_case(CaseInput(case_id="CASE_RATED", patient=Patient(age=50, sex="male")), status="completed")
    storage.save_report("CASE_RATED", SAMPLE_REPORT.model_copy(update={"case_id": "CASE_RATED"}))

    response = client.post(
        "/review/CASE_RATED",
        json={"decision": "accept", "reviewer": "Dr. Smith", "rating": 4},
    )

    assert response.status_code == 200
    assert response.json()["rating"] == 4

    fetched = storage.get_case("CASE_RATED")
    with storage._get_connection() as conn:
        row = conn.execute("SELECT status FROM cases WHERE case_id = ?", ("CASE_RATED",)).fetchone()
    assert row["status"] == "closed"
    assert fetched is not None  # sanity: the case row itself still exists


def test_review_rating_out_of_bounds_is_rejected(diagnosed_case):
    response = client.post("/review/CASE001", json={"decision": "accept", "reviewer": "Dr. Smith", "rating": 6})
    assert response.status_code == 422
