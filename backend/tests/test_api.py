"""Tests for the FastAPI routes (backend/api/routes.py, backend/main.py).

orchestrator.run_case() is mocked at the backend.api.routes module
boundary (where it's imported), so these tests never touch Claude,
Qdrant, PubMed, or BiomedCLIP -- fully offline and fast. The in-memory
storage stub is cleared between tests via an autouse fixture, since its
module-level dicts would otherwise leak state across tests.
"""

import io
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import backend.api.routes as routes
from backend.main import app
from backend.models.schemas import FinalReport, ImagingFindings
from backend.services import storage

client = TestClient(app)

SAMPLE_REPORT = FinalReport(
    case_id="CASE001",
    diagnoses=[{"name": "Tension-type headache", "confidence": 0.8, "supporting_evidence": [], "rank": 1}],
    confidence=0.8,
    evidence=[{"type": "rag", "text": "evidence text", "source": "synthetic", "score": 0.9}],
    review_required=False,
)


@pytest.fixture(autouse=True)
def _clear_storage(tmp_path, monkeypatch):
    # Cases, reports, reviews, and agent_logs are all real SQLite now --
    # point at an isolated temp file per test rather than the real dev
    # database, and (re)create the tables there.
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test_api.db"))
    storage._init_tables()


@pytest.fixture
def mock_run_case(monkeypatch):
    mock = MagicMock(return_value=SAMPLE_REPORT)
    monkeypatch.setattr(routes, "run_case", mock)
    return mock


# --------------------------------------------------------------------------
# GET /health
# --------------------------------------------------------------------------


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --------------------------------------------------------------------------
# POST /diagnose
# --------------------------------------------------------------------------


def test_diagnose_text_only(mock_run_case):
    response = client.post(
        "/diagnose",
        data={
            "case_id": "CASE001",
            "age": 45,
            "sex": "male",
            "clinical_text": "headache and dizziness, history of hypertension",
            "history": json.dumps(["hypertension"]),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == "CASE001"
    assert body["confidence"] == 0.8
    assert body["review_required"] is False

    mock_run_case.assert_called_once()
    submitted_case = mock_run_case.call_args.args[0]
    assert submitted_case.case_id == "CASE001"
    assert submitted_case.clinical_text == "headache and dizziness, history of hypertension"
    assert submitted_case.history == ["hypertension"]
    assert submitted_case.image_path is None


def test_diagnose_generates_case_id_when_not_provided(mock_run_case):
    response = client.post("/diagnose", data={"age": 30, "sex": "female"})

    assert response.status_code == 200
    submitted_case = mock_run_case.call_args.args[0]
    assert submitted_case.case_id  # non-empty, auto-generated
    assert len(submitted_case.case_id) > 0


def test_diagnose_with_image_upload(mock_run_case):
    fake_image = io.BytesIO(b"not a real image, just bytes for the test")

    response = client.post(
        "/diagnose",
        data={"case_id": "CASE_IMG", "age": 60, "sex": "female"},
        files={"image": ("scan.png", fake_image, "image/png")},
    )

    assert response.status_code == 200
    submitted_case = mock_run_case.call_args.args[0]
    assert submitted_case.image_path is not None
    assert submitted_case.image_path.endswith(".png")


def test_diagnose_missing_required_fields_returns_422(mock_run_case):
    # missing both "age" and "sex"
    response = client.post("/diagnose", data={"case_id": "CASE001"})

    assert response.status_code == 422
    mock_run_case.assert_not_called()


def test_diagnose_invalid_labs_json_returns_422(mock_run_case):
    response = client.post(
        "/diagnose",
        data={"age": 45, "sex": "male", "labs": "{not valid json"},
    )

    assert response.status_code == 422
    assert "labs" in response.json()["detail"]
    mock_run_case.assert_not_called()


def test_diagnose_internal_error_returns_500_without_stack_trace(monkeypatch):
    monkeypatch.setattr(routes, "run_case", MagicMock(side_effect=RuntimeError("secret internal detail")))

    response = client.post("/diagnose", data={"age": 45, "sex": "male"})

    assert response.status_code == 500
    body = response.json()
    assert "secret internal detail" not in json.dumps(body)
    assert body["detail"] == "Internal server error while processing the case"


# --------------------------------------------------------------------------
# GET /diagnose/{case_id}
# --------------------------------------------------------------------------


def test_get_diagnosis_not_found():
    response = client.get("/diagnose/does-not-exist")
    assert response.status_code == 404


def test_get_diagnosis_found_after_diagnose(mock_run_case):
    client.post("/diagnose", data={"case_id": "CASE001", "age": 45, "sex": "male"})

    response = client.get("/diagnose/CASE001")

    assert response.status_code == 200
    assert response.json()["case_id"] == "CASE001"


def test_imaging_findings_round_trip_through_diagnose_and_get(monkeypatch):
    """The clinician report needs the imaging findings (incl. an
    unsupported modality, rendered as "not assessed"), so they must survive
    POST /diagnose -> storage -> GET /diagnose/{id}."""
    report = SAMPLE_REPORT.model_copy(
        update={
            "imaging": ImagingFindings(
                modality="unspecified medical image",
                findings=["no findings model available for this modality (supported: MRI brain scan)"],
                confidence=0.0,
                unsupported_modality=True,
            )
        }
    )
    monkeypatch.setattr(routes, "run_case", MagicMock(return_value=report))

    posted = client.post("/diagnose", data={"case_id": "CASE001", "age": 45, "sex": "male"}).json()
    fetched = client.get("/diagnose/CASE001").json()

    assert posted["imaging"]["unsupported_modality"] is True
    assert posted["imaging"]["modality"] == "unspecified medical image"
    assert fetched["imaging"] == posted["imaging"]


def test_report_without_imaging_serializes_imaging_as_null(mock_run_case):
    response = client.post("/diagnose", data={"case_id": "CASE001", "age": 45, "sex": "male"})

    assert response.json()["imaging"] is None  # old/no-image reports stay valid


# --------------------------------------------------------------------------
# POST /review/{case_id}
# --------------------------------------------------------------------------


def test_review_not_found():
    response = client.post(
        "/review/does-not-exist",
        json={"decision": "accept", "reviewer": "Dr. Smith"},
    )
    assert response.status_code == 404


def test_review_accept(mock_run_case):
    client.post("/diagnose", data={"case_id": "CASE001", "age": 45, "sex": "male"})

    response = client.post(
        "/review/CASE001",
        json={"decision": "accept", "reviewer": "Dr. Smith"},
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.pop("timestamp"), str) and body["case_id"]  # server-assigned, just check it's there
    assert body == {
        "case_id": "CASE001",
        "decision": "accept",
        "reviewer": "Dr. Smith",
        "annotation": None,
        "rating": None,
        "recorded": True,
    }


def test_review_annotate_without_annotation_returns_422(mock_run_case):
    client.post("/diagnose", data={"case_id": "CASE001", "age": 45, "sex": "male"})

    response = client.post(
        "/review/CASE001",
        json={"decision": "annotate", "reviewer": "Dr. Smith"},
    )

    assert response.status_code == 422


def test_review_annotate_with_annotation_succeeds(mock_run_case):
    client.post("/diagnose", data={"case_id": "CASE001", "age": 45, "sex": "male"})

    response = client.post(
        "/review/CASE001",
        json={"decision": "annotate", "annotation": "Consider MRI follow-up.", "reviewer": "Dr. Smith"},
    )

    assert response.status_code == 200
    assert response.json()["annotation"] == "Consider MRI follow-up."


# --------------------------------------------------------------------------
# Async analysis: POST /diagnose?async=true + GET /diagnose/{case_id}/status
# --------------------------------------------------------------------------

import threading  # noqa: E402
import time  # noqa: E402

from backend.services import jobs as jobs_module  # noqa: E402
from backend.services.jobs import JobRegistry  # noqa: E402


@pytest.fixture
def fresh_registry(monkeypatch):
    reg = JobRegistry(workers=2)
    monkeypatch.setattr(routes, "registry", reg)
    monkeypatch.setattr(jobs_module, "log_event", lambda *a, **k: None)
    return reg


def _poll_status(case_id, want, timeout=5.0):
    end = time.time() + timeout
    body = None
    while time.time() < end:
        body = client.get(f"/diagnose/{case_id}/status").json()
        if body["status"] == want:
            return body
        time.sleep(0.02)
    raise AssertionError(f"never reached {want!r}; last status: {body}")


def _progress_run_case(case, progress=None):
    """A fake pipeline that reports progress the way the real orchestrator does."""
    progress.planned(["clinical", "imaging", "rag", "evidence", "fusion"])
    progress.step_started("clinical")
    progress.step_finished("clinical", "done", "2 symptom(s)", 10)
    progress.step_skipped("imaging", "no image attached")
    for step in ("rag", "evidence", "fusion"):
        progress.step_started(step)
        progress.step_finished(step, "done", f"{step} ok", 5)
    progress.step_skipped("optimizer", "not needed: confidence 80% is at or above the 65% threshold")
    return SAMPLE_REPORT


def test_async_diagnose_returns_202_immediately_and_status_reaches_completed_with_the_report(monkeypatch, fresh_registry):
    release = threading.Event()

    def slow(case, progress=None):
        release.wait(5)
        return _progress_run_case(case, progress)

    monkeypatch.setattr(routes, "run_case", slow)
    response = client.post("/diagnose?async=true", data={"case_id": "A1", "age": "45", "sex": "male", "clinical_text": "headache"})

    assert response.status_code == 202
    assert response.json() == {"case_id": "A1", "status": "queued", "status_url": "/diagnose/A1/status"}
    assert client.get("/diagnose/A1/status").json()["report"] is None  # still running: no report yet
    assert client.get("/diagnose/A1").status_code == 404                 # and none stored yet
    release.set()

    done = _poll_status("A1", "completed")
    assert done["report"]["diagnoses"][0]["name"] == "Tension-type headache"
    steps = {s["name"]: s for s in done["steps"]}
    assert steps["clinical"]["detail"] == "2 symptom(s)" and steps["clinical"]["status"] == "done"
    assert steps["imaging"]["status"] == "skipped" and steps["optimizer"]["status"] == "skipped"
    assert client.get("/diagnose/A1").status_code == 200                 # persisted like the sync path


def test_async_run_persists_case_report_and_breakdown_like_the_sync_path(monkeypatch, fresh_registry):
    monkeypatch.setattr(routes, "run_case", _progress_run_case)
    client.post("/diagnose?async=1", data={"case_id": "A2", "age": "30", "sex": "female", "clinical_text": "cough"})
    _poll_status("A2", "completed")
    assert storage.get_report("A2") is not None
    assert storage.get_case("A2").clinical_text == "cough"


def test_async_failure_is_reported_generically_and_marks_the_case_failed(monkeypatch, fresh_registry):
    def boom(case, progress=None):
        raise RuntimeError("secret provider detail: billing account 99")

    monkeypatch.setattr(routes, "run_case", boom)
    client.post("/diagnose?async=true", data={"case_id": "A3", "age": "30", "sex": "female", "clinical_text": "x"})
    failed = _poll_status("A3", "failed")
    assert "99" not in json.dumps(failed) and failed["error"]
    assert failed["report"] is None


def test_a_second_async_submission_of_a_running_case_is_a_409(monkeypatch, fresh_registry):
    release = threading.Event()
    monkeypatch.setattr(routes, "run_case", lambda case, progress=None: (release.wait(5), SAMPLE_REPORT)[1])
    data = {"case_id": "A4", "age": "30", "sex": "female", "clinical_text": "x"}
    assert client.post("/diagnose?async=true", data=data).status_code == 202
    assert client.post("/diagnose?async=true", data=data).status_code == 409
    release.set()
    _poll_status("A4", "completed")


def test_async_still_validates_the_request_synchronously(monkeypatch, fresh_registry):
    monkeypatch.setattr(routes, "run_case", _progress_run_case)
    assert client.post("/diagnose?async=true", data={"sex": "male"}).status_code == 422
    assert client.post("/diagnose?async=true", data={"age": "5", "sex": "male", "labs": "{bad"}).status_code == 422


def test_status_for_an_unknown_case_is_404():
    assert client.get("/diagnose/never-submitted/status").status_code == 404


def test_status_falls_back_to_the_stored_report_when_the_job_is_gone(mock_run_case, fresh_registry):
    client.post("/diagnose", data={"case_id": "S1", "age": "45", "sex": "male", "clinical_text": "x"})  # sync run, no job
    body = client.get("/diagnose/S1/status").json()
    assert body["status"] == "completed" and body["report"]["case_id"] == "CASE001" and body["steps"] == []


def test_status_reports_an_interrupted_run_when_the_case_exists_but_no_job_or_report(fresh_registry):
    from backend.models.schemas import CaseInput, Patient

    storage.save_case(CaseInput(case_id="S2", patient=Patient(age=40, sex="male"), clinical_text="x"), status="pending")
    body = client.get("/diagnose/S2/status").json()
    assert body["status"] == "failed" and "interrupted" in body["error"] and body["report"] is None


def test_sync_diagnose_is_unchanged_when_async_is_false(mock_run_case):
    response = client.post("/diagnose?async=false", data={"case_id": "Z1", "age": "45", "sex": "male", "clinical_text": "x"})
    assert response.status_code == 200 and response.json()["case_id"] == "CASE001"
    mock_run_case.assert_called_once()
    assert "progress" not in mock_run_case.call_args.kwargs  # the sync path passes no sink
