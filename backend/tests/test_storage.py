"""Tests for backend.services.storage.

Every test runs against an isolated temp SQLite file (never the real
dev database at DATABASE_URL) via the autouse fixture below, which
monkeypatches storage.DB_PATH and re-runs _init_tables() there.
"""

import pytest

import backend.services.storage as storage
from backend.models.schemas import CaseInput, FinalReport, Patient


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test_storage.db"))
    storage._init_tables()


CASE = CaseInput(
    case_id="CASE001",
    patient=Patient(age=45, sex="male"),
    clinical_text="headache and dizziness, history of hypertension",
    history=["hypertension"],
)

REPORT = FinalReport(
    case_id="CASE001",
    diagnoses=[{"name": "Tension-type headache", "confidence": 0.8, "supporting_evidence": [], "rank": 1}],
    confidence=0.8,
    evidence=[{"type": "rag", "text": "evidence text", "source": "synthetic", "score": 0.9}],
    review_required=False,
    conflicts=[],
    iteration_count=2,
)


# --------------------------------------------------------------------------
# Table creation
# --------------------------------------------------------------------------


def test_init_tables_creates_all_four_tables():
    with storage._get_connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    table_names = {row["name"] for row in rows}
    assert {"cases", "reports", "reviews", "agent_logs"} <= table_names


def test_init_tables_is_idempotent():
    # Calling it again (e.g. a second process start) must not raise.
    storage._init_tables()
    storage._init_tables()


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------


def test_save_and_get_case_round_trips():
    storage.save_case(CASE)

    fetched = storage.get_case("CASE001")

    assert fetched is not None
    assert fetched.case_id == "CASE001"
    assert fetched.patient.age == 45
    assert fetched.patient.sex == "male"
    assert fetched.clinical_text == CASE.clinical_text
    assert fetched.history == ["hypertension"]


def test_get_case_returns_none_when_not_found():
    assert storage.get_case("does-not-exist") is None


def test_save_case_default_status_is_pending():
    storage.save_case(CASE)
    with storage._get_connection() as conn:
        row = conn.execute("SELECT status FROM cases WHERE case_id = ?", ("CASE001",)).fetchone()
    assert row["status"] == "pending"


def test_save_case_updates_status_without_resetting_created_at():
    storage.save_case(CASE, status="pending")
    with storage._get_connection() as conn:
        first = conn.execute("SELECT created_at, status FROM cases WHERE case_id = ?", ("CASE001",)).fetchone()

    storage.save_case(CASE, status="completed")
    with storage._get_connection() as conn:
        second = conn.execute("SELECT created_at, status FROM cases WHERE case_id = ?", ("CASE001",)).fetchone()

    assert second["status"] == "completed"
    assert second["created_at"] == first["created_at"]  # preserved across the update


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------


def test_save_and_get_report_round_trips():
    storage.save_report("CASE001", REPORT)

    fetched = storage.get_report("CASE001")

    assert fetched is not None
    assert fetched.case_id == "CASE001"
    assert fetched.confidence == 0.8
    assert fetched.iteration_count == 2
    assert fetched.diagnoses == REPORT.diagnoses
    assert fetched.evidence == REPORT.evidence


def test_get_report_returns_none_when_not_found():
    assert storage.get_report("does-not-exist") is None


def test_save_report_upserts_replacing_previous_report_for_same_case():
    storage.save_report("CASE001", REPORT)

    updated_report = REPORT.model_copy(update={"confidence": 0.4, "iteration_count": 3})
    storage.save_report("CASE001", updated_report)

    fetched = storage.get_report("CASE001")
    assert fetched.confidence == 0.4
    assert fetched.iteration_count == 3

    with storage._get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM reports WHERE case_id = ?", ("CASE001",)).fetchone()
    assert count["n"] == 1  # replaced, not appended


# --------------------------------------------------------------------------
# reviews
# --------------------------------------------------------------------------


def test_save_review_then_get_reviews_round_trips():
    stored = storage.save_review("CASE001", {"decision": "accept", "annotation": None, "reviewer": "Dr. Smith"})

    reviews = storage.get_reviews("CASE001")

    assert len(reviews) == 1
    assert reviews[0]["id"] == stored["id"]
    assert reviews[0]["case_id"] == "CASE001"
    assert reviews[0]["decision"] == "accept"
    assert reviews[0]["reviewer"] == "Dr. Smith"
    assert reviews[0]["annotation"] is None
    assert reviews[0]["timestamp"] == stored["timestamp"]


def test_get_reviews_returns_empty_list_when_never_reviewed():
    assert storage.get_reviews("does-not-exist") == []


def test_get_reviews_appends_rather_than_overwrites():
    storage.save_review("CASE001", {"decision": "annotate", "annotation": "first note", "reviewer": "Dr. A"})
    storage.save_review("CASE001", {"decision": "override", "annotation": "second note", "reviewer": "Dr. B"})

    reviews = storage.get_reviews("CASE001")

    assert len(reviews) == 2
    assert reviews[0]["reviewer"] == "Dr. B"  # most recent first
    assert reviews[1]["reviewer"] == "Dr. A"


# --------------------------------------------------------------------------
# agent_logs
# --------------------------------------------------------------------------


def test_log_agent_event_then_get_agent_logs_round_trips():
    storage.log_agent_event(
        case_id="CASE001", agent_name="clinical", iteration=0, status="success", latency=12.5, confidence=None
    )

    logs = storage.get_agent_logs("CASE001")

    assert len(logs) == 1
    assert logs[0]["case_id"] == "CASE001"
    assert logs[0]["agent_name"] == "clinical"
    assert logs[0]["iteration"] == 0
    assert logs[0]["status"] == "success"
    assert logs[0]["latency"] == 12.5
    assert logs[0]["confidence"] is None


def test_get_agent_logs_returns_multiple_events_in_recorded_order():
    storage.log_agent_event(case_id="CASE001", agent_name="clinical", iteration=0, status="success", latency=10.0)
    storage.log_agent_event(case_id="CASE001", agent_name="rag", iteration=0, status="success", latency=20.0)
    storage.log_agent_event(
        case_id="CASE001", agent_name="optimizer", iteration=1, status="success", latency=None, confidence=0.5
    )

    logs = storage.get_agent_logs("CASE001")

    assert [log["agent_name"] for log in logs] == ["clinical", "rag", "optimizer"]
    assert logs[2]["iteration"] == 1
    assert logs[2]["confidence"] == 0.5


def test_get_agent_logs_returns_empty_list_when_no_events():
    assert storage.get_agent_logs("does-not-exist") == []


def test_agent_logs_scoped_per_case():
    storage.log_agent_event(case_id="CASE001", agent_name="clinical", iteration=0, status="success")
    storage.log_agent_event(case_id="CASE002", agent_name="clinical", iteration=0, status="success")

    assert len(storage.get_agent_logs("CASE001")) == 1
    assert len(storage.get_agent_logs("CASE002")) == 1


# --------------------------------------------------------------------------
# Phase 13/14 schema additions
# --------------------------------------------------------------------------


def test_init_tables_creates_case_memory_log_and_weight_history_tables():
    with storage._get_connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    table_names = {row["name"] for row in rows}
    assert {"case_memory_log", "confidence_weight_history"} <= table_names


def test_reviews_rating_round_trips():
    stored = storage.save_review(
        "CASE001", {"decision": "accept", "annotation": None, "reviewer": "Dr. Smith", "rating": 4}
    )
    assert stored["rating"] == 4

    reviews = storage.get_reviews("CASE001")
    assert reviews[0]["rating"] == 4


def test_reviews_rating_defaults_to_none_when_omitted():
    stored = storage.save_review("CASE001", {"decision": "accept", "annotation": None, "reviewer": "Dr. Smith"})
    assert stored["rating"] is None


def test_save_report_persists_and_returns_confidence_breakdown():
    breakdown = {"s_rag": 0.7, "s_web": 0.5, "agreement": 1.0, "confidence": 0.62}
    storage.save_report("CASE001", REPORT, confidence_breakdown=breakdown)

    assert storage.get_report_breakdown("CASE001") == breakdown


def test_get_report_breakdown_none_when_never_saved():
    storage.save_report("CASE001", REPORT)  # no breakdown passed
    assert storage.get_report_breakdown("CASE001") is None


def test_close_case_sets_status_closed():
    storage.save_case(CASE, status="completed")
    storage.close_case("CASE001")
    with storage._get_connection() as conn:
        row = conn.execute("SELECT status FROM cases WHERE case_id = ?", ("CASE001",)).fetchone()
    assert row["status"] == "closed"


def test_log_case_memory_embedding_round_trips_both_memory_types():
    storage.log_case_memory_embedding(case_id="CASE001", review_id=1, qdrant_point_id="point-a", memory_type="text")
    storage.log_case_memory_embedding(case_id="CASE001", review_id=1, qdrant_point_id="point-b", memory_type="visual")

    rows = storage.get_case_memory_log("CASE001")
    assert len(rows) == 2
    assert {r["memory_type"] for r in rows} == {"text", "visual"}
    assert {r["qdrant_point_id"] for r in rows} == {"point-a", "point-b"}


def test_log_case_memory_embedding_rejects_invalid_memory_type():
    with pytest.raises(ValueError):
        storage.log_case_memory_embedding(case_id="CASE001", review_id=None, qdrant_point_id="x", memory_type="bogus")


def test_weight_history_round_trips_and_current_row_is_most_recent():
    assert storage.get_current_weight_row() is None  # no recalibration yet

    storage.save_weight_update(0.4, 0.3, 0.3, num_reviews_used=10, notes="initial")
    storage.save_weight_update(0.5, 0.25, 0.25, num_reviews_used=15, notes="second pass")

    current = storage.get_current_weight_row()
    assert current["w1"] == 0.5
    assert current["num_reviews_used"] == 15

    history = storage.get_weight_history()
    assert len(history) == 2
    assert history[0]["notes"] == "initial"
    assert history[1]["notes"] == "second pass"


def test_get_reports_with_reviews_joins_breakdown_and_decision():
    storage.save_report("CASE001", REPORT, confidence_breakdown={"s_rag": 0.7, "s_web": 0.5, "agreement": 1.0, "confidence": 0.62})
    storage.save_review("CASE001", {"decision": "accept", "annotation": None, "reviewer": "Dr. Smith"})

    rows = storage.get_reports_with_reviews()

    assert len(rows) == 1
    assert rows[0]["case_id"] == "CASE001"
    assert rows[0]["decision"] == "accept"
    assert rows[0]["predicted_confidence"] == 0.62
    assert rows[0]["s_rag"] == 0.7


def test_get_reports_with_reviews_excludes_reports_without_breakdown():
    storage.save_report("CASE001", REPORT)  # no breakdown
    storage.save_review("CASE001", {"decision": "accept", "annotation": None, "reviewer": "Dr. Smith"})

    assert storage.get_reports_with_reviews() == []
