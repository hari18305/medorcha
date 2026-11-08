"""Tests for backend.services.weight_tuner (Phase 14).

Every test runs against an isolated temp SQLite file (never the real dev
database), seeded directly via storage.save_report()/save_review() --
no LLM/network calls anywhere in this module, so nothing to mock beyond
that isolation.
"""

import pytest

import backend.services.storage as storage
import backend.services.weight_tuner as weight_tuner
from backend.models.schemas import CaseInput, FinalReport, Patient
from backend.services.confidence import DEFAULT_WEIGHTS, get_current_weights
from backend.services.weight_tuner import collect_review_outcomes, recalibrate_weights


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "test_weight_tuner.db"))
    storage._init_tables()


def _seed_case(case_id: str, breakdown: dict, decision: str) -> None:
    storage.save_case(CaseInput(case_id=case_id, patient=Patient(age=40, sex="female")), status="completed")
    report = FinalReport(
        case_id=case_id,
        diagnoses=[{"name": "x", "confidence": breakdown["confidence"], "supporting_evidence": [], "rank": 1}],
        confidence=breakdown["confidence"],
        evidence=[],
        review_required=False,
    )
    storage.save_report(case_id, report, confidence_breakdown=breakdown)
    storage.save_review(case_id, {"decision": decision, "annotation": None, "reviewer": "Dr. Test"})


# --------------------------------------------------------------------------
# collect_review_outcomes()
# --------------------------------------------------------------------------


def test_collect_review_outcomes_maps_accept_and_override_correctly():
    _seed_case("C1", {"s_rag": 0.8, "s_web": 0.5, "agreement": 1.0, "confidence": 0.7}, "accept")
    _seed_case("C2", {"s_rag": 0.2, "s_web": 0.1, "agreement": 0.5, "confidence": 0.3}, "override")

    outcomes = collect_review_outcomes()

    assert len(outcomes) == 2
    by_case = {o["s_rag"]: o for o in outcomes}
    assert by_case[0.8]["outcome"] == 1.0
    assert by_case[0.2]["outcome"] == 0.0


def test_collect_review_outcomes_skips_annotate_decisions():
    _seed_case("C1", {"s_rag": 0.8, "s_web": 0.5, "agreement": 1.0, "confidence": 0.7}, "annotate")

    assert collect_review_outcomes() == []


def test_collect_review_outcomes_skips_reports_without_a_breakdown():
    storage.save_case(CaseInput(case_id="C1", patient=Patient(age=40, sex="female")), status="completed")
    storage.save_report("C1", FinalReport(case_id="C1", diagnoses=[], confidence=0.5, evidence=[], review_required=False))
    storage.save_review("C1", {"decision": "accept", "annotation": None, "reviewer": "Dr. Test"})

    assert collect_review_outcomes() == []


def test_collect_review_outcomes_skips_cases_never_reviewed():
    storage.save_case(CaseInput(case_id="C1", patient=Patient(age=40, sex="female")), status="completed")
    storage.save_report(
        "C1",
        FinalReport(case_id="C1", diagnoses=[], confidence=0.5, evidence=[], review_required=False),
        confidence_breakdown={"s_rag": 0.5, "s_web": 0.5, "agreement": 1.0, "confidence": 0.5},
    )

    assert collect_review_outcomes() == []


# --------------------------------------------------------------------------
# recalibrate_weights()
# --------------------------------------------------------------------------


def test_recalibrate_weights_below_min_reviews_is_a_noop():
    for i in range(5):
        _seed_case(f"C{i}", {"s_rag": 0.8, "s_web": 0.5, "agreement": 1.0, "confidence": 0.7}, "accept")

    result = recalibrate_weights(min_reviews=10)

    assert result is None
    assert get_current_weights() == DEFAULT_WEIGHTS  # untouched
    assert storage.get_weight_history() == []


def test_recalibrate_weights_with_clear_pattern_shifts_w1_relative_to_w2():
    """A clear pattern: whenever s_rag is high and the case was
    accepted, or s_rag is low and the case was overridden, s_rag is a
    genuinely predictive signal. s_web is deliberately uncorrelated
    with outcome (sometimes high, sometimes low, regardless of
    accept/override) -- so a real refit should raise w1 relative to w2.
    """
    accepted_high_rag = [
        {"s_rag": 0.9, "s_web": 0.2, "agreement": 1.0, "confidence": 0.7},
        {"s_rag": 0.85, "s_web": 0.8, "agreement": 1.0, "confidence": 0.7},
        {"s_rag": 0.95, "s_web": 0.1, "agreement": 0.75, "confidence": 0.7},
        {"s_rag": 0.9, "s_web": 0.6, "agreement": 1.0, "confidence": 0.7},
        {"s_rag": 0.88, "s_web": 0.3, "agreement": 1.0, "confidence": 0.7},
        {"s_rag": 0.92, "s_web": 0.9, "agreement": 0.75, "confidence": 0.7},
    ]
    overridden_low_rag = [
        {"s_rag": 0.1, "s_web": 0.7, "agreement": 1.0, "confidence": 0.4},
        {"s_rag": 0.05, "s_web": 0.2, "agreement": 0.75, "confidence": 0.4},
        {"s_rag": 0.15, "s_web": 0.9, "agreement": 1.0, "confidence": 0.4},
        {"s_rag": 0.1, "s_web": 0.3, "agreement": 0.75, "confidence": 0.4},
        {"s_rag": 0.2, "s_web": 0.8, "agreement": 1.0, "confidence": 0.4},
        {"s_rag": 0.08, "s_web": 0.4, "agreement": 1.0, "confidence": 0.4},
    ]
    for i, breakdown in enumerate(accepted_high_rag):
        _seed_case(f"ACC{i}", breakdown, "accept")
    for i, breakdown in enumerate(overridden_low_rag):
        _seed_case(f"OVR{i}", breakdown, "override")

    result = recalibrate_weights(min_reviews=10)

    assert result is not None
    assert result["num_reviews_used"] == 12
    assert result["w1"] > result["w2"]  # s_rag is the predictive signal here
    assert result["w1"] == pytest.approx(1.0 - result["w2"] - result["w3"], abs=1e-6)


def test_recalibrated_weights_always_sum_to_one_and_respect_floor():
    for i in range(12):
        _seed_case(f"C{i}", {"s_rag": 0.9, "s_web": 0.1, "agreement": 1.0, "confidence": 0.7}, "accept")

    result = recalibrate_weights(min_reviews=10)

    assert result["w1"] + result["w2"] + result["w3"] == pytest.approx(1.0)
    assert result["w1"] >= weight_tuner.MIN_WEIGHT - 1e-6
    assert result["w2"] >= weight_tuner.MIN_WEIGHT - 1e-6
    assert result["w3"] >= weight_tuner.MIN_WEIGHT - 1e-6


def test_recalibrate_weights_persists_and_confidence_module_picks_it_up():
    for i in range(12):
        _seed_case(f"C{i}", {"s_rag": 0.9, "s_web": 0.1, "agreement": 1.0, "confidence": 0.7}, "accept")

    result = recalibrate_weights(min_reviews=10)

    history = storage.get_weight_history()
    assert len(history) == 1
    assert history[0]["num_reviews_used"] == 12

    current = get_current_weights()
    assert current["w1"] == pytest.approx(result["w1"])
    assert current["w2"] == pytest.approx(result["w2"])
    assert current["w3"] == pytest.approx(result["w3"])
