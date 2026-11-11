"""Tests for the LLM-judge metrics. No model is called: ``ask`` is a fake that
returns canned JSON, so these pin the SCORING logic (parsing, citation
resolution, average precision, error handling), not the judge's opinions."""

import asyncio
import json

import pytest

from backend.tests import llm_judge as j
from backend.tests import run_llm_judge as r


def run(coro):
    return asyncio.run(coro)


def fake(reply):
    """ask() returning a fixed reply (str) or computed from the prompt (callable)."""
    async def ask(prompt):
        return reply(prompt) if callable(reply) else reply
    return ask


REPORT = {
    "case_id": "X",
    "confidence": 0.8,
    "review_required": False,
    "diagnoses": [
        {"name": "Pneumothorax", "rank": 1, "supporting_evidence": ["[RAG2] air in the pleural space", "[LIVE1] x"]},
        {"name": "Pleural effusion", "rank": 2, "supporting_evidence": []},
    ],
    "evidence": [
        {"type": "rag", "text": "first rag passage"},
        {"type": "rag", "text": "second   rag\npassage"},
        {"type": "live", "title": "Paper A", "summary": "about air"},
    ],
    "imaging": {"modality": "chest X-ray", "findings": ["pneumothorax"], "unsupported_modality": False},
}
CASE = {"case_id": "X", "patient": {"age": 30, "sex": "male"}, "clinical_text": "sudden chest pain"}


# ---- parsing ---------------------------------------------------------------
def test_parse_json_strips_fences_and_prose():
    assert j.parse_json('Sure!\n```json\n{"a": 1}\n```\nDone') == {"a": 1}


def test_parse_json_rejects_non_json():
    with pytest.raises(ValueError):
        j.parse_json("no braces here")


def test_ask_json_retries_once_then_succeeds():
    replies = iter(["garbage", '{"ok": true}'])
    assert run(j.ask_json(fake(lambda _: next(replies)), "p")) == {"ok": True}


def test_ask_json_gives_up_after_retry():
    with pytest.raises(ValueError):
        run(j.ask_json(fake("still garbage"), "p"))


# ---- citations -------------------------------------------------------------
def test_resolve_citation_uses_displayed_evidence_not_inline_text():
    rag, live = j.split_evidence(REPORT)
    assert rag[1] == "second rag passage"  # whitespace normalised
    assert j.resolve_citation("[RAG2] whatever the model wrote", rag, live) == "second rag passage"
    assert j.resolve_citation("[LIVE1] zzz", rag, live).startswith("Paper A.")


def test_resolve_citation_flags_out_of_range_index():
    rag, live = j.split_evidence(REPORT)
    assert j.resolve_citation("[RAG9] inline text", rag, live) == "(unresolved citation) inline text"


# ---- average precision (RAGAS context precision) ---------------------------
@pytest.mark.parametrize(
    "flags,expected",
    [
        ([True, True, True], 1.0),
        ([False, False], 0.0),
        ([True, False, True], (1 / 1 + 2 / 3) / 2),
        ([False, True], 0.5),
        ([], None),
    ],
)
def test_average_precision(flags, expected):
    assert j.average_precision(flags) == pytest.approx(expected) if expected is not None else j.average_precision(flags) is None


def test_rank_matters_for_precision():
    assert j.average_precision([True, False]) > j.average_precision([False, True])


# ---- diagnosis -------------------------------------------------------------
def test_abstain_case_is_correct_only_when_flagged_for_review():
    key = {"kind": "abstain"}
    never_called = fake(lambda _: pytest.fail("abstain must not call the judge"))
    flagged = {**REPORT, "review_required": True}
    unflagged_high = {**REPORT, "review_required": False, "confidence": 0.9}
    low_conf = {**REPORT, "review_required": False, "confidence": 0.4}
    assert run(j.judge_diagnosis(never_called, flagged, key))["correct"] is True
    assert run(j.judge_diagnosis(never_called, low_conf, key))["correct"] is True
    assert run(j.judge_diagnosis(never_called, unflagged_high, key))["correct"] is False


def test_differential_uses_judge_verdict():
    key = {"kind": "differential", "differential": ["pneumothorax"]}
    yes = fake('{"match": true, "matched_system_diagnosis": "Pneumothorax", "reason": "same"}')
    no = fake('{"match": false, "matched_system_diagnosis": null, "reason": "different"}')
    assert run(j.judge_diagnosis(yes, REPORT, key)) == {"correct": True, "matched": "Pneumothorax", "method": "judge"}
    assert run(j.judge_diagnosis(no, REPORT, key))["correct"] is False


def test_diagnosis_prompt_shows_only_top3_and_the_key():
    seen = []
    key = {"kind": "differential", "differential": ["pneumothorax"]}
    many = {**REPORT, "diagnoses": [{"name": f"D{i}", "rank": i} for i in range(1, 6)]}
    run(j.judge_diagnosis(fake(lambda p: seen.append(p) or '{"match": false}'), many, key))
    assert "D3" in seen[0] and "D4" not in seen[0] and "pneumothorax" in seen[0]


def test_no_diagnoses_is_incorrect_without_calling_judge():
    key = {"kind": "differential", "differential": ["x"]}
    assert run(j.judge_diagnosis(fake(lambda _: pytest.fail("no call")), {**REPORT, "diagnoses": []}, key))["correct"] is False


# ---- faithfulness ----------------------------------------------------------
def test_faithfulness_averages_verdicts():
    reply = json.dumps({"verdicts": [
        {"diagnosis": "Pneumothorax", "verdict": "supported"},
        {"diagnosis": "Pleural effusion", "verdict": "unsupported"},
    ]})
    assert run(j.judge_faithfulness(fake(reply), REPORT, CASE))["score"] == 0.5


def test_faithfulness_partial_counts_half_and_ignores_unknown_verdicts():
    reply = json.dumps({"verdicts": [{"verdict": "partial"}, {"verdict": "banana"}]})
    assert run(j.judge_faithfulness(fake(reply), REPORT, CASE))["score"] == 0.5


def test_faithfulness_prompt_contains_resolved_text_and_marks_nothing_cited():
    seen = []
    run(j.judge_faithfulness(fake(lambda p: seen.append(p) or '{"verdicts": []}'), REPORT, CASE))
    assert "second rag passage" in seen[0] and "(none cited)" in seen[0]


# ---- context precision -----------------------------------------------------
def test_context_precision_scores_by_rank():
    out = run(j.judge_context_precision(fake('{"relevant": [false, true]}'), REPORT, CASE))
    assert out["score"] == pytest.approx(0.5) and out["relevant"] == [False, True]


def test_context_precision_rejects_wrong_flag_count():
    with pytest.raises(ValueError):
        run(j.judge_context_precision(fake('{"relevant": [true]}'), REPORT, CASE))


def test_context_precision_none_without_rag_evidence():
    out = run(j.judge_context_precision(fake(lambda _: pytest.fail("no call")), {**REPORT, "evidence": []}, CASE))
    assert out["score"] is None


# ---- unsupported claims ----------------------------------------------------
def test_unsupported_claims_counted_and_blanks_dropped():
    out = run(j.judge_unsupported_claims(fake('{"unsupported_claims": ["recent trauma", "  "]}'), REPORT, CASE))
    assert out == {"count": 1, "claims": ["recent trauma"]}


# ---- whole report + aggregation --------------------------------------------
def test_judge_report_records_a_failing_metric_and_keeps_going():
    def reply(prompt):
        if "Acceptable diagnoses" in prompt:
            raise RuntimeError("quota")
        if "Retrieved passages" in prompt:
            return '{"relevant": [true, true]}'
        if "Reply: {\"verdicts\"" in prompt or "verdicts" in prompt:
            return '{"verdicts": [{"verdict": "supported"}]}'
        return '{"unsupported_claims": []}'

    key = {"kind": "differential", "differential": ["pneumothorax"]}
    res = run(j.judge_report(fake(reply), REPORT, CASE, key))
    assert "RuntimeError" in res["diagnosis"]["error"]
    assert res["faithfulness"]["score"] == 1.0
    assert res["context_precision"]["score"] == 1.0
    assert res["unsupported"]["count"] == 0


def test_aggregate_counts_errors_instead_of_dropping_them_silently():
    results = [
        {"diagnosis": {"correct": True}, "faithfulness": {"score": 1.0}, "context_precision": {"score": 0.5}, "unsupported": {"count": 0}},
        {"diagnosis": {"correct": False}, "faithfulness": {"score": 0.0}, "context_precision": {"score": None}, "unsupported": {"count": 2}},
        {"diagnosis": {"error": "x"}, "faithfulness": {"error": "y"}, "context_precision": {"error": "z"}, "unsupported": {"error": "w"}},
    ]
    agg = r.aggregate(results)
    assert agg["diagnosis_correct"] == 0.5 and agg["diagnosis_n"] == 2
    assert agg["faithfulness"] == 0.5 and agg["context_precision"] == 0.5 and agg["context_precision_n"] == 1
    assert agg["cases_with_unsupported_claims"] == 1 and agg["unsupported_claims_total"] == 2
    assert agg["judge_errors"] == 4 and agg["n_cases"] == 3


def test_markdown_states_the_limits():
    agg = r.aggregate([])
    md = r.render_md([], agg, "gemini:test")
    assert "NOT clinician-verified" in md and "not clinical validation" in md and "gemini:test" in md
