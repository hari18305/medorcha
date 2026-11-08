"""Tests for backend.rag.reranker's calibrated relevance score.

The model itself is faked (no weights, no inference); what is tested is the
logit -> 0-1 rescale, which is what feeds S_rag in the confidence formula.
"""

import math

import pytest

import backend.rag.reranker as reranker


def test_the_center_logit_maps_to_one_half():
    assert reranker.relevance_score(reranker.SCORE_CENTER) == pytest.approx(0.5)


def test_score_is_monotonic_in_the_logit():
    logits = [-20.0, -12.0, -9.0, -7.0, -5.0, -2.0, 1.0, 6.0]
    scores = [reranker.relevance_score(x) for x in logits]

    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores)


def test_score_stays_strictly_inside_zero_one_even_for_absurd_logits():
    # a plain 1/(1+exp(-x)) would raise OverflowError for very negative x
    assert 0.0 <= reranker.relevance_score(-1e6) < 1e-6
    assert 1 - 1e-6 < reranker.relevance_score(1e6) <= 1.0
    assert not math.isnan(reranker.relevance_score(-1e6))


def test_realistic_logits_use_the_whole_range_instead_of_collapsing_to_zero():
    """The reason for the rescale: against the real knowledge base, on-topic
    symptom queries give logits around -4 to -8 and off-topic ones about -9
    to -10.5. A plain sigmoid made ALL of these ~0.00 (S_rag ~ 0.04, so
    confidence could never reach the 0.65 threshold)."""
    on_topic = [reranker.relevance_score(x) for x in (-4.2, -5.7, -6.5)]
    off_topic = [reranker.relevance_score(x) for x in (-9.7, -10.4)]

    assert all(0.25 < s < 0.9 for s in on_topic)
    assert all(s < 0.25 for s in off_topic)  # measured off-topic range ~0.15-0.29
    assert min(on_topic) > max(off_topic)
    # the old (uncalibrated) scale really did flatten the on-topic ones to ~0
    assert all(reranker._sigmoid(x) < 0.02 for x in (-5.7, -6.5))


def test_rerank_orders_by_the_raw_logit_and_reports_the_calibrated_score(monkeypatch):
    class _Model:
        def predict(self, pairs):
            return [-9.0, -4.0, -7.0]

    monkeypatch.setattr(reranker, "_get_model", lambda: _Model())
    candidates = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.1}, {"text": "c", "score": 0.5}]

    result = reranker.rerank("q", candidates, top_k=3)

    assert [c["text"] for c in result] == ["b", "c", "a"]  # ranking unchanged by the rescale
    assert result[1]["score"] == pytest.approx(0.5)         # logit -7 -> 0.5
    assert result[0]["score"] > result[1]["score"] > result[2]["score"]


def test_rerank_of_nothing_is_empty():
    assert reranker.rerank("q", []) == []
