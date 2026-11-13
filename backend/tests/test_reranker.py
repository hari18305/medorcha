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


def test_realistic_ms_marco_logits_use_the_whole_range_instead_of_collapsing_to_zero():
    """The reason for the rescale (MS MARCO model): against the real knowledge
    base, on-topic symptom queries give logits around -4 to -8 and off-topic ones
    about -9 to -10.5. A plain sigmoid made ALL of these ~0.00 (S_rag ~ 0.04, so
    confidence could never reach the 0.65 threshold)."""
    center, scale = reranker.CALIBRATION_BY_MODEL[reranker.MS_MARCO_MODEL]

    def score(x):
        return reranker._sigmoid((x - center) / scale)

    on_topic = [score(x) for x in (-4.2, -5.7, -6.5)]
    off_topic = [score(x) for x in (-9.7, -10.4)]

    assert all(0.25 < s < 0.9 for s in on_topic)
    assert all(s < 0.25 for s in off_topic)  # measured off-topic range ~0.15-0.29
    assert min(on_topic) > max(off_topic)
    # the old (uncalibrated) scale really did flatten the on-topic ones to ~0
    assert all(reranker._sigmoid(x) < 0.02 for x in (-5.7, -6.5))


def test_rerank_orders_by_the_raw_logit_and_reports_the_calibrated_score(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_ORDER", "ce")  # the pure cross-encoder order
    c = reranker.SCORE_CENTER

    class _Model:
        def predict(self, pairs):
            return [c - 2.0, c + 3.0, c]  # a, b, c

    monkeypatch.setattr(reranker, "_get_model", lambda: _Model())
    candidates = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.1}, {"text": "c", "score": 0.5}]

    result = reranker.rerank("q", candidates, top_k=3)

    assert [c_["text"] for c_ in result] == ["b", "c", "a"]  # ranking unchanged by the rescale
    assert result[1]["score"] == pytest.approx(0.5)           # the center logit -> 0.5
    assert result[0]["score"] > result[1]["score"] > result[2]["score"]


# --------------------------------------------------------------------------
# RERANK_ORDER="rrf": the cross-encoder picks the top-k SET, RRF orders it
# --------------------------------------------------------------------------


def _fake_model(monkeypatch, logits):
    class _Model:
        def predict(self, pairs):
            return list(logits)

    monkeypatch.setattr(reranker, "_get_model", lambda: _Model())


def _cands(n):
    # in dense-retrieval order: "d1" is the best cosine match
    return [{"text": f"d{i}", "score": 1.0 - i / 100} for i in range(1, n + 1)]


def test_rrf_lets_a_strong_dense_rank_beat_a_slightly_better_cross_encoder_rank(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_ORDER", "rrf")
    # CE prefers d3 (-4) over d1 (-5), but dense retrieval had d1 first and d3 last.
    _fake_model(monkeypatch, [-5.0, -9.0, -4.0])  # d1, d2, d3 -> CE order: d3, d1, d2
    result = reranker.rerank("q", _cands(3), top_k=3)

    # d3: dense rank 3 / CE rank 1; d1: dense rank 1 / CE rank 2 -> d1 wins on fused rank
    assert [c["text"] for c in result] == ["d1", "d3", "d2"]


def test_rrf_never_changes_which_passages_make_the_top_k_or_their_scores(monkeypatch):
    # S_rag is the mean of the top-k scores; ordering must not move it.
    logits = [-3.0, -8.0, -6.0, -9.0, -4.0, -7.0]
    _fake_model(monkeypatch, logits)
    monkeypatch.setattr(reranker, "RERANK_ORDER", "ce")
    ce = reranker.rerank("q", _cands(6), top_k=3)
    monkeypatch.setattr(reranker, "RERANK_ORDER", "rrf")
    rrf = reranker.rerank("q", _cands(6), top_k=3)

    assert {c["text"] for c in rrf} == {c["text"] for c in ce}
    assert sorted(c["score"] for c in rrf) == sorted(c["score"] for c in ce)
    assert sum(c["score"] for c in rrf) == pytest.approx(sum(c["score"] for c in ce))


def test_rrf_leaves_no_internal_fields_on_the_results(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_ORDER", "rrf")
    _fake_model(monkeypatch, [-5.0, -6.0])
    for item in reranker.rerank("q", _cands(2), top_k=2):
        assert "_dense_rank" not in item


def test_rrf_handles_one_and_zero_candidates(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_ORDER", "rrf")
    _fake_model(monkeypatch, [-5.0])
    assert [c["text"] for c in reranker.rerank("q", _cands(1), top_k=5)] == ["d1"]
    assert reranker.rerank("q", [], top_k=5) == []


def _reload_with_env(**env):
    """Reload the module under a temporary environment, restoring both afterwards."""
    import importlib
    import os

    saved = {k: os.environ.get(k) for k in ("RERANK_MODEL", "RERANK_ORDER", "RERANK_SCORE_CENTER", "RERANK_SCORE_SCALE")}
    for k in saved:
        os.environ.pop(k, None)
    os.environ.update(env)
    try:
        return importlib.reload(reranker), saved
    except Exception:
        _restore_env(saved)
        raise


def _restore_env(saved):
    import importlib
    import os

    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v
    importlib.reload(reranker)


def test_defaults_are_medcpt_with_cross_encoder_order():
    mod, saved = _reload_with_env()
    try:
        assert mod.MODEL_NAME == "ncbi/MedCPT-Cross-Encoder"
        assert mod.RERANK_ORDER == "ce"
        assert (mod.SCORE_CENTER, mod.SCORE_SCALE) == mod.CALIBRATION_BY_MODEL[mod.MEDCPT_MODEL]
    finally:
        _restore_env(saved)


def test_ms_marco_keeps_its_original_calibration_when_selected():
    mod, saved = _reload_with_env(RERANK_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        assert (mod.SCORE_CENTER, mod.SCORE_SCALE) == (-7.0, 2.0)
    finally:
        _restore_env(saved)


def test_explicit_score_constants_override_the_model_defaults():
    mod, saved = _reload_with_env(RERANK_SCORE_CENTER="-3", RERANK_SCORE_SCALE="1.5")
    try:
        assert (mod.SCORE_CENTER, mod.SCORE_SCALE) == (-3.0, 1.5)
    finally:
        _restore_env(saved)


def test_each_model_maps_its_own_typical_logits_into_a_useful_range():
    # Guards the reason for per-model constants: MedCPT's on-topic logits (~ -8)
    # would be ~0 under MS MARCO's constants, and off-topic ones (~ -15.7) must stay low.
    c, s = reranker.CALIBRATION_BY_MODEL[reranker.MEDCPT_MODEL]
    on, off = reranker._sigmoid((-8.0 - c) / s), reranker._sigmoid((-15.7 - c) / s)
    assert 0.55 < on < 0.8
    assert off < 0.35
    c0, s0 = reranker.CALIBRATION_BY_MODEL[reranker.MS_MARCO_MODEL]
    assert reranker._sigmoid((-8.0 - c0) / s0) < 0.4  # the wrong constants would squash MedCPT's scores


def test_model_is_loaded_for_raw_logits_not_sigmoid_scores(monkeypatch):
    # sentence-transformers applies a sigmoid to single-label models unless told
    # otherwise (MedCPT's config does not say; MS MARCO's does), which would turn
    # every logit into ~1e-5 and make the calibration meaningless.
    import torch

    seen = {}

    class _CE:
        def __init__(self, name, **kwargs):
            seen["name"], seen["kwargs"] = name, kwargs

    monkeypatch.setattr(reranker, "CrossEncoder", _CE)
    monkeypatch.setattr(reranker, "_model", None)
    reranker._get_model()

    assert seen["name"] == reranker.MODEL_NAME
    assert isinstance(seen["kwargs"]["activation_fn"], torch.nn.Identity)
    assert seen["kwargs"]["max_length"] == 512
