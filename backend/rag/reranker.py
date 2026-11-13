"""Cross-encoder reranking of retrieved evidence for answer relevance.

Design choice: (a) a cross-encoder model (cross-encoder/ms-marco-MiniLM-
L-6-v2), not (b) an Anthropic Claude scoring call.

Why: reranking runs on every retrieval -- including the fallback retry
in rag_agent.py -- so it needs to be fast, free of per-call API
latency/cost, and available offline, consistent with the rest of this
RAG layer being a local, deterministic retrieval pipeline (BioBERT
embeddings + Qdrant). A cross-encoder jointly encodes (query, passage)
pairs, so it scores actual query-passage relevance rather than just
re-sorting by the same embedding similarity used for retrieval -- this
is a real second-stage relevance signal, not a no-op.

Caveat: MS MARCO cross-encoders are general-domain, not medically
fine-tuned. Swapping in a biomedical cross-encoder (once one is
selected and evaluated) is a natural future upgrade; the rerank()
interface would not need to change.

Score scale (calibrated, NOT a plain sigmoid): the confidence formula uses
these scores directly as S_rag, so they have to span a useful 0-1 range.
A plain sigmoid of this model's raw logit does not: against the real
MedlinePlus/PubMed knowledge base, symptom-list queries produce logits of
roughly -4 to -10 even for clearly on-topic passages (and -9 to -10.5 for
off-topic ones), which sigmoid() flattens to ~0.00 everywhere -- capping
confidence below the 0.65 threshold for nearly every case. The raw logit
ordering IS informative (on-topic mean -6.5 vs off-topic mean -9.7, 19
queries), so the fix is a monotonic rescale, which never changes which
passage ranks first:

    score = sigmoid((logit - SCORE_CENTER) / SCORE_SCALE)

with SCORE_CENTER=-7, SCORE_SCALE=2 (on-topic ~0.16-0.80, mean ~0.56;
off-topic ~0.15-0.29). These two numbers were set BY HAND from a small
sample on the current 30-passage knowledge base -- a prototype calibration,
not a validated one. Re-fit them with `python -m backend.rag.calibrate_reranker`
whenever the knowledge base or the reranker model changes. Override with
RERANK_SCORE_CENTER / RERANK_SCORE_SCALE.
"""

import math
import os

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _float_from_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


SCORE_CENTER = _float_from_env("RERANK_SCORE_CENTER", -7.0)
SCORE_SCALE = _float_from_env("RERANK_SCORE_SCALE", 2.0)

# How the final top-k is ORDERED (which passages make the top-k is always the
# cross-encoder's call, so S_rag - the mean of those scores - never depends on
# this setting):
#   "rrf" (default) - reciprocal rank fusion of the dense (BioBERT) rank and the
#                     cross-encoder rank, k=60 (the standard constant, not tuned).
#   "ce"            - cross-encoder order only (the behaviour before 2026-11-13).
# Why: the MS MARCO cross-encoder is general-domain and sometimes puts an
# off-topic passage first (a subarachnoid-haemorrhage passage topped plain
# "headache, dizziness; history of hypertension"; pleural effusion and TB made
# the top 3 for a child's headache) where BioBERT had the right one at rank 1.
# Measured with scripts/eval_retrieval.py on 14 real agent queries: MRR@5
# 0.768 -> 0.827, hit@1 8 -> 10, hit@5 unchanged at 14/14. Helped 3 queries and
# hurt 1 (the CASE009 subdural passage moved from 2nd to 4th). n=14, developer-
# judged relevance: a regression check, not proof.
RERANK_ORDER = os.getenv("RERANK_ORDER", "rrf").strip().lower()
RRF_K = 60

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Lazily load and cache the cross-encoder model at module level."""
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def _sigmoid(x: float) -> float:
    # Numerically stable for any x (a plain 1/(1+exp(-x)) overflows for very negative x).
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def relevance_score(logit: float) -> float:
    """Raw cross-encoder logit -> calibrated 0-1 relevance score (see the
    module docstring). Monotonic, so ranking is unaffected; -7 maps to 0.5."""
    return _sigmoid((logit - SCORE_CENTER) / SCORE_SCALE)


def _rrf_order(top: list[dict]) -> list[dict]:
    """Reorder `top` (already the cross-encoder's chosen top-k, best first) by
    reciprocal rank fusion of each item's dense rank and cross-encoder rank.
    Each item carries "_dense_rank" (1-based position in the retrieval order)."""
    ce_rank = {id(item): i for i, item in enumerate(top, 1)}
    return sorted(
        top,
        key=lambda item: -(1.0 / (RRF_K + item["_dense_rank"]) + 1.0 / (RRF_K + ce_rank[id(item)])),
    )


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank `candidates` by cross-encoder relevance to `query`.

    `candidates` must be in dense-retrieval order (best cosine first), which is
    what retriever.retrieve() returns: that position is the dense rank used by
    the optional RRF ordering (see RERANK_ORDER).

    Each candidate's "score" is overwritten with the cross-encoder's
    relevance score -- its raw logit rescaled onto an interpretable (0, 1)
    range by relevance_score() -- which is what callers compare against a
    relevance threshold and what feeds S_rag. The top_k are chosen by this
    score; their final order is the cross-encoder's, or the RRF order (default).
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, candidate["text"]) for candidate in candidates]
    raw_scores = model.predict(pairs)

    reranked = [
        {**candidate, "score": relevance_score(float(raw_score)), "_dense_rank": rank}
        for rank, (candidate, raw_score) in enumerate(zip(candidates, raw_scores), 1)
    ]
    reranked.sort(key=lambda c: c["score"], reverse=True)

    top = reranked[:top_k]
    if RERANK_ORDER == "rrf" and len(top) > 1:
        top = _rrf_order(top)
    for item in top:
        del item["_dense_rank"]
    return top
