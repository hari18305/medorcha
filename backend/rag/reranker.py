"""Cross-encoder reranking of retrieved evidence for answer relevance.

Design choice: (a) a local cross-encoder model, not (b) an Anthropic Claude
scoring call. The model is chosen by RERANK_MODEL: ncbi/MedCPT-Cross-Encoder
(default; biomedical, trained on PubMed search logs, ~440 MB) or
cross-encoder/ms-marco-MiniLM-L-6-v2 (general-domain, ~90 MB, the original).

Why: reranking runs on every retrieval -- including the fallback retry
in rag_agent.py -- so it needs to be fast, free of per-call API
latency/cost, and available offline, consistent with the rest of this
RAG layer being a local, deterministic retrieval pipeline (BioBERT
embeddings + Qdrant). A cross-encoder jointly encodes (query, passage)
pairs, so it scores actual query-passage relevance rather than just
re-sorting by the same embedding similarity used for retrieval -- this
is a real second-stage relevance signal, not a no-op.

Why MedCPT is the default (2026-11-13): on 14 real agent queries
(scripts/eval_retrieval.py) it beat the general-domain MS MARCO model --
MRR@5 0.827 -> 0.863, hit@1 10 -> 11/14 (vs. MS MARCO with its cross-encoder
order: 0.768 / 8) -- and stopped it putting off-topic
passages first (pulmonary tuberculosis topped a child's headache + cold).
It costs a ~440 MB download and is NOT strictly better: the subdural-hematoma
passage for the anticoagulated-fall case (CASE009) is 3rd rather than 2nd.
n=14, developer-judged relevance: a regression check, not a benchmark.

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

with a per-model (center, scale) -- MS MARCO: -7 / 2 (on-topic ~0.16-0.80,
mean ~0.56; off-topic ~0.15-0.29); MedCPT: -12.5 / 3.5. MedCPT's raw logits sit on
a very different range (on-topic top-5 mean about -8, off-topic about -15.7), so
its constants were fitted by hand to leave average confidence where it was:
the pair (-12.5, 3.5) reproduced the MS MARCO pipeline's mean S_rag over the 14
real agent queries (0.659 -> 0.659) and over 6 off-topic queries (0.284 -> 0.288) on
the original 42-passage knowledge base. Growing it to 67 passages (and the candidate
pool from 10 to 20) lifted the same queries' mean to 0.754 and off-topic to 0.315, so
the centre was shifted to -10.4, which restores 0.659 on the 14 queries (off-topic 0.209).
They are a prototype calibration on a small sample, not a validated one: only the
AVERAGE is matched, per-query S_rag still differs between the models (up to about
0.3, i.e. 0.12 of confidence, on the 14 queries). Re-fit with
`python -m backend.rag.calibrate_reranker` whenever the knowledge base or the
reranker model changes. Override with RERANK_SCORE_CENTER / RERANK_SCORE_SCALE.
"""

import math
import os

import torch
from sentence_transformers import CrossEncoder

MS_MARCO_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MEDCPT_MODEL = "ncbi/MedCPT-Cross-Encoder"
MODEL_NAME = os.getenv("RERANK_MODEL", MEDCPT_MODEL).strip() or MEDCPT_MODEL

# (center, scale) of the logit -> 0-1 rescale, per model. An unlisted model falls
# back to the MS MARCO values and should be re-fit with calibrate_reranker.
CALIBRATION_BY_MODEL = {
    MS_MARCO_MODEL: (-7.0, 2.0),
    MEDCPT_MODEL: (-10.4, 3.5),  # centre was -12.5 for the 42-passage KB / top-10 pool; refit 2026-11-13 for 67 passages / top-20
}


def _float_from_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


_DEFAULT_CENTER, _DEFAULT_SCALE = CALIBRATION_BY_MODEL.get(MODEL_NAME, CALIBRATION_BY_MODEL[MS_MARCO_MODEL])
SCORE_CENTER = _float_from_env("RERANK_SCORE_CENTER", _DEFAULT_CENTER)
SCORE_SCALE = _float_from_env("RERANK_SCORE_SCALE", _DEFAULT_SCALE)

# How the final top-k is ORDERED (which passages make the top-k is always the
# cross-encoder's call, so S_rag - the mean of those scores - never depends on
# this setting):
#   "ce"  (default) - cross-encoder order only.
#   "rrf"           - reciprocal rank fusion of the dense (BioBERT) rank and the
#                     cross-encoder rank, k=60 (the standard constant, not tuned).
# RRF was the default for one day, to compensate for the general-domain MS MARCO
# model (MRR@5 0.768 -> 0.827). With MedCPT the gain is smaller (0.863 -> 0.895,
# hit@1 11 -> 12) and it pushes the CASE009 subdural passage from 3rd to 5th,
# the edge of the top 5, so it is off by default. Set RERANK_ORDER=rrf to use it
# (it still helps the MS MARCO model). Measured with scripts/eval_retrieval.py on
# 14 real agent queries, developer-judged relevance; n=14, not a benchmark.
RERANK_ORDER = os.getenv("RERANK_ORDER", "ce").strip().lower()
RRF_K = 60

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Lazily load and cache the cross-encoder model at module level."""
    global _model
    if _model is None:
        # Raw logits, whatever the model's own default: MS MARCO's config sets an
        # identity activation but MedCPT's does not, so sentence-transformers would
        # sigmoid it (scores of ~1e-5) and the calibration below would be nonsense.
        _model = CrossEncoder(MODEL_NAME, activation_fn=torch.nn.Identity(), max_length=512)
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
