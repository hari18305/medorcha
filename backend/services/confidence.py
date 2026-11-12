"""Confidence scoring: blends the fusion agent's evidence signals into a
single confidence score per the spec formula:

    C = w1 * S_rag + w2 * S_web + w3 * A

  S_rag -- offline RAG retrieval signal: the mean of the top-K
           rag.evidence[].score values (0-1 cross-encoder relevance
           scores from reranker.py, rescaled by reranker.relevance_score()
           -- a hand-calibrated prototype scale, see that module). 0.0 if
           no RAG evidence.
  S_web -- live evidence signal: each live.sources[] item is weighted
           by its evidence_level, then averaged. 0.0 if no live
           evidence.
  A     -- agreement between evidence streams: 1.0 minus a fixed
           penalty per item in fusion.conflicts, floored at 0. See
           _compute_agreement() docstring for why this heuristic (over
           the fuzzy-text-match alternative) was chosen.

PROTOTYPE WEIGHTS -- NOT CLINICALLY VALIDATED. w1/w2/w3 below are
placeholder starting points, not derived from any validation study.
They are tuning targets for the Phase 15 evaluation pass, not a
finished calibration -- do not treat the resulting confidence values as
clinically meaningful until that tuning happens.

Weight recalibration (Phase 14, NEW in v5): get_current_weights() reads
the most recent row from storage.py's confidence_weight_history table,
if backend/services/weight_tuner.py has ever run a recalibration,
falling back to DEFAULT_WEIGHTS (the module-level constants above)
otherwise. Both compute_confidence() and compute_confidence_breakdown()
call get_current_weights() -- neither ever references
W1_RAG/W2_WEB/W3_AGREEMENT directly -- so a recalibration takes effect
on the very next call, with no restart needed. THRESHOLD stays a true
fixed constant, never subject to automatic recalibration.
"""

from backend.models.schemas import FusionResult, LiveEvidence, RAGEvidence
from backend.services import storage

# --------------------------------------------------------------------------
# PROTOTYPE WEIGHTS -- NOT CLINICALLY VALIDATED.
# Starting points for Phase 15 evaluation/tuning, not a finished
# calibration. w1 + w2 + w3 == 1.0 so C stays naturally in [0, 1].
# --------------------------------------------------------------------------
W1_RAG = 0.4
W2_WEB = 0.3
W3_AGREEMENT = 0.3

# Same three defaults, keyed for get_current_weights()/weight_tuner.py's
# consumption -- kept in sync with W1_RAG/W2_WEB/W3_AGREEMENT above (which
# stay as the original, individually-named constants for readability
# elsewhere in this module).
DEFAULT_WEIGHTS = {"w1": W1_RAG, "w2": W2_WEB, "w3": W3_AGREEMENT}  # PROTOTYPE DEFAULTS -- NOT CLINICALLY VALIDATED

# Confidence threshold: below this, the pipeline should route the case
# for mandatory human review rather than presenting the result as
# sufficiently supported. Import THRESHOLD from here everywhere it's
# needed (optimizer_agent.py, API layer, etc.) -- never hardcode 0.65.
THRESHOLD = 0.65

# How many of the top (already-reranked, already sorted by score)
# rag.evidence items to average for S_rag. rag_agent.py's pipeline
# already caps rag.evidence at 5 items (RERANK_TOP_K), so this mostly
# matters as a safety bound if a caller passes a larger RAGEvidence.
TOP_K_FOR_RAG_SIGNAL = 5

# Evidence-level -> weight mapping for S_web. Mirrors the same
# prototype-simplification spirit as evidence_agent.py's
# _assess_evidence_level heuristic: a crude tiering, not a validated
# evidence-grading scale. Any evidence_level not in this map (should
# not happen given evidence_agent.py's fixed output, but defensively
# handled) contributes a weight of 0.0.
EVIDENCE_LEVEL_WEIGHTS = {
    "high": 1.0,
    "moderate": 0.6,
    "low": 0.3,
}

# Agreement heuristic: how much each flagged conflict reduces A. At 4+
# conflicts, A floors at 0 -- i.e. a fusion result with many flagged
# disagreements between evidence streams contributes no agreement
# credit at all, rather than going negative.
CONFLICT_PENALTY = 0.25


def _compute_s_rag(rag: RAGEvidence) -> float:
    """Mean of the top-K rag.evidence scores; 0.0 if there is none."""
    if not rag.evidence:
        return 0.0
    top_scores = sorted((item.score for item in rag.evidence), reverse=True)[:TOP_K_FOR_RAG_SIGNAL]
    return sum(top_scores) / len(top_scores)


def _compute_s_web(live: LiveEvidence) -> float:
    """Mean of live.sources evidence_level weights; 0.0 if there is none."""
    if not live.sources:
        return 0.0
    weights = [EVIDENCE_LEVEL_WEIGHTS.get(source.evidence_level, 0.0) for source in live.sources]
    return sum(weights) / len(weights)


def _compute_agreement(fusion: FusionResult) -> float:
    """Agreement heuristic: 1.0 minus a fixed penalty per flagged conflict,
    floored at 0.

    Chosen over the alternative (fuzzy text-matching whether the top
    diagnosis is "supported" by both rag and live evidence) because it
    is directly explainable to a clinician reviewing the score: "reduced
    because N disagreements were flagged between evidence sources" is a
    concrete, auditable reason, whereas a text-similarity match is
    fragile (wording differences look like disagreement) and much
    harder to justify in a UI tooltip. fusion_agent.py's conflicts list
    is already produced by an LLM reading all the evidence, so counting
    it directly reuses that reasoning rather than re-deriving a weaker
    signal from raw text.
    """
    return max(0.0, 1.0 - CONFLICT_PENALTY * len(fusion.conflicts))


def get_current_weights() -> dict:
    """Reads the most recent confidence_weight_history row (see
    backend/services/weight_tuner.py's recalibrate_weights()), or falls
    back to DEFAULT_WEIGHTS if no recalibration has ever happened. This
    is the ONLY place either confidence function reads weights from --
    never reference W1_RAG/W2_WEB/W3_AGREEMENT directly outside this
    function, so a recalibration is picked up immediately.
    """
    row = storage.get_current_weight_row()
    if row is None:
        return dict(DEFAULT_WEIGHTS)
    return {"w1": row["w1"], "w2": row["w2"], "w3": row["w3"]}


def compute_confidence_breakdown(rag: RAGEvidence, live: LiveEvidence, fusion: FusionResult) -> dict:
    """Exposes all three intermediate signals alongside the final score
    -- Phase 14 persists this dict (via storage.save_report's
    confidence_breakdown param) so weight_tuner.py can later recalibrate
    w1/w2/w3 from real accept/override outcomes joined against exactly
    these numbers.
    """
    weights = get_current_weights()
    s_rag = _compute_s_rag(rag)
    s_web = _compute_s_web(live)
    agreement = _compute_agreement(fusion)

    confidence = weights["w1"] * s_rag + weights["w2"] * s_web + weights["w3"] * agreement
    # Safety clamp: mathematically redundant while w1+w2+w3 == 1.0 and
    # each signal is already in [0, 1], but kept so a recalibrated set
    # of weights (or a future formula change) can't silently push C
    # outside [0, 1].
    confidence = max(0.0, min(1.0, confidence))
    # No diagnosis means there is nothing to be confident in. Without this,
    # strong retrieval for a refined query (the optimizer can retrieve well
    # for an empty case) scored 0.79 for a report with zero diagnoses, which
    # cleared THRESHOLD and skipped human review (found by the 31-case run,
    # CASE013).
    if not fusion.diagnoses:
        confidence = 0.0

    return {"s_rag": s_rag, "s_web": s_web, "agreement": agreement, "confidence": confidence}


def compute_confidence(rag: RAGEvidence, live: LiveEvidence, fusion: FusionResult) -> float:
    """Thin wrapper around compute_confidence_breakdown() for callers
    that only need the final float (most of the pipeline) -- see that
    function's docstring for the full breakdown."""
    return compute_confidence_breakdown(rag, live, fusion)["confidence"]
