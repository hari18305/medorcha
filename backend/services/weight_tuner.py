"""Confidence weight recalibration (Phase 14).

Goal: periodically recalibrate w1/w2/w3 (the confidence formula's
weights -- see backend/services/confidence.py) from real clinician
outcomes (accept vs. override), not from any fine-grained expert
annotation. This is deliberately:

  - Auditable: every recalibration is persisted as a new
    confidence_weight_history row (see storage.save_weight_update()),
    never overwriting a previous one -- the full history stays
    inspectable.
  - Gated against small samples: recalibrate_weights() no-ops below
    min_reviews qualifying outcomes, logging why, rather than fitting
    noise from a handful of reviews.
  - Never silent: every no-op and every successful recalibration logs
    what happened and why.
  - Deliberately NOT continuous per-review auto-adjustment: this is an
    explicit, manually-triggered step (see the CLI entry point at the
    bottom of this file: `python -m backend.services.weight_tuner`),
    never run automatically on every POST /review/{case_id}.

THRESHOLD (0.65) is never touched here -- it's a true fixed constant
(see confidence.py), not a recalibration target. Only w1/w2/w3 move.
"""

import logging

from scipy.optimize import minimize

from backend.services import storage

logger = logging.getLogger(__name__)

# Each weight must stay at least this large -- the spec's floor,
# preventing recalibration from ever zeroing out one evidence signal
# entirely (e.g. because a small sample happened to make web evidence
# look useless this batch).
MIN_WEIGHT = 0.1

DEFAULT_MIN_REVIEWS = 10


def collect_review_outcomes() -> list[dict]:
    """For every case with BOTH a stored confidence breakdown AND a
    review decision, returns {s_rag, s_web, agreement, predicted_confidence,
    outcome} rows -- outcome is "accept" (positive, 1.0) or "override"
    (negative, 0.0); "annotate" is neutral and skipped (an annotation
    doesn't tell us whether the confidence score itself was right).

    A case reviewed more than once contributes one row per review
    (storage.get_reports_with_reviews() already does this join) -- each
    row still shares the same predicted_confidence/breakdown, since
    that's fixed at diagnosis time, not at review time.
    """
    rows = storage.get_reports_with_reviews()
    outcomes = []
    for row in rows:
        if row["decision"] == "accept":
            outcome = 1.0
        elif row["decision"] == "override":
            outcome = 0.0
        else:  # "annotate" -- neutral, no accept/override signal
            continue
        if row["s_rag"] is None or row["s_web"] is None or row["agreement"] is None:
            continue  # incomplete breakdown -- skip rather than guess
        outcomes.append(
            {
                "s_rag": row["s_rag"],
                "s_web": row["s_web"],
                "agreement": row["agreement"],
                "predicted_confidence": row["predicted_confidence"],
                "outcome": outcome,
            }
        )
    return outcomes


def _objective(weights: list[float], outcomes: list[dict]) -> float:
    """Squared-error gap between predicted confidence (using candidate
    weights) and the observed accept/override outcome (1.0/0.0) --
    minimizing this fits w1/w2/w3 to what actually correlated with real
    clinician acceptance, not just what the current fixed defaults
    happen to produce.
    """
    w1, w2, w3 = weights
    total = 0.0
    for row in outcomes:
        predicted = w1 * row["s_rag"] + w2 * row["s_web"] + w3 * row["agreement"]
        total += (predicted - row["outcome"]) ** 2
    return total / len(outcomes)


def recalibrate_weights(min_reviews: int = DEFAULT_MIN_REVIEWS) -> dict | None:
    """Refits w1/w2/w3 from real accept/override outcomes, subject to
    w1+w2+w3=1 and each weight >= MIN_WEIGHT. Persists the result via
    storage.save_weight_update() (so confidence.get_current_weights()
    picks it up immediately) and returns
    {"w1", "w2", "w3", "num_reviews_used"}.

    Returns None (a documented no-op, logged, never silent) if fewer
    than min_reviews qualifying outcomes exist.
    """
    outcomes = collect_review_outcomes()
    if len(outcomes) < min_reviews:
        logger.info(
            "recalibrate_weights(): only %d qualifying review outcome(s) available, "
            "need at least %d -- no-op (no recalibration performed).",
            len(outcomes),
            min_reviews,
        )
        return None

    # w1+w2+w3=1 constraint, each weight in [MIN_WEIGHT, 1 - 2*MIN_WEIGHT]
    # so the other two can still each independently satisfy their own
    # floor. Start from the current weights (not always DEFAULT_WEIGHTS)
    # so a second recalibration continues refining rather than always
    # re-fitting from the original defaults.
    from backend.services.confidence import get_current_weights

    current = get_current_weights()
    x0 = [current["w1"], current["w2"], current["w3"]]

    constraints = [{"type": "eq", "fun": lambda w: sum(w) - 1.0}]
    bounds = [(MIN_WEIGHT, 1.0 - 2 * MIN_WEIGHT)] * 3

    result = minimize(
        _objective, x0=x0, args=(outcomes,), method="SLSQP", bounds=bounds, constraints=constraints
    )

    if not result.success:
        logger.warning(
            "recalibrate_weights(): optimizer did not converge (%s) -- no-op, keeping existing weights.",
            result.message,
        )
        return None

    w1, w2, w3 = (float(w) for w in result.x)
    # Numerical cleanup: SLSQP can leave the equality constraint
    # satisfied only to solver tolerance (e.g. sums to 0.999999998) --
    # renormalize so w1+w2+w3 is exactly 1.0, never subject to silent
    # downstream drift in compute_confidence()'s formula.
    total = w1 + w2 + w3
    w1, w2, w3 = w1 / total, w2 / total, w3 / total

    saved = storage.save_weight_update(
        w1, w2, w3,
        num_reviews_used=len(outcomes),
        notes=f"recalibrated from {len(outcomes)} accept/override outcome(s)",
    )
    logger.info(
        "recalibrate_weights(): recalibrated to w1=%.3f w2=%.3f w3=%.3f from %d outcome(s) "
        "(previous: w1=%.3f w2=%.3f w3=%.3f).",
        w1, w2, w3, len(outcomes), current["w1"], current["w2"], current["w3"],
    )
    return {"w1": w1, "w2": w2, "w3": w3, "num_reviews_used": len(outcomes), "id": saved["id"]}


def main() -> None:
    """Explicit, manually-triggered CLI entry point -- NOT run
    automatically on every review (see module docstring)."""
    logging.basicConfig(level=logging.INFO)
    from backend.services.confidence import get_current_weights

    before = get_current_weights()
    result = recalibrate_weights()
    if result is None:
        print("No recalibration performed (see log above for why).")
        return

    print(f"Before: w1={before['w1']:.3f} w2={before['w2']:.3f} w3={before['w3']:.3f}")
    print(
        f"After:  w1={result['w1']:.3f} w2={result['w2']:.3f} w3={result['w3']:.3f} "
        f"(from {result['num_reviews_used']} review outcome(s))"
    )


if __name__ == "__main__":
    main()
