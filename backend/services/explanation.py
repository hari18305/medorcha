"""Plain-language explanation of a finished report, written for the reviewing clinician.

The report already carries the ranked differential, its citations and one confidence number. This module
turns what the pipeline actually measured into statements a clinician can read without knowing how the
system works: what the case contained (including normal findings and pertinent negatives), how well the
evidence supports the result, why review is or is not required, and what would help.

Everything here is DETERMINISTIC: it reads the same numbers and flags the confidence score was computed
from, and calls no language model. So it cannot add a claim the pipeline did not make, and it costs no extra
provider quota. (The per-diagnosis "findings for / against" lists that the interface also shows come from the
fusion step itself, not from this module.)

The wording bands below (strong / moderate / limited) are display choices for readability. They are NOT
validated cut-offs and they do not change the confidence score or the review decision.
"""

from backend.agents.imaging_labels import FINDING_DESCRIPTIONS, NORMAL_FINDING_LABELS
from backend.models.schemas import ClinicalFindings, ImagingFindings, LiveEvidence, RAGEvidence
from backend.services.confidence import DEFAULT_WEIGHTS, INSUFFICIENT_INPUT_CAP, NONSPECIFIC_INPUT_CAP, THRESHOLD, get_current_weights

# Display bands for the two evidence signals (see the module docstring: readability only).
REFERENCE_STRONG = 0.75
REFERENCE_MODERATE = 0.55
LITERATURE_STRONG = 0.8
LITERATURE_MODERATE = 0.5

_PRECEDENT_PREFIXES = ("clinician_verified:", "visual_precedent:")

_MODALITY_NAMES = {"MRI brain scan": "Brain MRI", "CT scan": "Head CT", "chest X-ray": "Chest X-ray"}

IMAGING_CAVEAT = "An automated screening aid, not a radiologist's report."

# What each evidence level means in clinical terms (the level comes from study type, not from relevance).
_LEVEL_WORDS = {
    "high": "guideline, systematic review or trial",
    "moderate": "review, cohort or observational study, or registered trial",
    "low": "other report or overview",
}


def _pct(value: float) -> int:
    return round(max(0.0, min(1.0, value)) * 100)


def _is_precedent(item) -> bool:
    return isinstance(item.source, str) and item.source.startswith(_PRECEDENT_PREFIXES)


def _split_symptoms(symptoms: list[str]) -> tuple[list[str], list[str]]:
    """(positive symptoms, pertinent negatives as short statements such as "No fever")."""
    present: list[str] = []
    negatives: list[str] = []
    for raw in symptoms or []:
        text = (raw or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered.startswith("absence of "):
            negatives.append("No " + text[len("absence of "):].strip())
        elif lowered.startswith(("no ", "denies ", "without ")):
            negatives.append(text[0].upper() + text[1:])
        else:
            present.append(text)
    return present, negatives


def _findings_block(clinical: ClinicalFindings | None) -> dict:
    if clinical is None:
        return {"present": [], "negatives": [], "history": [], "labs": [], "report_findings": []}
    present, negatives = _split_symptoms(clinical.symptoms)
    return {
        "present": present,
        "negatives": negatives,
        "history": [h for h in (clinical.history or []) if h],
        "labs": [f"{name}: {value}" for name, value in (clinical.labs or {}).items()],
        "report_findings": [f for f in (clinical.imaging_text_findings or []) if f],
    }


def _imaging_block(imaging: ImagingFindings | None) -> dict | None:
    """What the image check amounts to, in words. None when there was no image."""
    if imaging is None:
        return None
    name = _MODALITY_NAMES.get(imaging.modality, imaging.modality)
    base = {"modality": name, "caveat": IMAGING_CAVEAT}
    if imaging.unsupported_modality:
        return {**base, "status": "not_assessed", "normal": False,
                "statement": f"{name}: not assessed. This image type is not covered by the system, so nothing from the image was used."}
    if imaging.inconclusive:
        return {**base, "status": "inconclusive", "normal": False,
                "statement": f"{name}: no reliable automated read. The image check could not decide, so it was not used."}
    if imaging.contested:
        return {**base, "status": "contested", "normal": False,
                "statement": f"{name}: two readings of the image disagreed, so neither was used."}
    findings = imaging.findings or []
    if not findings:
        return {**base, "status": "no_findings", "normal": False,
                "statement": f"{name}: no findings were recorded."}
    if findings[0] in NORMAL_FINDING_LABELS:
        return {**base, "status": "normal", "normal": True,
                "statement": f"{name}: no acute abnormality identified."}
    descriptions = FINDING_DESCRIPTIONS.get(imaging.modality, {})
    phrases = [descriptions.get(f, f) for f in findings]
    statement = f"{name}: the automated read suggests {phrases[0]}."
    if len(phrases) > 1:
        statement += " Also considered: " + "; ".join(phrases[1:]) + "."
    return {**base, "status": "abnormal", "normal": False, "statement": statement}


def _reference_signal(s_rag: float, rag: RAGEvidence | None) -> dict:
    items = rag.evidence if rag else []
    library = [i for i in items if not _is_precedent(i)]
    precedents = [i for i in items if _is_precedent(i)]
    if s_rag >= REFERENCE_STRONG:
        level, detail = "strong", "The curated reference texts (MedlinePlus, PubMed) match this presentation closely."
    elif s_rag >= REFERENCE_MODERATE:
        level, detail = "moderate", "The curated reference texts partly match this presentation."
    else:
        level, detail = "limited", (
            "The reference library has little that matches this presentation. The condition may lie outside its coverage."
        )
    if library:
        detail += f" {len(library)} passage{'s' if len(library) != 1 else ''} used."
    if precedents:
        detail += f" Also used: {len(precedents)} past clinician decision{'s' if len(precedents) != 1 else ''} on similar cases."
    return {"key": "reference", "label": "Match with the reference library", "level": level,
            "value": round(s_rag, 2), "detail": detail}


def _literature_signal(s_web: float, live: LiveEvidence | None) -> dict:
    sources = live.sources if live else []
    label = "Published literature"
    unavailable = live is None or live.sources_queried == 0 or (live.all_sources_failed and not sources)
    if unavailable:
        return {"key": "literature", "label": label, "level": "unavailable", "value": 0.0,
                "detail": "The literature sources could not be reached, so this result has not been checked against current publications."}
    if not sources:
        return {"key": "literature", "label": label, "level": "none", "value": 0.0,
                "detail": "The search found no matching publications or trials."}
    counts = {"high": 0, "moderate": 0, "low": 0}
    for source in sources:
        counts[source.evidence_level if source.evidence_level in counts else "low"] += 1
    parts = [f"{n} {_LEVEL_WORDS[lvl]}" for lvl, n in counts.items() if n]
    if s_web >= LITERATURE_STRONG:
        level = "strong"
    elif s_web >= LITERATURE_MODERATE:
        level = "moderate"
    else:
        level = "limited"
    detail = f"Found {len(sources)} item{'s' if len(sources) != 1 else ''}. By type: " + "; ".join(parts) + "."
    detail += " Strength reflects the type of study, not how closely it matches this patient."
    return {"key": "literature", "label": label, "level": level, "value": round(s_web, 2), "detail": detail}


def _agreement_signal(agreement: float, conflicts: list[str]) -> dict:
    n = len(conflicts)
    if n == 0:
        return {"key": "agreement", "label": "Agreement between sources", "level": "consistent",
                "value": round(agreement, 2), "detail": "No contradictions were flagged between the evidence sources."}
    return {"key": "agreement", "label": "Agreement between sources", "level": "disagreement",
            "value": round(agreement, 2),
            "detail": f"{n} disagreement{'s were' if n != 1 else ' was'} flagged between sources (listed under Conflicting evidence)."}


# ---------------------------------------------------------------------------------------------------
# The score ledger: what "80%" is made of
# ---------------------------------------------------------------------------------------------------
# The confidence is C = w1*S_rag + w2*S_web + w3*A, with weights summing to 1. Read as points out of 100,
# each signal can earn at most 100*w points, so the part of the score that is MISSING splits exactly into
# how much each signal fell short. That is what the interface shows instead of an unexplained percentage.

_PART_SPECS = (
    ("reference", "w1", "s_rag", "Reference library"),
    ("literature", "w2", "s_web", "Published research"),
    ("agreement", "w3", "agreement", "Agreement between sources"),
)


def _round_to_total(values: list[float], total: int) -> list[int]:
    """Round floats to ints that add up to `total` (largest-remainder), so displayed parts always sum."""
    floors = [int(v) for v in values]
    order = sorted(range(len(values)), key=lambda i: values[i] - floors[i], reverse=True)
    for i in order[: max(0, total - sum(floors))]:
        floors[i] += 1
    return floors


def _weights(breakdown: dict) -> dict:
    stored = breakdown.get("weights")
    if isinstance(stored, dict) and all(k in stored for k in ("w1", "w2", "w3")):
        return {k: float(stored[k]) for k in ("w1", "w2", "w3")}
    try:
        return {k: float(v) for k, v in get_current_weights().items()}
    except Exception:  # noqa: BLE001 - an explanation must never fail on a settings read
        return dict(DEFAULT_WEIGHTS)


def _why_lost(key: str, signal: dict, conflicts: list[str]) -> str | None:
    level = signal["level"]
    if key == "reference":
        return {
            "strong": "The best reference passages matched closely, but not perfectly.",
            "moderate": "The reference texts matched this presentation only partly.",
        }.get(level, "Little in the reference library matches this presentation. It may lie outside the library's coverage.")
    if key == "literature":
        if level == "unavailable":
            return "The literature sources could not be reached, so none of these points could be earned."
        if level == "none":
            return "No matching publications or trials were found."
        return ("Points depend on the type of study: a guideline, systematic review or trial earns the most, "
                "a review or observational study 60% of that, other reports 30%.")
    n = len(conflicts)
    return f"{n} disagreement{'s were' if n != 1 else ' was'} flagged between sources. Each one removes 25% of these points."


def _how_to_raise(key: str, signal: dict) -> str:
    if key == "reference":
        return "Add specific symptoms, examination findings or laboratory values so the library can match more precisely."
    if key == "literature":
        if signal["level"] == "unavailable":
            return "Try again when the literature sources are reachable."
        return "More specific history and findings help the search reach stronger studies."
    return "Check the flagged disagreements against the source documents."


def _score_block(breakdown: dict, signals: list[dict], conflicts: list[str], confidence: float,
                 quality: str, has_diagnoses: bool) -> dict:
    weights = _weights(breakdown)
    by_key = {s["key"]: s for s in signals}
    earned_f, max_f = [], []
    for key, wkey, skey, _label in _PART_SPECS:
        earned_f.append(max(0.0, min(1.0, float(breakdown.get(skey, 0.0)))) * weights[wkey] * 100)
        max_f.append(weights[wkey] * 100)
    raw_total = round(sum(earned_f))
    maxima = _round_to_total(max_f, 100)
    earned = [min(e, m) for e, m in zip(_round_to_total(earned_f, raw_total), maxima)]
    raw_total = sum(earned)
    final = _pct(confidence)

    parts = []
    for (key, _w, _s, label), e, m in zip(_PART_SPECS, earned, maxima):
        signal = by_key[key]
        lost = m - e
        parts.append({
            "key": key, "label": label, "max": m, "earned": e, "lost": lost, "level": signal["level"],
            "detail": signal["detail"],
            "why_lost": _why_lost(key, signal, conflicts) if lost > 0 else None,
            "how_to_raise": _how_to_raise(key, signal) if lost > 0 else None,
        })

    cap = None
    if final < raw_total:
        if not has_diagnoses:
            reason = "No diagnosis could be proposed, so the score is set to 0."
        elif quality == "no_anchor":
            reason = (f"Capped at {_pct(INSUFFICIENT_INPUT_CAP)}: no presenting symptom, examination finding or "
                      "laboratory value could be identified, and evidence alone cannot make up for that.")
        elif quality == "nonspecific":
            reason = (f"Capped at {_pct(NONSPECIFIC_INPUT_CAP)}: only general symptoms (for example tiredness) "
                      "were identified, which fit too many conditions.")
        else:
            reason = "A safety rule lowered the score."
        cap = {"points_removed": raw_total - final, "reason": reason}

    review_line = _pct(THRESHOLD)
    meaning = {
        "is": f"{final} out of 100 evidence points: how well the evidence the system found supports this result.",
        "is_not": ("It is not the chance that the diagnosis is correct. A correct diagnosis can score low when "
                   "the evidence found is thin, and a wrong one can score high when plenty of evidence points "
                   "the wrong way."),
        "remainder": (
            "Nothing was lost, but even a full score is not certainty." if final >= 100 else
            f"The other {100 - final} points were not earned: evidence the system did not find or could not confirm"
            + (", or points removed by a safety rule" if cap else "")
            + f". They are not a {100 - final}% chance of error. The breakdown shows where they were lost."
        ),
        "review_line": f"The review line is {review_line} points. Below it, a clinician must check the result before it is used.",
    }
    return {"final": final, "raw": raw_total, "review_line": review_line, "above_review_line": final >= review_line,
            "parts": parts, "cap": cap, "meaning": meaning}


def build_explanation(
    *,
    clinical: ClinicalFindings | None,
    imaging: ImagingFindings | None,
    breakdown: dict | None,
    diagnoses: list[dict],
    conflicts: list[str],
    rag: RAGEvidence | None,
    live: LiveEvidence | None,
    confidence: float,
    review_required: bool,
    iteration_count: int,
) -> dict:
    """The explanation block of a FinalReport. `conflicts` are the evidence conflicts only (not the
    input-quality notes the orchestrator prepends afterwards)."""
    breakdown = breakdown or {}
    quality = breakdown.get("input_quality") or ("no_anchor" if breakdown.get("input_sufficient") is False else "ok")
    signals = [
        _reference_signal(float(breakdown.get("s_rag", 0.0)), rag),
        _literature_signal(float(breakdown.get("s_web", 0.0)), live),
        _agreement_signal(float(breakdown.get("agreement", 1.0)), conflicts),
    ]
    by_key = {s["key"]: s for s in signals}
    imaging_block = _imaging_block(imaging)
    threshold_pct = _pct(THRESHOLD)

    # ---- why review is (or is not) required ------------------------------------------------------
    review_reasons: list[str] = []
    if review_required:
        if not diagnoses:
            review_reasons.append("The system could not propose a diagnosis from the information and evidence available.")
        if quality == "no_anchor":
            review_reasons.append(
                "No presenting symptom, examination finding or laboratory value could be identified, "
                f"so the score is capped at {_pct(INSUFFICIENT_INPUT_CAP)}%."
            )
        elif quality == "nonspecific":
            review_reasons.append(
                "Only general symptoms (for example tiredness or feeling unwell) were identified, "
                f"so the score is capped at {_pct(NONSPECIFIC_INPUT_CAP)}%."
            )
        if diagnoses and quality == "ok" and confidence < THRESHOLD:
            weak = [
                {"reference": "the reference library matched only weakly",
                 "literature": "the literature support was limited or missing"}[key]
                for key in ("reference", "literature")
                if by_key[key]["level"] in ("limited", "none", "unavailable")
            ]
            why = f" mainly because {' and '.join(weak)}" if weak else ""
            review_reasons.append(
                f"The combined evidence score ({_pct(confidence)}%) is below the {threshold_pct}% review line{why}."
            )
        if conflicts:
            review_reasons.append(
                f"{len(conflicts)} disagreement{'s' if len(conflicts) != 1 else ''} between evidence sources "
                f"{'were' if len(conflicts) != 1 else 'was'} flagged."
            )
        if not review_reasons:
            review_reasons.append("Part of the analysis could not be completed, so the result needs a clinician's check.")

    # ---- cautions that matter whether or not review is required ----------------------------------
    cautions: list[str] = []
    if imaging_block and imaging_block["status"] in ("inconclusive", "contested", "not_assessed"):
        cautions.append(imaging_block["statement"])
    if by_key["literature"]["level"] == "unavailable":
        cautions.append(by_key["literature"]["detail"])
    if iteration_count > 0:
        cautions.append(
            f"The search was refined {iteration_count} time{'s' if iteration_count != 1 else ''} "
            "to try to strengthen the evidence."
        )

    # ---- what would help -------------------------------------------------------------------------
    next_steps: list[str] = []
    findings = _findings_block(clinical)
    if quality in ("no_anchor", "nonspecific"):
        next_steps.append("Add the presenting symptoms with onset and duration, examination findings and any laboratory values.")
    elif review_required and not findings["labs"] and not findings["report_findings"]:
        next_steps.append("If available, add laboratory results or a report: they sharpen the evidence search.")
    if by_key["reference"]["level"] == "limited" and diagnoses:
        next_steps.append("Check the differential against current local guidelines: the reference library may not cover this condition.")
    if by_key["literature"]["level"] in ("none", "unavailable"):
        next_steps.append("Check current guidelines or literature directly: none could be matched automatically.")
    if imaging_block and imaging_block["status"] in ("inconclusive", "contested"):
        next_steps.append("Look at the image directly: the automated read could not be relied on.")
    if conflicts:
        next_steps.append("Check the flagged disagreements against the source documents.")
    if review_required:
        next_steps.append("Confirm, override or annotate the differential on the review screen. Your decision is recorded.")

    # ---- one-sentence headline -------------------------------------------------------------------
    if not diagnoses:
        headline = "No diagnosis could be proposed from the information provided."
    else:
        ordered = sorted(diagnoses, key=lambda d: d.get("rank", 0))
        top = str(ordered[0].get("name", "")).strip() or "the first-ranked diagnosis"
        cited = len(ordered[0].get("supporting_evidence") or [])
        support = (f"supported by {cited} cited source{'s' if cited != 1 else ''}" if cited
                   else "with no cited source")
        if review_required:
            headline = f"Leading consideration: {top}, {support}. This result needs clinician review before it is relied on."
        else:
            headline = (f"Leading consideration: {top}, {support}. The evidence score ({_pct(confidence)}%) is above the "
                        f"{threshold_pct}% review line; clinical judgement still applies.")

    return {
        "headline": headline,
        "review_required": bool(review_required),
        "review_reasons": review_reasons,
        "cautions": cautions,
        "next_steps": list(dict.fromkeys(next_steps)),
        "signals": signals,
        "score": _score_block(breakdown, signals, conflicts, confidence, quality, bool(diagnoses)),
        "findings": findings,
        "imaging": imaging_block,
    }
