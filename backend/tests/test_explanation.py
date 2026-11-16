"""The clinician-facing explanation: deterministic, built from the numbers the confidence score uses."""

from backend.models.schemas import ClinicalFindings, EvidenceItem, ImagingFindings, LiveEvidence, LiveEvidenceSource, RAGEvidence
from backend.services.explanation import build_explanation


def _clinical(symptoms=("sudden chest pain",), history=("none",), labs=None, report=()):
    return ClinicalFindings(
        demographics={"age": 24, "sex": "male"}, symptoms=list(symptoms), history=list(history),
        labs=labs or {}, imaging_text_findings=list(report),
    )


def _live(levels=("high", "moderate")):
    return LiveEvidence(
        sources=[
            LiveEvidenceSource(title=f"t{i}", source="PubMed", url="http://x", summary="s", evidence_level=lvl)
            for i, lvl in enumerate(levels)
        ],
        sources_queried=4,
    )


def _rag(n=5, precedents=0):
    items = [EvidenceItem(text="p", source="MedlinePlus: X", score=0.7) for _ in range(n)]
    items += [EvidenceItem(text="past", source="clinician_verified:dr", score=0.8) for _ in range(precedents)]
    return RAGEvidence(evidence=items)


def _build(**overrides):
    args = dict(
        clinical=_clinical(), imaging=None,
        breakdown={"s_rag": 0.7, "s_web": 0.8, "agreement": 1.0, "confidence": 0.77, "input_quality": "ok"},
        diagnoses=[{"name": "Primary spontaneous pneumothorax", "rank": 1, "confidence": 0.7,
                    "supporting_evidence": ["[RAG1] a", "[LIVE1] b"]}],
        conflicts=[], rag=_rag(), live=_live(), confidence=0.77, review_required=False, iteration_count=0,
    )
    args.update(overrides)
    return build_explanation(**args)


def _signal(explanation, key):
    return next(s for s in explanation["signals"] if s["key"] == key)


def test_a_supported_case_says_so_and_needs_no_review_reasons():
    e = _build()
    assert e["review_required"] is False and e["review_reasons"] == []
    assert "Primary spontaneous pneumothorax" in e["headline"] and "2 cited sources" in e["headline"]
    assert "77%" in e["headline"] and "65%" in e["headline"]
    assert [s["key"] for s in e["signals"]] == ["reference", "literature", "agreement"]
    assert _signal(e, "agreement")["level"] == "consistent"


def test_signal_levels_follow_the_scores_without_changing_them():
    weak = _build(breakdown={"s_rag": 0.3, "s_web": 0.3, "agreement": 0.75, "confidence": 0.4, "input_quality": "ok"},
                  conflicts=["a disagreement"], confidence=0.4, review_required=True)
    assert _signal(weak, "reference")["level"] == "limited"
    assert _signal(weak, "literature")["level"] == "limited"
    assert _signal(weak, "agreement")["level"] == "disagreement"
    # the explanation reports the model's numbers as given; it never recomputes the confidence
    assert _signal(weak, "reference")["value"] == 0.3


def test_review_reasons_name_the_weak_signals_and_the_threshold():
    e = _build(breakdown={"s_rag": 0.3, "s_web": 0.0, "agreement": 1.0, "confidence": 0.42, "input_quality": "ok"},
               live=LiveEvidence(sources=[], sources_queried=4), confidence=0.42, review_required=True)
    joined = " ".join(e["review_reasons"])
    assert "42%" in joined and "65%" in joined
    assert "reference library matched only weakly" in joined and "literature support was limited or missing" in joined
    assert _signal(e, "literature")["level"] == "none"


def test_input_caps_are_explained_in_plain_words():
    none = _build(breakdown={"s_rag": 0.7, "s_web": 0.8, "agreement": 1.0, "confidence": 0.5, "input_quality": "no_anchor"},
                  clinical=_clinical(symptoms=()), confidence=0.5, review_required=True)
    assert any("capped at 50%" in r for r in none["review_reasons"])
    assert any("symptoms with onset" in s for s in none["next_steps"])
    vague = _build(breakdown={"s_rag": 0.7, "s_web": 0.8, "agreement": 1.0, "confidence": 0.6, "input_quality": "nonspecific"},
                   clinical=_clinical(symptoms=("mild fatigue",)), confidence=0.6, review_required=True)
    assert any("capped at 60%" in r for r in vague["review_reasons"])


def test_no_diagnosis_is_stated_not_hidden():
    e = _build(diagnoses=[], confidence=0.0, review_required=True,
               breakdown={"s_rag": 0.7, "s_web": 0.8, "agreement": 1.0, "confidence": 0.0, "input_quality": "ok"})
    assert e["headline"].startswith("No diagnosis could be proposed")
    assert any("could not propose a diagnosis" in r for r in e["review_reasons"])


def test_unreachable_literature_is_called_out_as_unchecked():
    live = LiveEvidence(sources=[], sources_queried=4, failed_sources=["a", "b", "c", "d"])
    e = _build(live=live)
    assert _signal(e, "literature")["level"] == "unavailable"
    assert any("not been checked against current publications" in c for c in e["cautions"])
    assert any("Check current guidelines" in s for s in e["next_steps"])


def test_precedents_are_counted_separately_from_library_passages():
    e = _build(rag=_rag(n=5, precedents=2))
    detail = _signal(e, "reference")["detail"]
    assert "5 passages used" in detail and "2 past clinician decisions" in detail


def test_pertinent_negatives_and_present_findings_are_separated():
    clinical = _clinical(symptoms=("sudden chest pain", "absence of fever", "no cough", "Denies trauma"),
                         history=("hypertension",), labs={"hemoglobin": "10.2 g/dL"}, report=("hyperinflation",))
    f = _build(clinical=clinical)["findings"]
    assert f["present"] == ["sudden chest pain"]
    assert f["negatives"] == ["No fever", "No cough", "Denies trauma"]
    assert f["history"] == ["hypertension"] and f["labs"] == ["hemoglobin: 10.2 g/dL"]
    assert f["report_findings"] == ["hyperinflation"]


def test_a_normal_image_read_is_reported_as_a_normal_finding():
    normal = ImagingFindings(modality="chest X-ray", findings=["normal chest X-ray, no acute findings"], confidence=0.55)
    img = _build(imaging=normal)["imaging"]
    assert img["status"] == "normal" and img["normal"] is True
    assert "no acute abnormality" in img["statement"] and "not a radiologist" in img["caveat"]


def test_an_abnormal_image_read_uses_clinical_wording_and_the_screening_caveat():
    abnormal = ImagingFindings(modality="chest X-ray", findings=["pneumothorax", "rib fracture"], confidence=0.68)
    img = _build(imaging=abnormal)["imaging"]
    assert img["status"] == "abnormal" and img["normal"] is False
    assert "suggests a pneumothorax" in img["statement"] and "a rib fracture" in img["statement"]


def test_unusable_image_reads_are_never_presented_as_findings():
    inconclusive = ImagingFindings(modality="MRI brain scan", findings=["edema"], confidence=0.34, inconclusive=True)
    contested = ImagingFindings(modality="CT scan", findings=["sinus abnormality"], confidence=0.89, contested=True,
                                cross_check_label="normal head CT, no acute findings")
    unsupported = ImagingFindings(modality="ultrasound", findings=["no findings model available"], confidence=0.0,
                                  unsupported_modality=True)
    for imaging, status in ((inconclusive, "inconclusive"), (contested, "contested"), (unsupported, "not_assessed")):
        e = _build(imaging=imaging)
        assert e["imaging"]["status"] == status and e["imaging"]["normal"] is False
        assert e["cautions"], status  # surfaced as a caution even when no review is required
    assert "edema" not in _build(imaging=inconclusive)["imaging"]["statement"]


def test_refinement_is_mentioned_and_next_steps_are_not_duplicated():
    e = _build(iteration_count=2, review_required=True, confidence=0.6,
               breakdown={"s_rag": 0.7, "s_web": 0.8, "agreement": 1.0, "confidence": 0.6, "input_quality": "ok"},
               conflicts=["x", "y"])
    assert any("refined 2 times" in c for c in e["cautions"])
    assert len(e["next_steps"]) == len(set(e["next_steps"]))
    assert any("review screen" in s for s in e["next_steps"])


def test_a_missing_breakdown_still_yields_a_usable_explanation():
    e = _build(breakdown=None, confidence=0.0, review_required=True)
    assert e["review_required"] is True and e["signals"]


# ---------------------------------------------------------------------------
# The score ledger: what "80%" is made of
# ---------------------------------------------------------------------------


def _consistent(s_rag, s_web, agreement, weights=None, **overrides):
    w = weights or {"w1": 0.4, "w2": 0.3, "w3": 0.3}
    confidence = w["w1"] * s_rag + w["w2"] * s_web + w["w3"] * agreement
    breakdown = {"s_rag": s_rag, "s_web": s_web, "agreement": agreement, "confidence": confidence,
                 "input_quality": "ok", "weights": w}
    args = dict(breakdown=breakdown, confidence=confidence)
    args.update(overrides)
    return _build(**args)["score"]


def test_the_score_is_split_into_points_earned_and_points_lost_per_source():
    score = _consistent(0.65, 0.8, 1.0)
    assert score["final"] == 80 and score["raw"] == 80 and score["cap"] is None
    parts = {p["key"]: p for p in score["parts"]}
    assert (parts["reference"]["earned"], parts["reference"]["max"], parts["reference"]["lost"]) == (26, 40, 14)
    assert (parts["literature"]["earned"], parts["literature"]["max"], parts["literature"]["lost"]) == (24, 30, 6)
    assert (parts["agreement"]["earned"], parts["agreement"]["max"], parts["agreement"]["lost"]) == (30, 30, 0)
    # the missing 20 points are exactly the points lost across the parts
    assert sum(p["lost"] for p in score["parts"]) == 100 - score["final"]
    # lost points carry a reason and a way to raise them; a full part carries neither
    assert parts["reference"]["why_lost"] and parts["reference"]["how_to_raise"]
    assert parts["agreement"]["why_lost"] is None and parts["agreement"]["how_to_raise"] is None


def test_the_meaning_text_says_what_the_number_is_and_is_not_with_the_real_figures():
    meaning = _consistent(0.65, 0.8, 1.0)["meaning"]
    assert "80 out of 100" in meaning["is"]
    assert "not the chance that the diagnosis is correct" in meaning["is_not"]
    assert "other 20 points" in meaning["remainder"] and "not a 20% chance of error" in meaning["remainder"]
    assert "safety rule" not in meaning["remainder"]  # nothing was capped, so the rule is not mentioned
    assert "65 points" in meaning["review_line"]


def test_parts_always_add_up_to_the_headline_and_to_one_hundred():
    for s_rag, s_web, agreement in [(0.659, 0.6, 0.75), (0.333, 0.3, 0.5), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.71, 0.46, 0.25)]:
        score = _consistent(s_rag, s_web, agreement)
        assert sum(p["earned"] for p in score["parts"]) == score["raw"] == score["final"]
        assert sum(p["max"] for p in score["parts"]) == 100
        assert all(0 <= p["earned"] <= p["max"] for p in score["parts"])


def test_recalibrated_weights_change_each_sources_share_of_the_hundred_points():
    score = _consistent(0.8, 0.8, 0.8, weights={"w1": 0.5, "w2": 0.25, "w3": 0.25})
    assert [p["max"] for p in score["parts"]] == [50, 25, 25]
    assert [p["earned"] for p in score["parts"]] == [40, 20, 20]


def test_a_safety_cap_is_shown_as_points_removed_with_its_reason():
    breakdown = {"s_rag": 0.9, "s_web": 0.9, "agreement": 1.0, "confidence": 0.5, "input_quality": "no_anchor",
                 "weights": {"w1": 0.4, "w2": 0.3, "w3": 0.3}}
    score = _build(breakdown=breakdown, confidence=0.5, review_required=True, clinical=_clinical(symptoms=()))["score"]
    assert score["final"] == 50 and score["raw"] == 93
    assert score["cap"]["points_removed"] == 43 and "Capped at 50" in score["cap"]["reason"]
    assert "removed by a safety rule" in score["meaning"]["remainder"]


def test_no_diagnosis_sets_the_score_to_zero_and_says_why():
    breakdown = {"s_rag": 0.7, "s_web": 0.8, "agreement": 1.0, "confidence": 0.0, "input_quality": "ok",
                 "weights": {"w1": 0.4, "w2": 0.3, "w3": 0.3}}
    score = _build(breakdown=breakdown, confidence=0.0, diagnoses=[], review_required=True)["score"]
    assert score["final"] == 0 and "set to 0" in score["cap"]["reason"]


def test_unreachable_literature_loses_all_its_points_and_says_so():
    live = LiveEvidence(sources=[], sources_queried=4, failed_sources=["a", "b", "c", "d"])
    score = _consistent(0.65, 0.0, 1.0, live=live)
    literature = next(p for p in score["parts"] if p["key"] == "literature")
    assert literature["earned"] == 0 and literature["lost"] == 30 and "could not be reached" in literature["why_lost"]
    assert "Try again" in literature["how_to_raise"]


def test_disagreements_cost_a_quarter_of_the_agreement_points_each():
    score = _consistent(0.65, 0.8, 0.5, conflicts=["x", "y"])
    agreement = next(p for p in score["parts"] if p["key"] == "agreement")
    assert agreement["earned"] == 15 and agreement["lost"] == 15
    assert "2 disagreements were flagged" in agreement["why_lost"] and "25%" in agreement["why_lost"]


def test_a_perfect_score_does_not_claim_certainty():
    score = _consistent(1.0, 1.0, 1.0)
    assert score["final"] == 100 and "not certainty" in score["meaning"]["remainder"]


def test_a_breakdown_without_stored_weights_falls_back_to_the_current_weights():
    breakdown = {"s_rag": 0.65, "s_web": 0.8, "agreement": 1.0, "confidence": 0.8, "input_quality": "ok"}
    score = _build(breakdown=breakdown, confidence=0.8)["score"]
    assert [p["max"] for p in score["parts"]] == [40, 30, 30]
