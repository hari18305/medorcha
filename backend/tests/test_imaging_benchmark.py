"""Tests for backend/tests/imaging_benchmark.py and the real-image manifest it scores.

The classifier is faked (no model weights); what is tested is the scoring arithmetic and,
separately, that the committed manifest is sound (files exist, labels belong to the label
sets, licences are open, nothing synthetic slipped back in).
"""

import json
import math
from pathlib import Path

import pytest

import backend.tests.imaging_benchmark as ib
from backend.agents.imaging_labels import FINDING_LABELS_BY_MODALITY

CXR, CT, MRI = "chest X-ray", "CT scan", "MRI brain scan"


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def test_wilson_interval_known_values():
    lo, hi = ib.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)


def test_wilson_handles_the_edges_without_leaving_zero_one():
    assert ib.wilson(0, 0) is None
    lo, hi = ib.wilson(0, 5)
    assert lo == 0.0 and 0.4 < hi < 0.5
    lo, hi = ib.wilson(5, 5)
    assert hi == 1.0 and 0.5 < lo < 0.6


def test_wilson_interval_narrows_as_n_grows():
    narrow = ib.wilson(50, 100)
    wide = ib.wilson(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_chance_top1_and_topk():
    # 1 right label among 10, random top-3 -> 3/10
    assert ib.chance_topk(1, 10, 3) == pytest.approx(0.3)
    assert ib.chance_topk(1, 10, 1) == pytest.approx(0.1)
    # 2 acceptable of 4, top-3: only misses if all 3 picked are... impossible -> certain
    assert ib.chance_topk(2, 4, 3) == 1.0
    assert ib.chance_topk(1, 3, 5) == 1.0  # k larger than the label set


# --------------------------------------------------------------------------
# scoring with a fake classifier
# --------------------------------------------------------------------------


def _fake_rank(finding_ranking, detected=CXR, modality_scores=None):
    """A rank_fn that returns a fixed ranking; `detected` is the modality the no-hint call finds."""

    def rank(path, hint):
        modality = hint or detected
        scores = None if hint else (modality_scores or [(detected, 0.9), ("CT scan", 0.1)])
        ranking = finding_ranking if modality == CXR else [(lab, 1 / (i + 2)) for i, lab in enumerate(FINDING_LABELS_BY_MODALITY.get(modality, []))]
        return {"modality": modality, "modality_scores": scores, "finding_scores": ranking or None}

    return rank


def _entry(labels, modality=CXR, group="g", **kw):
    return {"file": "x.png", "modality": modality, "labels": labels, "group": group, **kw}


def test_score_entry_top1_top3_and_truth_rank():
    ranking = [("pleural effusion", 0.5), ("cardiomegaly", 0.3), ("pneumothorax", 0.2)]
    ranking += [(lab, 0.0) for lab in FINDING_LABELS_BY_MODALITY[CXR] if lab not in {r[0] for r in ranking}]
    row = ib.score_entry(_entry(["pneumothorax"]), _fake_rank(ranking))
    assert row["top1"] == "pleural effusion" and not row["top1_ok"]
    assert row["top3_ok"] and row["truth_rank"] == 3


def test_any_accepted_label_counts_and_best_rank_is_used():
    ranking = [("rib fracture", 0.6), ("pneumothorax", 0.4)]
    ranking += [(lab, 0.0) for lab in FINDING_LABELS_BY_MODALITY[CXR] if lab not in {r[0] for r in ranking}]
    row = ib.score_entry(_entry(["pneumothorax", "rib fracture"]), _fake_rank(ranking))
    assert row["top1_ok"] and row["truth_rank"] == 1


def test_wrong_modality_fails_end_to_end_but_not_hinted_findings():
    labels = FINDING_LABELS_BY_MODALITY[CXR]
    ranking = [(labels[1], 0.9)] + [(lab, 0.0) for lab in labels if lab != labels[1]]
    row = ib.score_entry(_entry([labels[1]]), _fake_rank(ranking, detected=CT))
    assert row["top1_ok"]                 # given the true modality the finding is right
    assert not row["modality_ok"]         # but the pipeline would have picked the wrong label set
    assert not row["e2e_top1_ok"]


def test_returned_findings_follow_the_production_threshold_and_flag_wrong_ones():
    labels = FINDING_LABELS_BY_MODALITY[CXR]
    # top 0.6, second 0.5 (>= 0.6*0.6 and > 0.15) -> both returned; the second is not accepted
    ranking = [(labels[1], 0.6), (labels[2], 0.5)] + [(lab, 0.0) for lab in labels if lab not in (labels[1], labels[2])]
    row = ib.score_entry(_entry([labels[1]]), _fake_rank(ranking))
    assert row["returned"] == [labels[1], labels[2]]
    assert row["returned_hit"] and row["returned_wrong"] == [labels[2]]


def test_normal_vs_abnormal_uses_the_primary_label_and_top1():
    normal = "normal chest X-ray, no acute findings"
    labels = FINDING_LABELS_BY_MODALITY[CXR]
    rank_normal = [(normal, 0.9)] + [(lab, 0.0) for lab in labels if lab != normal]
    rank_abn = [("pneumothorax", 0.9)] + [(lab, 0.0) for lab in labels if lab != "pneumothorax"]
    a = ib.score_entry(_entry(["pneumothorax"]), _fake_rank(rank_normal))   # abnormal film called normal
    b = ib.score_entry(_entry([normal]), _fake_rank(rank_abn))               # normal film called abnormal
    assert a["true_abnormal"] and not a["pred_abnormal"]
    assert not b["true_abnormal"] and b["pred_abnormal"]


# --------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------


def _row(ok, group="g", modality=CXR, accepted=("pneumothorax",), abnormal=True, pred_abnormal=True, **kw):
    r = {"modality": modality, "accepted": list(accepted), "group": group, "top1_ok": ok, "top3_ok": ok, "modality_ok": True,
         "e2e_top1_ok": ok, "returned_hit": ok, "returned": ["a"], "returned_wrong": [], "true_abnormal": abnormal,
         "pred_abnormal": pred_abnormal, "possible_pretraining_overlap": False, "has_annotation": False}
    r.update(kw)
    return r


def test_summarise_counts_rates_and_intervals():
    s = ib.summarise([_row(True), _row(True), _row(False), _row(False)])
    assert s["top1"]["k"] == 2 and s["top1"]["n"] == 4 and s["top1"]["rate"] == 0.5
    assert s["top1"]["ci"][0] < 0.5 < s["top1"]["ci"][1]


def test_group_collapsed_accuracy_counts_each_series_once():
    # one 3-slice series all correct + three singletons all wrong:
    # image-level 3/6 = 50%, but group-level (1 + 0 + 0 + 0)/4 = 25%
    rows = [_row(True, "series")] * 3 + [_row(False, f"s{i}") for i in range(3)]
    s = ib.summarise(rows)
    assert s["top1"]["rate"] == 0.5
    assert s["top1_group_collapsed"] == pytest.approx(0.25)
    assert s["n_groups"] == 4


def test_sensitivity_and_specificity_use_the_right_denominators():
    rows = [_row(True, abnormal=True, pred_abnormal=True), _row(True, abnormal=True, pred_abnormal=False),
            _row(True, abnormal=False, pred_abnormal=False), _row(True, abnormal=False, pred_abnormal=False),
            _row(True, abnormal=False, pred_abnormal=True)]
    s = ib.summarise(rows)
    assert (s["sensitivity_abnormal"]["k"], s["sensitivity_abnormal"]["n"]) == (1, 2)
    assert (s["specificity_normal"]["k"], s["specificity_normal"]["n"]) == (2, 3)


def test_majority_baseline_predicts_the_commonest_label_per_modality():
    rows = [_row(True, accepted=("pneumothorax",)), _row(True, accepted=("pneumothorax",)), _row(True, accepted=("cardiomegaly",))]
    assert ib.summarise(rows)["majority_top1"] == pytest.approx(2 / 3)


def test_chance_is_the_mean_over_images_of_their_own_chance():
    n = len(FINDING_LABELS_BY_MODALITY[CXR])
    rows = [_row(True, accepted=("a",)), _row(True, accepted=("a", "b"))]
    assert ib.summarise(rows)["chance_top1"] == pytest.approx((1 / n + 2 / n) / 2)


def test_run_aggregates_subsets_labels_and_confusion():
    labels = FINDING_LABELS_BY_MODALITY[CXR]
    ranking = [("pneumothorax", 0.9)] + [(lab, 0.0) for lab in labels if lab != "pneumothorax"]
    manifest = [_entry(["pneumothorax"], group="a"), _entry(["pneumothorax"], group="b", possible_pretraining_overlap=True)]
    res = ib.run(manifest, rank_fn=_fake_rank(ranking))
    assert res["overall"]["n_images"] == 2
    assert res["subsets"]["excluding possible pretraining overlap"]["n_images"] == 1
    assert res["by_label"][f"{CXR} | pneumothorax"]["top1_ok"] == 2
    assert res["modality_confusion"][CXR] == {CXR: 2}
    assert "Read this first" in ib.render_md(res)  # the caveats always travel with the numbers


def test_run_reports_dev_and_test_halves_when_given_a_split():
    labels = FINDING_LABELS_BY_MODALITY[CXR]
    ranking = [("pneumothorax", 0.9)] + [(lab, 0.0) for lab in labels if lab != "pneumothorax"]
    manifest = [_entry(["pneumothorax"], group="a"), _entry(["cardiomegaly"], group="b"), _entry(["pneumothorax"], group="c")]
    res = ib.run(manifest, rank_fn=_fake_rank(ranking), split={"a": "dev", "b": "test", "c": "dev"})
    assert res["subsets"]["TEST half: held out from the prompt choice"]["n_images"] == 1
    assert res["subsets"]["DEV half: used to choose the prompt (optimistic)"]["n_images"] == 2
    assert res["subsets"]["TEST half: held out from the prompt choice"]["top1"]["k"] == 0   # b's truth is cardiomegaly
    assert "held out" in ib.render_md(res) and "tuned on this set" in ib.render_md(res)


# --------------------------------------------------------------------------
# the committed manifest
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest():
    return ib.load_manifest()


def test_manifest_is_sound(manifest):
    assert ib.validate_manifest(manifest) == []


def test_manifest_has_real_images_for_all_three_modalities_with_a_normal_class(manifest):
    mods = {e["modality"] for e in manifest}
    assert mods == set(FINDING_LABELS_BY_MODALITY)
    for m in mods:
        assert any(e["labels"][0] in ib.NORMAL_LABELS for e in manifest if e["modality"] == m), m


def test_manifest_contains_no_synthetic_images(manifest):
    # the old benchmark's generated gradient images are what this replaces
    for e in manifest:
        assert "synthetic" not in e["file"].lower()
        assert not e["file"].startswith("data/benchmarks/imaging_eval")


def test_overlap_and_annotation_flags_are_real_booleans_and_annotation_is_disclosed(manifest):
    for e in manifest:
        assert isinstance(e["possible_pretraining_overlap"], bool)
    annotated = [e for e in manifest if e.get("has_annotation")]
    assert annotated, "the one known annotated film must stay flagged, not silently kept"


def test_validate_manifest_catches_problems(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    good = {"file": str(img), "modality": CXR, "labels": ["pneumothorax"], "group": "g", "license": "CC0"}
    assert ib.validate_manifest([good]) == []
    bad = [
        {**good, "labels": ["not a real finding"]},
        {**good, "modality": "ultrasound"},
        {**good, "license": "All rights reserved"},
        {**good, "file": str(tmp_path / "missing.png")},
        {**good, "group": ""},
        {**good, "labels": []},
    ]
    for entry in bad:
        assert ib.validate_manifest([entry]), entry
    assert any("duplicate" in p for p in ib.validate_manifest([good, good]))


def test_attribution_file_lists_every_commons_image(manifest):
    text = Path("data/benchmarks/imaging_real/ATTRIBUTION.md").read_text(encoding="utf-8")
    for e in manifest:
        if e["source"] == "wikimedia_commons":
            assert Path(e["file"]).name in text, e["file"]
            assert e["license"] in text
    assert "downscaled" in text  # CC BY / CC BY-SA require stating the change


def test_results_json_matches_the_manifest_size(manifest):
    results = json.loads(ib.RESULTS_JSON.read_text(encoding="utf-8"))
    assert results["overall"]["n_images"] == len(manifest)
    assert math.isclose(
        results["overall"]["top1"]["rate"], sum(r["top1_ok"] for r in results["rows"]) / len(results["rows"])
    )
