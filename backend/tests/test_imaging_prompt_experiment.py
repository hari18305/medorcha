"""Tests for the held-out prompt experiment: the split, the pre-registered rules, and the lock."""

import json

import pytest
import torch

import backend.tests.imaging_prompt_experiment as pe
from backend.agents.imaging_labels import FINDING_LABELS_BY_MODALITY, MODALITY_LABELS
from backend.tests.imaging_benchmark import load_manifest


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(only="original")


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------


def test_split_assigns_every_group_exactly_once_and_never_splits_a_group(manifest):
    split = pe.make_split(manifest)
    assert set(split["groups"]) == {e["group"] for e in manifest}
    assert set(split["groups"].values()) == {"dev", "test"}
    # an image's half is its group's half, so a series (e.g. 3 slices of one subject) cannot leak across
    halves = {}
    for e in manifest:
        halves.setdefault(e["group"], set()).add(split["groups"][e["group"]])
    assert all(len(h) == 1 for h in halves.values())


def test_split_is_deterministic_and_seed_dependent(manifest):
    assert pe.make_split(manifest) == pe.make_split(manifest)
    assert pe.make_split(manifest, seed=1)["groups"] != pe.make_split(manifest, seed=2)["groups"]


def test_both_halves_contain_every_modality(manifest):
    split = pe.make_split(manifest)
    for half in ("dev", "test"):
        mods = {e["modality"] for e in manifest if split["groups"][e["group"]] == half}
        assert mods == set(FINDING_LABELS_BY_MODALITY), half


def test_committed_split_matches_the_manifest_and_the_seeded_algorithm(manifest):
    committed = json.loads(pe.SPLIT_FILE.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(pe.make_split(manifest)))  # the split on disk is the pre-registered one


def test_stratified_halves_are_roughly_balanced(manifest):
    split = pe.make_split(manifest)
    dev = sum(split["groups"][e["group"]] == "dev" for e in manifest)
    assert 0.4 * len(manifest) <= dev <= 0.65 * len(manifest)


# --------------------------------------------------------------------------
# the candidates
# --------------------------------------------------------------------------


def test_every_finding_label_has_a_description():
    for mod, labels in FINDING_LABELS_BY_MODALITY.items():
        assert set(pe.DESCRIPTIONS[mod]) == set(labels), mod


def test_baseline_candidates_are_the_current_production_wording():
    assert next(iter(pe.FINDING_CANDIDATES)) == "T0_raw_label"
    for mod, labels in FINDING_LABELS_BY_MODALITY.items():
        assert [pe.FINDING_CANDIDATES["T0_raw_label"](mod, lab) for lab in labels] == [[lab] for lab in labels]
    assert [pe.MODALITY_CANDIDATES["M0_raw_label"](lab) for lab in MODALITY_LABELS] == [[lab] for lab in MODALITY_LABELS]


def test_every_candidate_builds_nonempty_prompts_for_every_label():
    for builder in pe.FINDING_CANDIDATES.values():
        for mod, labels in FINDING_LABELS_BY_MODALITY.items():
            for lab in labels:
                prompts = builder(mod, lab)
                assert prompts and all(isinstance(p, str) and p.strip() for p in prompts)
    for builder in pe.MODALITY_CANDIDATES.values():
        for lab in MODALITY_LABELS:
            assert builder(lab)


def test_the_ensemble_contains_the_other_templates():
    mod, lab = "CT scan", "hemorrhage"
    ens = pe.FINDING_CANDIDATES["T4_ensemble"](mod, lab)
    assert set(pe.FINDING_CANDIDATES["T0_raw_label"](mod, lab) + pe.FINDING_CANDIDATES["T1_showing"](mod, lab)
               + pe.FINDING_CANDIDATES["T2_this_is"](mod, lab)) == set(ens)


# --------------------------------------------------------------------------
# scoring maths (fake text encoder)
# --------------------------------------------------------------------------


class _FakeModel:
    """encode_text maps each token string to a fixed one-hot-ish vector so results are predictable."""

    def encode_text(self, tokens):
        return torch.stack([torch.tensor(t, dtype=torch.float32) for t in tokens])


def _tok(prompts):
    table = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0], "ab": [1.0, 1.0, 0.0]}
    return [table[p] for p in prompts]


def test_embed_prompts_averages_then_renormalises():
    out = pe.embed_prompts(_FakeModel(), _tok, [["a"], ["a", "b"]])
    assert out.shape == (2, 3)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(out[1], torch.tensor([1.0, 1.0, 0.0]) / 2**0.5, atol=1e-6)


def test_rank_labels_orders_by_cosine():
    text = pe.embed_prompts(_FakeModel(), _tok, [["a"], ["b"], ["c"]])
    img = pe._unit(torch.tensor([[0.1, 0.9, 0.3]]))
    assert pe.rank_labels(img, text, ["la", "lb", "lc"]) == ["lb", "lc", "la"]


def test_summaries_and_mrr():
    rows = [{"rank": 1, "top1": True, "top3": True}, {"rank": 2, "top1": False, "top3": True}, {"rank": 4, "top1": False, "top3": False}]
    s = pe.summarise_findings(rows)
    assert (s["n"], s["top1"], s["top3"]) == (3, 1, 2)
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0.25) / 3, abs=1e-4)
    assert pe.summarise_modality([{"ok": True}, {"ok": False}]) == {"n": 2, "correct": 1}


# --------------------------------------------------------------------------
# selection and adoption rules (pre-registered)
# --------------------------------------------------------------------------


def test_selection_keeps_the_baseline_unless_strictly_better_on_dev():
    dev = {"T0": {"mrr": 0.60}, "T1": {"mrr": 0.60}, "T2": {"mrr": 0.55}}
    assert pe.choose_finding_candidate(dev) == "T0"          # a tie does not displace the current wording
    dev["T2"]["mrr"] = 0.61
    assert pe.choose_finding_candidate(dev) == "T2"
    assert pe.choose_modality_candidate({"M0": {"correct": 30}, "M1": {"correct": 30}}) == "M0"
    assert pe.choose_modality_candidate({"M0": {"correct": 30}, "M1": {"correct": 31}}) == "M1"


@pytest.mark.parametrize("gains, losses, expected", [(0, 0, 1.0), (5, 0, 0.0625), (6, 0, 0.03125), (3, 3, 1.0), (1, 4, 0.375)])
def test_mcnemar_exact_known_values(gains, losses, expected):
    assert pe.mcnemar_exact(gains, losses) == pytest.approx(expected)


def test_paired_counts_discordant_pairs():
    p = pe.paired([True, True, False, False, True], [True, False, True, True, True])
    assert (p["gains"], p["losses"], p["net"]) == (2, 1, 1)


def _s(top1, mrr, n=25):
    return {"n": n, "top1": top1, "top3": top1, "mrr": mrr}


def test_adoption_needs_the_pre_registered_gain_and_no_mrr_drop():
    pair = {"mcnemar_p": 0.5}
    base = _s(10, 0.50)
    assert pe.decide_findings(base, _s(12, 0.55), pair, "T1", "T0")["adopt"] is True                      # +2 images, MRR up
    assert pe.decide_findings(base, _s(11, 0.60), pair, "T1", "T0")["adopt"] is False                     # +1 image is below the bar
    assert pe.decide_findings(base, _s(13, 0.49), pair, "T1", "T0")["adopt"] is False                     # MRR fell
    assert pe.decide_findings(base, _s(15, 0.70), pair, "T0", "T0")["adopt"] is False                     # nothing was chosen
    assert pe.decide_findings(base, _s(12, 0.55), {"mcnemar_p": 0.01}, "T1", "T0")["significant_at_0.05"] is True


def test_modality_adoption_rule():
    pair = {"mcnemar_p": 0.2}
    b, c = {"n": 25, "correct": 22}, {"n": 25, "correct": 24}
    assert pe.decide_modality(b, c, pair, "M1", "M0")["adopt"] is True
    assert pe.decide_modality(b, {"n": 25, "correct": 23}, pair, "M1", "M0")["adopt"] is False


# --------------------------------------------------------------------------
# the lock: the test half is looked at once
# --------------------------------------------------------------------------


def test_final_refuses_to_run_twice(tmp_path, monkeypatch):
    done = tmp_path / "final.json"
    done.write_text("{}")
    monkeypatch.setattr(pe, "FINAL_JSON", done)
    with pytest.raises(SystemExit, match="already been used"):
        pe.run_final()


def test_final_refuses_to_run_before_dev(tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "FINAL_JSON", tmp_path / "final.json")
    monkeypatch.setattr(pe, "DEV_FILE", tmp_path / "dev.json")
    with pytest.raises(SystemExit, match="dev stage first"):
        pe.run_final()
