"""Does rewording BiomedCLIP's text prompts improve zero-shot imaging labels? A held-out test.

    python -m backend.tests.imaging_prompt_experiment dev     # choose a prompt on the DEV half
    python -m backend.tests.imaging_prompt_experiment final   # judge the choice ONCE on the held-out TEST half

WHY THIS IS STRUCTURED LIKE THIS. The benchmark has 62 images. Trying several prompt
wordings and keeping whichever scores best on all of them would mostly pick the wording
that fits those 62 images. So, FIXED BEFORE ANY RESULT WAS SEEN (this file, in git):

  * the split: groups (a series/patient stays together) are divided into DEV and TEST,
    stratified by modality and finding, with a fixed seed -> data/benchmarks/imaging_real/split.json;
  * the candidates: a small, fixed list of prompt templates and finding descriptions below
    (no tuning loop: five finding-prompt candidates, four modality-prompt candidates);
  * the selection rule: highest DEV mean reciprocal rank (ties go to the current wording);
  * the adoption rule: adopt a change only if, on TEST, it gains at least MIN_TEST_GAIN_IMAGES
    top-1 images over the current wording AND does not lower MRR. Nothing else counts.

The TEST half is evaluated once (`final` refuses to run twice; the result is committed).
`final` reports the paired comparison and an exact McNemar p-value. With ~30 test images
only a large effect can clear that bar, so "not adopted" mostly means "not shown to help",
not "shown not to help".

HONEST LIMITS. (1) I had already seen full-set failure patterns (e.g. hemorrhage called infarct)
when I wrote the descriptions below, so the candidates are informed by them, though not tuned
against any run. (2) The split halves the already-small benchmark, so intervals are very wide.
(3) The descriptions embed clinical knowledge (e.g. "hyperdense blood") written by a non-radiologist.
(4) Labels are the uploader's, not a radiologist's; see RESULTS.md.
"""

import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from backend.agents.imaging_agent import LOGIT_SCALE, _get_model_bundle  # noqa: E402
from backend.agents.imaging_labels import FINDING_LABELS_BY_MODALITY, MODALITY_LABELS  # noqa: E402
from backend.tests.imaging_benchmark import load_manifest  # noqa: E402

BENCH_DIR = Path("data/benchmarks/imaging_real")
SPLIT_FILE = BENCH_DIR / "split.json"
DEV_FILE = BENCH_DIR / "prompt_experiment_dev.json"
FINAL_JSON = BENCH_DIR / "prompt_experiment_final.json"
REPORT_MD = BENCH_DIR / "PROMPT_EXPERIMENT.md"

SEED = 20261113
MIN_TEST_GAIN_IMAGES = 2  # pre-registered adoption bar (top-1 images on TEST), plus MRR not lower

# ---- pre-registered candidates ------------------------------------------------------

# How each modality is spoken of inside a finding prompt.
MODALITY_PHRASE = {"chest X-ray": "chest X-ray", "CT scan": "head CT scan", "MRI brain scan": "brain MRI scan"}

# A short plain-language description for every finding label (what a prompt says instead of the
# bare label). Written by a non-radiologist from general knowledge; never edited after a run.
DESCRIPTIONS = {
    "chest X-ray": {
        "normal chest X-ray, no acute findings": "no acute cardiopulmonary abnormality",
        "pulmonary nodule or mass": "a pulmonary nodule or lung mass",
        "consolidation": "airspace consolidation, pneumonia",
        "pleural effusion": "a pleural effusion",
        "pneumothorax": "a pneumothorax",
        "cardiomegaly": "an enlarged heart, cardiomegaly",
        "pulmonary edema": "pulmonary edema",
        "hyperinflation consistent with emphysema": "hyperinflated lungs, emphysema",
        "atelectasis": "atelectasis, collapsed lung",
        "hilar or mediastinal widening": "hilar or mediastinal widening",
        "rib fracture": "a rib fracture",
        "no significant abnormality": "no significant abnormality",
    },
    "CT scan": {
        "normal head CT, no acute findings": "no acute intracranial abnormality",
        "hypodense lesion": "a hypodense lesion",
        "mass lesion": "an intracranial mass lesion",
        "midline shift": "midline shift",
        "ventricular enlargement": "enlarged ventricles, hydrocephalus",
        "acute infarct": "an acute ischemic infarct, a hypodense region",
        "hemorrhage": "intracranial hemorrhage, hyperdense blood",
        "sinus abnormality": "sinus opacification, sinusitis",
        "calcification": "intracranial calcification",
        "no significant abnormality": "no significant abnormality",
    },
    "MRI brain scan": {
        "normal brain MRI, no acute findings": "no acute intracranial abnormality",
        "white matter hyperintensity": "white matter hyperintensities",
        "mass lesion": "a brain tumor, a mass lesion",
        "midline shift": "midline shift",
        "ventricular enlargement": "enlarged ventricles, hydrocephalus",
        "acute infarct": "an acute infarct with restricted diffusion",
        "chronic infarct": "a chronic infarct",
        "hemorrhage": "intracranial hemorrhage",
        "sinus abnormality": "sinus abnormality",
        "cerebral atrophy": "cerebral atrophy, volume loss",
        "edema": "cerebral edema",
        "vascular malformation": "a vascular malformation",
        "no significant abnormality": "no significant abnormality",
    },
}


def _t0(mod, label):
    return [label]


def _t1(mod, label):
    return [f"{MODALITY_PHRASE[mod]} showing {DESCRIPTIONS[mod][label]}"]


def _t2(mod, label):
    return [f"this is a {MODALITY_PHRASE[mod]} of {DESCRIPTIONS[mod][label]}"]


def _t3(mod, label):
    return [DESCRIPTIONS[mod][label]]


def _t4(mod, label):  # prompt ensemble: raw label + both templates, text embeddings averaged
    return _t0(mod, label) + _t1(mod, label) + _t2(mod, label)


# name -> prompt builder. The FIRST is the current production wording (the baseline).
FINDING_CANDIDATES = {
    "T0_raw_label": _t0,
    "T1_showing": _t1,
    "T2_this_is": _t2,
    "T3_description_only": _t3,
    "T4_ensemble": _t4,
}

# Modality prompts, aligned with MODALITY_LABELS (the output names stay the same).
_MOD_DESC = {
    "MRI brain scan": "an MRI scan of the brain",
    "CT scan": "a CT scan of the head",
    "chest X-ray": "a chest X-ray radiograph",
    "unspecified medical image": "an unspecified medical image",
}
MODALITY_CANDIDATES = {
    "M0_raw_label": lambda lab: [lab],
    "M1_descriptive": lambda lab: [_MOD_DESC[lab]],
    "M2_this_is": lambda lab: [f"this is {_MOD_DESC[lab]}"],
    "M3_ensemble": lambda lab: [lab, _MOD_DESC[lab], f"this is {_MOD_DESC[lab]}"],
}


# ---- split ---------------------------------------------------------------------------


def make_split(entries: list[dict], seed: int = SEED) -> dict:
    """Assign each GROUP to 'dev' or 'test', stratified by (modality, first image's primary label).
    Within a stratum groups are ordered by a seeded hash and alternate dev/test; single-group
    strata alternate globally so test is not starved. A group never spans both halves."""
    groups = {}
    for e in entries:
        groups.setdefault(e["group"], (e["modality"], e["labels"][0]))
    strata = defaultdict(list)
    for g, key in groups.items():
        strata[key].append(g)
    assign, lone_toggle = {}, 0
    for key in sorted(strata):
        ordered = sorted(strata[key], key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
        if len(ordered) == 1:
            assign[ordered[0]] = "dev" if lone_toggle % 2 == 0 else "test"
            lone_toggle += 1
        else:
            for i, g in enumerate(ordered):
                assign[g] = "dev" if i % 2 == 0 else "test"
    return {"seed": seed, "groups": dict(sorted(assign.items()))}


def load_split(entries: list[dict]) -> dict:
    if SPLIT_FILE.exists():
        split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
        if set(split["groups"]) != {e["group"] for e in entries}:
            raise SystemExit("split.json does not match the manifest's groups; the split is fixed, do not regenerate it casually")
        return split
    split = make_split(entries)
    SPLIT_FILE.write_text(json.dumps(split, indent=1), encoding="utf-8")
    return split


# ---- scoring -------------------------------------------------------------------------


def _unit(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True)


def embed_prompts(model, tokenizer, prompts_per_label: list[list[str]]) -> torch.Tensor:
    """One unit text vector per label: the mean of its prompts' unit embeddings, renormalised."""
    rows = []
    with torch.no_grad():
        for prompts in prompts_per_label:
            feats = _unit(model.encode_text(tokenizer(prompts)))
            rows.append(_unit(feats.mean(dim=0, keepdim=True))[0])
    return torch.stack(rows)


def image_features(model, preprocess, path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    with torch.no_grad():
        return _unit(model.encode_image(preprocess(image).unsqueeze(0)))


def rank_labels(img_feat: torch.Tensor, text_feats: torch.Tensor, labels: list[str]) -> list[str]:
    scores = (LOGIT_SCALE * img_feat @ text_feats.T).softmax(dim=-1).squeeze(0).tolist()
    return [lab for lab, _ in sorted(zip(labels, scores), key=lambda p: p[1], reverse=True)]


def finding_metrics_per_image(entries, feats, model, tokenizer, builder) -> list[dict]:
    """Per image: rank of the best accepted label under `builder`, GIVEN the true modality."""
    cache, rows = {}, []
    for e in entries:
        mod = e["modality"]
        if mod not in cache:
            labels = FINDING_LABELS_BY_MODALITY[mod]
            cache[mod] = (labels, embed_prompts(model, tokenizer, [builder(mod, lab) for lab in labels]))
        labels, tf = cache[mod]
        ranked = rank_labels(feats[e["file"]], tf, labels)
        rank = min(ranked.index(a) + 1 for a in e["labels"])
        rows.append({"file": e["file"], "group": e["group"], "rank": rank, "top1": rank == 1, "top3": rank <= 3})
    return rows


def modality_correct_per_image(entries, feats, model, tokenizer, builder) -> list[dict]:
    tf = embed_prompts(model, tokenizer, [builder(lab) for lab in MODALITY_LABELS])
    return [
        {"file": e["file"], "group": e["group"], "ok": rank_labels(feats[e["file"]], tf, MODALITY_LABELS)[0] == e["modality"]}
        for e in entries
    ]


def summarise_findings(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n, "top1": sum(r["top1"] for r in rows), "top3": sum(r["top3"] for r in rows),
            "mrr": round(statistics.fmean(1 / r["rank"] for r in rows), 4) if n else None}


def summarise_modality(rows: list[dict]) -> dict:
    return {"n": len(rows), "correct": sum(r["ok"] for r in rows)}


# ---- decisions -----------------------------------------------------------------------


def choose_finding_candidate(dev: dict) -> str:
    """Highest DEV MRR; ties (and the baseline) keep the current wording."""
    names = list(dev)
    best = names[0]
    for name in names[1:]:
        if dev[name]["mrr"] > dev[best]["mrr"]:
            best = name
    return best


def choose_modality_candidate(dev: dict) -> str:
    names = list(dev)
    best = names[0]
    for name in names[1:]:
        if dev[name]["correct"] > dev[best]["correct"]:
            best = name
    return best


def mcnemar_exact(gains: int, losses: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs (candidate right/baseline wrong = gains)."""
    n = gains + losses
    if n == 0:
        return 1.0
    k = min(gains, losses)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)


def paired(base_flags: list[bool], cand_flags: list[bool]) -> dict:
    gains = sum(c and not b for b, c in zip(base_flags, cand_flags))
    losses = sum(b and not c for b, c in zip(base_flags, cand_flags))
    return {"gains": gains, "losses": losses, "net": gains - losses, "mcnemar_p": round(mcnemar_exact(gains, losses), 4)}


def decide_findings(base: dict, cand: dict, pair: dict, chosen: str, baseline: str) -> dict:
    """The pre-registered adoption rule for finding prompts."""
    if chosen == baseline:
        return {"adopt": False, "reason": "the dev set did not favour any alternative over the current wording"}
    gain = cand["top1"] - base["top1"]
    ok = gain >= MIN_TEST_GAIN_IMAGES and cand["mrr"] >= base["mrr"]
    reason = (f"test top-1 {base['top1']} -> {cand['top1']} (gain {gain}, bar {MIN_TEST_GAIN_IMAGES}); "
              f"MRR {base['mrr']} -> {cand['mrr']}; McNemar p={pair['mcnemar_p']}")
    return {"adopt": ok, "reason": reason, "significant_at_0.05": pair["mcnemar_p"] < 0.05}


def decide_modality(base: dict, cand: dict, pair: dict, chosen: str, baseline: str) -> dict:
    if chosen == baseline:
        return {"adopt": False, "reason": "the dev set did not favour any alternative over the current wording"}
    gain = cand["correct"] - base["correct"]
    ok = gain >= MIN_TEST_GAIN_IMAGES
    return {"adopt": ok, "reason": f"test modality correct {base['correct']} -> {cand['correct']} of {cand['n']} "
                                    f"(gain {gain}, bar {MIN_TEST_GAIN_IMAGES}); McNemar p={pair['mcnemar_p']}",
            "significant_at_0.05": pair["mcnemar_p"] < 0.05}


# ---- stages --------------------------------------------------------------------------


def _setup():
    manifest = load_manifest(only="original")  # the extension set is never used for tuning
    split = load_split(manifest)
    model, preprocess, tokenizer = _get_model_bundle()
    feats = {e["file"]: image_features(model, preprocess, e["file"]) for e in manifest}
    part = {h: [e for e in manifest if split["groups"][e["group"]] == h] for h in ("dev", "test")}
    return manifest, part, model, tokenizer, feats


def run_dev() -> dict:
    manifest, part, model, tok, feats = _setup()
    dev = part["dev"]
    f_dev = {name: summarise_findings(finding_metrics_per_image(dev, feats, model, tok, b)) for name, b in FINDING_CANDIDATES.items()}
    m_dev = {name: summarise_modality(modality_correct_per_image(dev, feats, model, tok, b)) for name, b in MODALITY_CANDIDATES.items()}
    result = {
        "split": {"seed": SEED, "dev_images": len(part["dev"]), "test_images": len(part["test"]),
                  "dev_groups": len({e['group'] for e in part['dev']}), "test_groups": len({e['group'] for e in part['test']})},
        "findings_dev": f_dev, "modality_dev": m_dev,
        "chosen_findings": choose_finding_candidate(f_dev), "chosen_modality": choose_modality_candidate(m_dev),
    }
    DEV_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_final() -> dict:
    if FINAL_JSON.exists():
        raise SystemExit(f"{FINAL_JSON} exists: the TEST half has already been used once. Re-running would defeat the held-out design.")
    if not DEV_FILE.exists():
        raise SystemExit("run the dev stage first (it fixes the choice before the test half is touched)")
    dev = json.loads(DEV_FILE.read_text(encoding="utf-8"))
    manifest, part, model, tok, feats = _setup()
    test = part["test"]
    base_name, chosen = next(iter(FINDING_CANDIDATES)), dev["chosen_findings"]
    base_rows = finding_metrics_per_image(test, feats, model, tok, FINDING_CANDIDATES[base_name])
    cand_rows = finding_metrics_per_image(test, feats, model, tok, FINDING_CANDIDATES[chosen])
    f_base, f_cand = summarise_findings(base_rows), summarise_findings(cand_rows)
    f_pair = paired([r["top1"] for r in base_rows], [r["top1"] for r in cand_rows])

    mbase_name, mchosen = next(iter(MODALITY_CANDIDATES)), dev["chosen_modality"]
    mb = modality_correct_per_image(test, feats, model, tok, MODALITY_CANDIDATES[mbase_name])
    mc = modality_correct_per_image(test, feats, model, tok, MODALITY_CANDIDATES[mchosen])
    m_base, m_cand = summarise_modality(mb), summarise_modality(mc)
    m_pair = paired([r["ok"] for r in mb], [r["ok"] for r in mc])

    result = {
        "findings": {"baseline": base_name, "chosen": chosen, "test_baseline": f_base, "test_chosen": f_cand, "paired_top1": f_pair,
                     "decision": decide_findings(f_base, f_cand, f_pair, chosen, base_name)},
        "modality": {"baseline": mbase_name, "chosen": mchosen, "test_baseline": m_base, "test_chosen": m_cand, "paired": m_pair,
                     "decision": decide_modality(m_base, m_cand, m_pair, mchosen, mbase_name)},
        "dev": dev,
    }
    FINAL_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def render_md(dev: dict, final: dict | None) -> str:
    s = dev["split"]
    L = ["# Prompt-wording experiment (held-out)", "",
         f"Split by group, seed {s['seed']}: DEV {s['dev_images']} images / {s['dev_groups']} groups, TEST {s['test_images']} / {s['test_groups']}. "
         "Candidates, split, selection rule and adoption rule were fixed in `imaging_prompt_experiment.py` before any run. "
         f"A change is adopted only if it gains >= {MIN_TEST_GAIN_IMAGES} top-1 images on TEST and does not lower MRR.", "",
         "## DEV (used to choose)", "", "### Finding prompts (true modality given)", "", "| candidate | top-1 | top-3 | MRR |", "|---|---|---|---|"]
    for k, v in dev["findings_dev"].items():
        mark = " **<- chosen**" if k == dev["chosen_findings"] else ""
        L.append(f"| {k}{mark} | {v['top1']}/{v['n']} | {v['top3']}/{v['n']} | {v['mrr']} |")
    L += ["", "### Modality prompts", "", "| candidate | detected correctly |", "|---|---|"]
    for k, v in dev["modality_dev"].items():
        mark = " **<- chosen**" if k == dev["chosen_modality"] else ""
        L.append(f"| {k}{mark} | {v['correct']}/{v['n']} |")
    if final:
        f, m = final["findings"], final["modality"]
        L += ["", "## TEST (held out, looked at once)", "", "### Finding prompts",
              f"Baseline `{f['baseline']}` vs chosen `{f['chosen']}`.", "",
              "| | top-1 | top-3 | MRR |", "|---|---|---|---|",
              f"| baseline | {f['test_baseline']['top1']}/{f['test_baseline']['n']} | {f['test_baseline']['top3']} | {f['test_baseline']['mrr']} |",
              f"| chosen | {f['test_chosen']['top1']}/{f['test_chosen']['n']} | {f['test_chosen']['top3']} | {f['test_chosen']['mrr']} |", "",
              f"Paired top-1: {f['paired_top1']['gains']} images gained, {f['paired_top1']['losses']} lost, McNemar exact p = {f['paired_top1']['mcnemar_p']}.", "",
              f"**Decision: {'ADOPT' if f['decision']['adopt'] else 'DO NOT ADOPT'}** - {f['decision']['reason']}.", "",
              "### Modality prompts", f"Baseline `{m['baseline']}` vs chosen `{m['chosen']}`.", "",
              f"Test correct: baseline {m['test_baseline']['correct']}/{m['test_baseline']['n']}, chosen {m['test_chosen']['correct']}/{m['test_chosen']['n']}; "
              f"paired {m['paired']['gains']} gained / {m['paired']['losses']} lost, p = {m['paired']['mcnemar_p']}.", "",
              f"**Decision: {'ADOPT' if m['decision']['adopt'] else 'DO NOT ADOPT'}** - {m['decision']['reason']}.", ""]
    L += ["", "## Limits", "",
          "- I had seen full-set failure patterns before writing the descriptions, so candidates are informed by them (not tuned against a run).",
          "- ~30 test images: only a large effect can clear the bar; 'not adopted' mostly means 'not shown to help'.",
          "- Descriptions are a non-radiologist's; labels are the uploader's (see RESULTS.md).", ""]
    return "\n".join(L)


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("dev", "final"):
        print(__doc__)
        return 2
    if sys.argv[1] == "dev":
        res = run_dev()
        print(render_md(res, None))
        print(f"\nchosen finding prompt: {res['chosen_findings']}   chosen modality prompt: {res['chosen_modality']}")
        print("Now commit this result BEFORE running `final`.")
        return 0
    final = run_final()
    md = render_md(final["dev"], final)
    REPORT_MD.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
