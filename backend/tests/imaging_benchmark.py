"""Imaging benchmark: how does BiomedCLIP zero-shot behave on REAL radiology images?

    python -m backend.tests.imaging_benchmark                 # all images, writes results
    python -m backend.tests.imaging_benchmark --limit 6       # quick smoke run, writes nothing

Replaces the old Mode 3 benchmark, which scored the classifier on generated
gradient-and-ellipse pictures with an arbitrary "intended label" (near chance by
construction, so it measured nothing). This one uses the manifest built by
scripts/build_imaging_benchmark.py: real chest X-rays, head CTs and brain MRIs with a
finding label each, taken from open-licence Wikimedia Commons files.

It calls the production classifier (imaging_agent.rank_image / select_findings), not a
reimplementation, and reports for each image:

  * modality detection (no hint, as in the pipeline): right modality?
  * finding ranking GIVEN the true modality (isolates the findings model from modality
    errors): is an accepted label top-1? top-3?
  * what the pipeline would actually report (thresholded findings): does it contain the
    truth, and how many wrong findings does it also pass downstream?
  * normal-vs-abnormal: sensitivity and specificity of the top-1 label.

Every proportion comes with a Wilson 95% interval and is set against a chance baseline
and a majority-class baseline, because with a few images per finding a bare "67%" says
almost nothing. Read RESULTS.md's caveats before quoting any number: small n, labels from
the uploader (not a radiologist), clear teaching examples (optimistic), and possible
overlap with BiomedCLIP's PubMed Central training data.
"""

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agents.imaging_agent import rank_image, select_findings  # noqa: E402
from backend.agents.imaging_labels import FINDING_LABELS_BY_MODALITY  # noqa: E402

BENCH_DIR = Path("data/benchmarks/imaging_real")
MANIFEST = BENCH_DIR / "manifest.json"
RESULTS_JSON = BENCH_DIR / "results.json"
RESULTS_MD = BENCH_DIR / "RESULTS.md"

NORMAL_LABELS = {
    "normal chest X-ray, no acute findings",
    "normal head CT, no acute findings",
    "normal brain MRI, no acute findings",
    "no significant abnormality",
}
ALLOWED_LICENSE_PREFIXES = ("CC0", "CC BY", "Public domain")


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for k successes in n trials; None if n == 0."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def chance_topk(n_accepted: int, n_labels: int, k: int) -> float:
    """P(at least one accepted label lands in a uniformly random top-k of n_labels)."""
    k = min(k, n_labels)
    if n_labels - n_accepted < k:
        return 1.0
    return 1.0 - math.comb(n_labels - n_accepted, k) / math.comb(n_labels, k)


def _prop(k: int, n: int) -> dict:
    ci = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci": list(ci) if ci else None}


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["images"]


def validate_manifest(entries: list[dict]) -> list[str]:
    """Problems that would make the benchmark meaningless; [] when sound."""
    problems = []
    seen = set()
    for e in entries:
        f = e["file"]
        if f in seen:
            problems.append(f"duplicate file {f}")
        seen.add(f)
        if not Path(f).exists():
            problems.append(f"missing file {f}")
        labels = FINDING_LABELS_BY_MODALITY.get(e["modality"])
        if labels is None:
            problems.append(f"{f}: modality {e['modality']!r} has no findings model")
            continue
        if not e["labels"]:
            problems.append(f"{f}: no accepted labels")
        for lab in e["labels"]:
            if lab not in labels:
                problems.append(f"{f}: label {lab!r} is not in the {e['modality']} label set")
        if not str(e.get("license", "")).startswith(ALLOWED_LICENSE_PREFIXES):
            problems.append(f"{f}: licence {e.get('license')!r} is not a recognised open licence")
        if not e.get("group"):
            problems.append(f"{f}: no group")
    return problems


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def score_entry(entry: dict, rank_fn=rank_image) -> dict:
    """Run the classifier on one image twice: with the true modality (isolates the findings
    model) and without a hint (the pipeline's real behaviour, including modality errors)."""
    path = Path(entry["file"])
    accepted = list(entry["labels"])
    hinted = rank_fn(path, entry["modality"])
    detected = rank_fn(path, None)

    ranked = [lab for lab, _ in hinted["finding_scores"]]
    scores = {lab: s for lab, s in hinted["finding_scores"]}
    truth_rank = min(ranked.index(lab) + 1 for lab in accepted)
    returned = select_findings(hinted["finding_scores"])

    det_modality = detected["modality"]
    det_ok = det_modality == entry["modality"]
    e2e_top1_ok = bool(det_ok and detected["finding_scores"] and detected["finding_scores"][0][0] in accepted)

    true_abnormal = accepted[0] not in NORMAL_LABELS
    pred_abnormal = ranked[0] not in NORMAL_LABELS
    return {
        "file": entry["file"], "modality": entry["modality"], "accepted": accepted, "group": entry["group"],
        "possible_pretraining_overlap": bool(entry.get("possible_pretraining_overlap")),
        "has_annotation": bool(entry.get("has_annotation")),
        "detected_modality": det_modality, "modality_ok": det_ok,
        "modality_scores": [[lab, round(s, 4)] for lab, s in (detected["modality_scores"] or [])[:3]],
        "top1": ranked[0], "top1_score": round(scores[ranked[0]], 4), "top3": ranked[:3],
        "top1_ok": ranked[0] in accepted, "top3_ok": any(lab in accepted for lab in ranked[:3]), "truth_rank": truth_rank,
        "returned": returned, "returned_hit": any(lab in accepted for lab in returned),
        "returned_wrong": [lab for lab in returned if lab not in accepted],
        "e2e_top1_ok": e2e_top1_ok,
        "true_abnormal": true_abnormal, "pred_abnormal": pred_abnormal,
    }


def _majority_rate(rows: list[dict], n_labels_by_modality: dict) -> float | None:
    """Accuracy of always predicting the modality's most frequent primary label."""
    if not rows:
        return None
    by_mod = defaultdict(list)
    for r in rows:
        by_mod[r["modality"]].append(r)
    correct = 0
    for rs in by_mod.values():
        top_label = Counter(r["accepted"][0] for r in rs).most_common(1)[0][0]
        correct += sum(top_label in r["accepted"] for r in rs)
    return correct / len(rows)


def summarise(rows: list[dict]) -> dict:
    """All headline metrics for one set of scored rows (any subset)."""
    n = len(rows)
    n_labels = {m: len(v) for m, v in FINDING_LABELS_BY_MODALITY.items()}
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r["top1_ok"])
    group_collapsed = statistics.fmean(statistics.fmean(v) for v in groups.values()) if groups else None
    abn = [r for r in rows if r["true_abnormal"]]
    nor = [r for r in rows if not r["true_abnormal"]]
    return {
        "n_images": n,
        "n_groups": len(groups),
        "modality_detection": _prop(sum(r["modality_ok"] for r in rows), n),
        "top1": _prop(sum(r["top1_ok"] for r in rows), n),
        "top3": _prop(sum(r["top3_ok"] for r in rows), n),
        "top1_group_collapsed": group_collapsed,
        "end_to_end_top1": _prop(sum(r["e2e_top1_ok"] for r in rows), n),
        "returned_contains_truth": _prop(sum(r["returned_hit"] for r in rows), n),
        "returned_has_wrong_finding": _prop(sum(bool(r["returned_wrong"]) for r in rows), n),
        "mean_findings_returned": statistics.fmean(len(r["returned"]) for r in rows) if rows else None,
        "sensitivity_abnormal": _prop(sum(r["pred_abnormal"] for r in abn), len(abn)),
        "specificity_normal": _prop(sum(not r["pred_abnormal"] for r in nor), len(nor)),
        "chance_top1": statistics.fmean(len(r["accepted"]) / n_labels[r["modality"]] for r in rows) if rows else None,
        "chance_top3": statistics.fmean(chance_topk(len(r["accepted"]), n_labels[r["modality"]], 3) for r in rows) if rows else None,
        "majority_top1": _majority_rate(rows, n_labels),
    }


def run(manifest: list[dict], rank_fn=rank_image, progress=None, split: dict | None = None) -> dict:
    """`split` = {group: 'dev'|'test'} (data/benchmarks/imaging_real/split.json). When given, the result
    also reports the DEV and TEST halves separately: the finding prompts were chosen on DEV and judged
    once on TEST (imaging_prompt_experiment.py), so only TEST is an unbiased read of the current prompts."""
    rows = []
    for i, entry in enumerate(manifest, 1):
        rows.append(score_entry(entry, rank_fn))
        if progress:
            progress(i, len(manifest), rows[-1])
    result = {"overall": summarise(rows), "by_modality": {}, "subsets": {}, "by_label": {}, "modality_confusion": {}, "rows": rows}
    for m in FINDING_LABELS_BY_MODALITY:
        sub = [r for r in rows if r["modality"] == m]
        if sub:
            result["by_modality"][m] = summarise(sub)
    result["subsets"] = {
        "excluding possible pretraining overlap": summarise([r for r in rows if not r["possible_pretraining_overlap"]]),
        "excluding annotated films": summarise([r for r in rows if not r["has_annotation"]]),
        "excluding both": summarise([r for r in rows if not r["possible_pretraining_overlap"] and not r["has_annotation"]]),
    }
    if split:
        for half, note in (("test", "TEST half: held out from the prompt choice"), ("dev", "DEV half: used to choose the prompt (optimistic)")):
            sub = [r for r in rows if split.get(r["group"]) == half]
            if sub:
                result["subsets"][note] = summarise(sub)
    by_label = defaultdict(list)
    for r in rows:
        by_label[(r["modality"], r["accepted"][0])].append(r)
    for (m, lab), rs in sorted(by_label.items()):
        wrong = Counter(r["top1"] for r in rs if not r["top1_ok"])
        result["by_label"][f"{m} | {lab}"] = {
            "n": len(rs), "top1_ok": sum(r["top1_ok"] for r in rs), "top3_ok": sum(r["top3_ok"] for r in rs),
            "common_wrong_top1": wrong.most_common(2),
        }
    conf = defaultdict(Counter)
    for r in rows:
        conf[r["modality"]][r["detected_modality"]] += 1
    result["modality_confusion"] = {m: dict(c) for m, c in conf.items()}
    return result


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _fmt(p: dict | None) -> str:
    if not p or p["rate"] is None:
        return "n/a"
    lo, hi = p["ci"]
    return f"{p['rate']:.0%} ({p['k']}/{p['n']}; {lo:.0%}-{hi:.0%})"


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


CAVEATS = [
    "**Small.** A few images per finding, so every per-finding figure has a wide confidence interval (shown). "
    "MRI atrophy and CT midline shift had no usable image and are not covered; MRI white-matter hyperintensity has one.",
    "**Labels are the uploader's, not a radiologist's.** They come from each file's Wikimedia Commons category or caption. "
    "The developer who selected them is not a radiologist, and nobody re-read the films for this project.",
    "**Optimistic.** Commons holds clear, typical teaching examples; real clinical films are harder and messier. "
    "Accuracy here is an upper bound on what to expect, not an estimate of it.",
    "**Possible training overlap.** BiomedCLIP was trained on PubMed Central figure-caption pairs. Images from journal "
    "articles are flagged by a keyword heuristic on their own description, which can miss. See the 'excluding' rows.",
    "**Not independent.** Slices of one patient/series share a `group`; the group-collapsed figure counts each once. "
    "There is exactly ONE normal head-CT subject (three slices), so CT specificity is close to meaningless.",
    "**The finding prompts were tuned on this set.** The wording was chosen on the DEV half and judged once on the "
    "TEST half (PROMPT_EXPERIMENT.md), so the all-image figures are partly in-sample. Quote the TEST-half row "
    "(see 'Sensitivity to the weaknesses') as the unbiased read; it is ~25 images.",
    "**Not a clinical validation.** It characterises one zero-shot model's behaviour on a hand-picked set; it says "
    "nothing about the whole system's diagnostic accuracy.",
]


def render_md(result: dict, manifest_built: str = "") -> str:
    o = result["overall"]
    L = ["# Imaging benchmark results (real images)", "",
         f"BiomedCLIP zero-shot (`imaging_agent.rank_image`), {o['n_images']} real images in {o['n_groups']} groups "
         f"(manifest built {manifest_built or 'n/a'}). Produced by `python -m backend.tests.imaging_benchmark`.", "",
         "## Read this first", ""]
    L += [f"- {c}" for c in CAVEATS]
    L += ["", "## Headline (proportion, count, Wilson 95% interval)", "",
          "| | images | modality detected | top-1 finding | top-3 | end-to-end top-1 | normal-vs-abnormal: sensitivity | specificity |",
          "|---|---|---|---|---|---|---|---|"]
    rows = [("**all**", o)] + [(m, s) for m, s in result["by_modality"].items()]
    for name, s in rows:
        L.append(f"| {name} | {s['n_images']} | {_fmt(s['modality_detection'])} | {_fmt(s['top1'])} | {_fmt(s['top3'])} | "
                 f"{_fmt(s['end_to_end_top1'])} | {_fmt(s['sensitivity_abnormal'])} | {_fmt(s['specificity_normal'])} |")
    L += ["", "Top-1 / top-3 use the TRUE modality's label set (isolating the findings model); end-to-end uses the modality "
          "the pipeline detects, so a wrong modality counts as wrong.", "",
          "## Against baselines", "",
          "| | top-1 | chance top-1 | always-the-commonest-label | top-3 | chance top-3 | group-collapsed top-1 |",
          "|---|---|---|---|---|---|---|"]
    for name, s in rows:
        L.append(f"| {name} | {_fmt(s['top1'])} | {_pct(s['chance_top1'])} | {_pct(s['majority_top1'])} | {_fmt(s['top3'])} | "
                 f"{_pct(s['chance_top3'])} | {_pct(s['top1_group_collapsed'])} |")
    L += ["", "## What the pipeline would actually pass downstream", "",
          "| | reported findings contain the truth | also reports a wrong finding | mean findings reported |", "|---|---|---|---|"]
    for name, s in rows:
        L.append(f"| {name} | {_fmt(s['returned_contains_truth'])} | {_fmt(s['returned_has_wrong_finding'])} | "
                 f"{s['mean_findings_returned']:.2f} |")
    L += ["", "## Sensitivity to the weaknesses above (top-1)", "", "| subset | images | top-1 | modality detected |", "|---|---|---|---|"]
    L.append(f"| all | {o['n_images']} | {_fmt(o['top1'])} | {_fmt(o['modality_detection'])} |")
    for name, s in result["subsets"].items():
        L.append(f"| {name} | {s['n_images']} | {_fmt(s['top1'])} | {_fmt(s['modality_detection'])} |")
    L += ["", "## Per finding", "", "| modality | finding | n | top-1 | top-3 | most common wrong top-1 |", "|---|---|---|---|---|---|"]
    for key, v in result["by_label"].items():
        m, lab = key.split(" | ", 1)
        wrong = "; ".join(f"{w} x{c}" for w, c in v["common_wrong_top1"]) or "-"
        L.append(f"| {m} | {lab} | {v['n']} | {v['top1_ok']}/{v['n']} | {v['top3_ok']}/{v['n']} | {wrong} |")
    L += ["", "## Modality detection (rows = truth, columns = detected)", ""]
    for m, c in result["modality_confusion"].items():
        L.append(f"- **{m}**: " + ", ".join(f"{k} x{v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])))
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="score only the first N images (smoke run; writes nothing)")
    args = ap.parse_args()

    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest = raw["images"]
    problems = validate_manifest(manifest)
    if problems:
        print("Manifest problems:\n  " + "\n  ".join(problems))
        return 2
    if args.limit:
        manifest = manifest[: args.limit]

    def progress(i, n, row):
        print(f"[{i:>2}/{n}] {'OK ' if row['top1_ok'] else 'no '} {row['modality']:<15} true={row['accepted'][0][:28]:<28} "
              f"top1={row['top1'][:28]:<28} modality->{row['detected_modality']}", flush=True)

    split_path = BENCH_DIR / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))["groups"] if split_path.exists() else None
    result = run(manifest, progress=progress, split=split)
    md = render_md(result, raw.get("_provenance", {}).get("built", ""))
    print("\n" + md)
    if not args.limit:
        RESULTS_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        RESULTS_MD.write_text(md, encoding="utf-8")
        print(f"\nwrote {RESULTS_MD} and {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
