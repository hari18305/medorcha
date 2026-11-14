"""Summarise the repeated 31-case runs (scripts/run_repeats.py): run-to-run spread, and cross-check off vs on.

    python scripts/analyze_repeats.py [--root eval_run/repeats]

Reads <root>/<arm><round>/reports/*.json and judge.json. Writes <root>/summary.json and prints the tables.
Small numbers, one model, one judge of the same family: the spread is the point, not an interval.
"""
import argparse
import glob
import json
import re
import statistics as st
from pathlib import Path

THRESHOLD = 0.65
ROOT = Path(__file__).resolve().parents[1]


def toks(name):
    return set(re.findall(r"[a-z]{4,}", name.lower())) - {"with", "disease", "syndrome", "related", "acute", "chronic", "attack", "disorder"}


def same_dx(a, b):
    ta, tb = toks(a), toks(b)
    return bool(ta & tb) and len(ta & tb) / max(1, min(len(ta), len(tb))) >= 0.5


def load(root: Path):
    runs = {}
    for d in sorted(root.glob("*[0-9]")):
        reps = {Path(f).stem: json.load(open(f, encoding="utf-8")) for f in glob.glob(str(d / "reports" / "*.json"))}
        judge = json.load(open(d / "judge.json", encoding="utf-8")) if (d / "judge.json").exists() else None
        runs[d.name] = {"arm": re.sub(r"\d+$", "", d.name), "reports": reps, "judge": judge}
    return runs


def top(r):
    ds = sorted(r["diagnoses"], key=lambda d: d["rank"])
    return ds[0]["name"] if ds else None


def f(x, n=3):
    return None if x is None else round(x, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "eval_run" / "repeats")
    args = ap.parse_args()
    runs = load(args.root)
    arms = {a: [n for n, r in runs.items() if r["arm"] == a] for a in ("off", "on")}
    cases = sorted(set.intersection(*[set(r["reports"]) for r in runs.values() if len(r["reports"]) == 31]))
    out = {"runs": {n: len(r["reports"]) for n, r in runs.items()}, "n_cases": len(cases)}

    # ---- per-run headline numbers
    per_run = {}
    for n, r in runs.items():
        reps = [r["reports"][c] for c in cases if c in r["reports"]]
        row = {"n": len(reps), "mean_conf": f(st.fmean(x["confidence"] for x in reps)),
               "review": sum(x["review_required"] for x in reps),
               "contested": sum(bool((x.get("imaging") or {}).get("contested")) for x in reps),
               "inconclusive": sum(bool((x.get("imaging") or {}).get("inconclusive")) for x in reps)}
        if r["judge"]:
            a = r["judge"]["aggregate"]
            row.update({k: a[k] for k in ("diagnosis_correct", "faithfulness", "context_precision", "cases_with_unsupported_claims", "unsupported_claims_total", "judge_errors")})
        per_run[n] = row
    out["per_run"] = per_run

    # ---- per-arm spread of the headline numbers
    def spread(arm, key):
        vals = [per_run[n][key] for n in arms[arm] if per_run[n].get(key) is not None]
        return {"values": vals, "mean": f(st.fmean(vals)) if vals else None, "sd": f(st.stdev(vals)) if len(vals) > 1 else None,
                "range": [min(vals), max(vals)] if vals else None}
    out["arm_summary"] = {a: {k: spread(a, k) for k in ("mean_conf", "review", "diagnosis_correct", "faithfulness", "context_precision", "cases_with_unsupported_claims", "contested")} for a in arms}

    # ---- per-case spread within each arm
    def per_case(arm):
        rows = {}
        for c in cases:
            reps = [runs[n]["reports"][c] for n in arms[arm]]
            confs = [x["confidence"] for x in reps]
            revs = [x["review_required"] for x in reps]
            tops = [top(x) for x in reps]
            rows[c] = {"conf": [f(v, 2) for v in confs], "sd": f(st.pstdev(confs)), "range": f(max(confs) - min(confs)),
                       "review": revs, "review_stable": len(set(revs)) == 1,
                       "top_stable": all(same_dx(tops[0], t) for t in tops[1:]) if tops[0] else all(t is None for t in tops),
                       "crossed": (min(confs) < THRESHOLD <= max(confs))}
        return rows
    pc = {a: per_case(a) for a in arms}
    out["per_case"] = pc
    for a in arms:
        rows = list(pc[a].values())
        out["arm_summary"][a]["case_level"] = {
            "mean_case_sd": f(st.fmean(r["sd"] for r in rows)), "median_range": f(st.median(r["range"] for r in rows)),
            "max_range": f(max(r["range"] for r in rows)),
            "cases_review_stable": sum(r["review_stable"] for r in rows), "cases_top_dx_stable": sum(r["top_stable"] for r in rows),
            "cases_crossing_threshold": [c for c, r in pc[a].items() if r["crossed"]],
        }

    # ---- cross-check effect, judged against the run-to-run noise
    eff = {}
    for c in cases:
        off = [runs[n]["reports"][c] for n in arms["off"]]
        on = [runs[n]["reports"][c] for n in arms["on"]]
        im_on = [x.get("imaging") or {} for x in on]
        im_off = [x.get("imaging") or {} for x in off]
        eff[c] = {
            "has_image": any(im_on) or any(im_off),
            "contested_in_on_runs": [bool(i.get("contested")) for i in im_on],
            "cross_check_label": [i.get("cross_check_label") for i in im_on],
            "zero_shot_top": [(i.get("findings") or [None])[0] for i in im_off][:1],
            "conf_off": [f(x["confidence"], 2) for x in off], "conf_on": [f(x["confidence"], 2) for x in on],
            "mean_diff_on_minus_off": f(st.fmean(x["confidence"] for x in on) - st.fmean(x["confidence"] for x in off)),
            "review_off": [x["review_required"] for x in off], "review_on": [x["review_required"] for x in on],
            "top_off": [top(x) for x in off], "top_on": [top(x) for x in on],
        }
    out["cross_check_effect"] = eff
    img = {c: e for c, e in eff.items() if e["has_image"]}
    out["image_cases"] = len(img)
    out["image_cases_ever_contested"] = [c for c, e in img.items() if any(e["contested_in_on_runs"])]
    out["non_image_cases_mean_abs_diff"] = f(st.fmean(abs(e["mean_diff_on_minus_off"]) for c, e in eff.items() if not e["has_image"]))
    out["image_cases_mean_abs_diff"] = f(st.fmean(abs(e["mean_diff_on_minus_off"]) for e in img.values())) if img else None

    (args.root / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("runs", "n_cases", "per_run", "arm_summary", "image_cases", "image_cases_ever_contested",
                                           "non_image_cases_mean_abs_diff", "image_cases_mean_abs_diff")}, indent=1, ensure_ascii=False))
    print("\nimage cases: contested in on-runs | conf off -> on | review off/on")
    for c, e in img.items():
        print(f"{c[:40]:40} {str(e['contested_in_on_runs']):22} {e['conf_off']} -> {e['conf_on']}  rev {sum(e['review_off'])}/{len(e['review_off'])} -> {sum(e['review_on'])}/{len(e['review_on'])}  clk={e['cross_check_label'][0]}")


if __name__ == "__main__":
    main()
