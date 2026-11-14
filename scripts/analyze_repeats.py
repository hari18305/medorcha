"""Summarise the repeat runs (scripts/run_repeats.py): cross-check off vs on, against the run-to-run noise.

    python scripts/analyze_repeats.py [--root eval_run/repeats]

Runs found under <root>: off<N> (all 31 cases, cross-check off) and on<N> (the 18 image cases, cross-check on), each with
reports/ and judge.json. The noise yardstick is how much a case moves between two runs of the SAME settings: with several
off runs that is the spread among them; the two earlier runs (eval_run/reports_final and reports_rerun, default captions,
cross-check off) are added as a reference when present, flagged because their caption wording differs.
Writes <root>/summary.json and prints the tables. Small numbers, one model, a judge of the same family.
"""
import argparse
import glob
import itertools
import json
import re
import statistics as st
from pathlib import Path

THRESHOLD = 0.65
ROOT = Path(__file__).resolve().parents[1]


def toks(name):
    return set(re.findall(r"[a-z]{4,}", name.lower())) - {"with", "disease", "syndrome", "related", "acute", "chronic", "attack", "disorder"}


def same_dx(a, b):
    if not a or not b:
        return a == b
    ta, tb = toks(a), toks(b)
    return bool(ta & tb) and len(ta & tb) / max(1, min(len(ta), len(tb))) >= 0.5


def top(r):
    ds = sorted(r["diagnoses"], key=lambda d: d["rank"])
    return ds[0]["name"] if ds else None


def f(x, n=3):
    return None if x is None else round(x, n)


def read_reports(d: Path):
    return {Path(p).stem: json.load(open(p, encoding="utf-8")) for p in glob.glob(str(d / "reports" / "*.json"))} if (d / "reports").exists() else {
        Path(p).stem: json.load(open(p, encoding="utf-8")) for p in glob.glob(str(d / "*.json"))}


def judge_means(judge, cases):
    """The judge aggregate restricted to `cases` (so a 31-case run and an 18-case run compare like with like)."""
    if not judge:
        return None
    rs = [r for r in judge["results"] if r["stem"] in cases]
    def mean(name):
        v = [r[name]["score"] for r in rs if r.get(name, {}).get("score") is not None]
        return f(st.fmean(v)) if v else None
    d = [float(r["diagnosis"]["correct"]) for r in rs if "correct" in r.get("diagnosis", {})]
    return {"n": len(rs), "diagnosis_correct": f(st.fmean(d)) if d else None, "faithfulness": mean("faithfulness"),
            "context_precision": mean("context_precision"),
            "cases_with_unsupported": sum(1 for r in rs if r.get("unsupported", {}).get("count", 0) > 0)}


def compare(a: dict, b: dict, cases):
    """Case-level differences between two runs over `cases`."""
    d = [abs(a[c]["confidence"] - b[c]["confidence"]) for c in cases]
    return {
        "n": len(cases), "mean_abs_conf_diff": f(st.fmean(d)), "median": f(st.median(d)), "max": f(max(d)),
        "review_same": sum(a[c]["review_required"] == b[c]["review_required"] for c in cases),
        "review_flipped": [(c, a[c]["review_required"], b[c]["review_required"]) for c in cases if a[c]["review_required"] != b[c]["review_required"]],
        "top_dx_same": sum(same_dx(top(a[c]), top(b[c])) for c in cases),
        "mean_conf_a": f(st.fmean(a[c]["confidence"] for c in cases)), "mean_conf_b": f(st.fmean(b[c]["confidence"] for c in cases)),
        "review_a": sum(a[c]["review_required"] for c in cases), "review_b": sum(b[c]["review_required"] for c in cases),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "eval_run" / "repeats")
    args = ap.parse_args()
    runs = {}
    for d in sorted(args.root.glob("*[0-9]")):
        runs[d.name] = {"arm": re.sub(r"\d+$", "", d.name), "reports": read_reports(d),
                        "judge": json.load(open(d / "judge.json", encoding="utf-8")) if (d / "judge.json").exists() else None}
    off = {n: r for n, r in runs.items() if r["arm"] == "off"}
    on = {n: r for n, r in runs.items() if r["arm"] == "on"}
    image_cases = sorted(set.intersection(*[set(r["reports"]) for r in on.values()])) if on else []
    all_cases = sorted(set.intersection(*[set(r["reports"]) for r in off.values()])) if off else []
    refs = {n: read_reports(ROOT / "eval_run" / n) for n in ("reports_final", "reports_rerun") if (ROOT / "eval_run" / n).exists()}
    out = {"off_runs": list(off), "on_runs": list(on), "n_all": len(all_cases), "n_image": len(image_cases), "image_cases": image_cases}

    # headline per run (own cases) and judge restricted to the image cases
    out["per_run"] = {}
    for n, r in runs.items():
        reps = list(r["reports"].values())
        out["per_run"][n] = {"n": len(reps), "mean_conf": f(st.fmean(x["confidence"] for x in reps)), "review": sum(x["review_required"] for x in reps),
                             "contested": sum(bool((x.get("imaging") or {}).get("contested")) for x in reps),
                             "inconclusive": sum(bool((x.get("imaging") or {}).get("inconclusive")) for x in reps),
                             "judge_all": judge_means(r["judge"], set(r["reports"])), "judge_on_image_cases": judge_means(r["judge"], set(image_cases))}

    # noise yardstick: run-to-run differences among same-setting runs, on the image cases and on all cases
    noise = {}
    same_setting = {**{n: r["reports"] for n, r in off.items()}, **refs}
    for (na, a), (nb, b) in itertools.combinations(same_setting.items(), 2):
        cs = sorted(set(a) & set(b))
        if image_cases:
            noise[f"{na} vs {nb} (image cases)"] = compare(a, b, [c for c in image_cases if c in cs])
        noise[f"{na} vs {nb} (all common)"] = compare(a, b, cs)
    out["noise_between_off_runs"] = noise

    # the effect: on vs off, on the image cases
    effect = {}
    for (nn, r_on), (nf, r_off) in itertools.product(on.items(), off.items()):
        effect[f"{nn} vs {nf}"] = compare(r_off["reports"], r_on["reports"], image_cases)
    out["on_vs_off_image_cases"] = effect

    # per image case: what the cross-check did
    rows = {}
    for c in image_cases:
        im_off = [(r["reports"][c].get("imaging") or {}) for r in off.values()]
        im_on = [(r["reports"][c].get("imaging") or {}) for r in on.values()]
        rows[c] = {"zero_shot": (im_off[0].get("findings") or [None])[0] if im_off else None, "zs_conf": f((im_off[0] or {}).get("confidence"), 2) if im_off else None,
                   "inconclusive": [bool(i.get("inconclusive")) for i in im_on], "contested": [bool(i.get("contested")) for i in im_on],
                   "llm_label": [i.get("cross_check_label") for i in im_on],
                   "conf_off": [f(r["reports"][c]["confidence"], 2) for r in off.values()], "conf_on": [f(r["reports"][c]["confidence"], 2) for r in on.values()],
                   "review_off": [r["reports"][c]["review_required"] for r in off.values()], "review_on": [r["reports"][c]["review_required"] for r in on.values()],
                   "top_off": [top(r["reports"][c]) for r in off.values()], "top_on": [top(r["reports"][c]) for r in on.values()]}
    out["per_image_case"] = rows
    out["cases_contested_any_on_run"] = [c for c, r in rows.items() if any(r["contested"])]

    (args.root / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("off_runs", "on_runs", "n_all", "n_image", "per_run", "noise_between_off_runs", "on_vs_off_image_cases", "cases_contested_any_on_run")}, indent=1, ensure_ascii=False))
    print("\nimage case | zero-shot top (conf) | contested | LLM label | conf off -> on | review off -> on | top dx same")
    for c, r in rows.items():
        same = [same_dx(a, b) for a in r["top_off"] for b in r["top_on"]]
        print(f"{c[:38]:38} {str(r['zero_shot'])[:22]:22} {r['zs_conf']} {str(r['contested']):8} {str([str(x)[:20] for x in r['llm_label']]):26} {r['conf_off']}->{r['conf_on']} rev {r['review_off']}->{r['review_on']} same={same}")


if __name__ == "__main__":
    main()
