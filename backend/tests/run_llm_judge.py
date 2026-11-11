"""Judge saved MedOrchestrate reports with an LLM (see llm_judge.py for the metrics).

Decoupled from the pipeline on purpose: it reads reports already saved by
``scripts/run_cases_live.py`` (one ``<case file stem>.json`` each), so judging can
be re-run, with a different judge, without re-running the expensive pipeline.

    python -m backend.tests.run_llm_judge --reports reports_dir [--limit N] [--out stem]

Judge choice: set JUDGE_PROVIDER (gemini | mistral | anthropic ...) and optionally
JUDGE_MODEL_ID. Default is whatever LLM_PROVIDER already is -- the same family
that wrote the reports, which is the weakest setup (self-preference); prefer a
different provider and say which one was used (it is printed and written out).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
from pathlib import Path

from backend.tests.llm_judge import judge_report

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "data" / "test_cases"


def load_key() -> dict[str, dict]:
    raw = json.loads((CASES / "answer_key.json").read_text(encoding="utf-8"))
    return {**raw["cases"], **raw["real_cases"]}


def load_case(stem: str) -> dict:
    for folder in (CASES, CASES / "real"):
        p = folder / f"{stem}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no case file for {stem}")


def make_ask():
    """Builds the judge callable from the project's own provider layer, so the
    free-tier throttle and provider switching apply to judge calls too."""
    from backend.services import llm_client

    provider = os.getenv("JUDGE_PROVIDER")
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    if os.getenv("JUDGE_MODEL_ID"):
        os.environ[{"gemini": "GEMINI_MODEL_ID", "mistral": "MISTRAL_MODEL_ID"}.get(
            os.environ.get("LLM_PROVIDER", "").lower(), "ANTHROPIC_CLAUDE_MODEL_ID")] = os.environ["JUDGE_MODEL_ID"]
    llm_client.reset_client_cache()
    client, model = llm_client.get_llm_client(), llm_client.get_model_name()

    async def ask(prompt: str) -> str:
        from backend.tests.llm_judge import SYSTEM

        resp = await client.messages.create(
            model=model, max_tokens=1500, system=SYSTEM, messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text

    return ask, f"{os.environ.get('LLM_PROVIDER', 'anthropic')}:{model}"


def _mean(xs: list[float]) -> float | None:
    return round(statistics.fmean(xs), 3) if xs else None


def aggregate(results: list[dict]) -> dict:
    """Corpus-level numbers. Errors are COUNTED, not silently dropped, so a
    flaky judge cannot quietly inflate a score by skipping its failures."""
    out: dict = {"n_cases": len(results)}
    diag = [r["diagnosis"] for r in results if "correct" in r.get("diagnosis", {})]
    out["diagnosis_correct"] = _mean([float(d["correct"]) for d in diag])
    out["diagnosis_n"] = len(diag)
    for name in ("faithfulness", "context_precision"):
        scored = [r[name]["score"] for r in results if r.get(name, {}).get("score") is not None]
        out[name] = _mean(scored)
        out[f"{name}_n"] = len(scored)
    unsup = [r["unsupported"]["count"] for r in results if "count" in r.get("unsupported", {})]
    out["cases_with_unsupported_claims"] = sum(1 for c in unsup if c > 0)
    out["unsupported_claims_total"] = sum(unsup)
    out["unsupported_n"] = len(unsup)
    out["judge_errors"] = sum(1 for r in results for m in ("diagnosis", "faithfulness", "context_precision", "unsupported") if "error" in r.get(m, {}))
    return out


def render_md(results: list[dict], agg: dict, judge: str) -> str:
    lines = [
        "# LLM-judged evaluation",
        "",
        f"Judge: `{judge}`. Answer key: developer-drafted, NOT clinician-verified. "
        "RAGAS-style metrics implemented natively (not the ragas library). "
        f"n = {agg['n_cases']} cases: a consistency check, not clinical validation.",
        "",
        "| metric | value | scored |",
        "|---|---|---|",
        f"| diagnosis correct (top-3 vs answer key) | {agg['diagnosis_correct']} | {agg['diagnosis_n']} |",
        f"| faithfulness (cited evidence supports diagnosis) | {agg['faithfulness']} | {agg['faithfulness_n']} |",
        f"| context precision (RAGAS average precision) | {agg['context_precision']} | {agg['context_precision_n']} |",
        f"| cases with unsupported claims | {agg['cases_with_unsupported_claims']} ({agg['unsupported_claims_total']} claims) | {agg['unsupported_n']} |",
        f"| judge errors (metrics that failed to score) | {agg['judge_errors']} | |",
        "",
        "## Per case",
        "",
        "| case | correct | matched | faithfulness | ctx precision | unsupported claims |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        d, f, c, u = r.get("diagnosis", {}), r.get("faithfulness", {}), r.get("context_precision", {}), r.get("unsupported", {})
        fmt = lambda x: "err" if x is None else (f"{x:.2f}" if isinstance(x, float) else x)  # noqa: E731
        lines.append(
            f"| {r['stem']} | {d.get('correct', 'err')} ({d.get('method', '')}) | {d.get('matched') or ''} | "
            f"{fmt(f.get('score', 'err') if 'error' not in f else None)} | {fmt(c.get('score', 'err') if 'error' not in c else None)} | "
            f"{'; '.join(u.get('claims', [])) or ('err' if 'error' in u else '-')} |"
        )
    return "\n".join(lines) + "\n"


async def run(reports_dir: Path, limit: int | None, ask) -> list[dict]:
    key = load_key()
    stems = sorted(p.stem for p in reports_dir.glob("*.json") if p.stem in key)
    if limit:
        stems = stems[:limit]
    results = []
    for stem in stems:
        report = json.loads((reports_dir / f"{stem}.json").read_text(encoding="utf-8"))
        res = await judge_report(ask, report, load_case(stem), key[stem])
        res["stem"] = stem
        results.append(res)
        print(f"judged {stem}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="llm_judge_eval", help="output file stem, written next to this script")
    args = ap.parse_args()

    ask, judge = make_ask()
    results = asyncio.run(run(args.reports, args.limit, ask))
    agg = aggregate(results)
    out = Path(__file__).parent / args.out
    out.with_suffix(".md").write_text(render_md(results, agg, judge), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"judge": judge, "aggregate": agg, "results": results}, indent=1), encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "correct", "faithfulness", "context_precision", "unsupported_claims"])
        for r in results:
            w.writerow([r["stem"], r.get("diagnosis", {}).get("correct"), r.get("faithfulness", {}).get("score"),
                        r.get("context_precision", {}).get("score"), r.get("unsupported", {}).get("count")])
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
