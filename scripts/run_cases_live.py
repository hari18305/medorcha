"""POST test-case files to a RUNNING backend and save each FinalReport as
``<out_dir>/<case file stem>.json`` (the layout run_llm_judge.py reads).

    python scripts/run_cases_live.py OUT_DIR [--only REAL001 CASE014 ...] [--url http://127.0.0.1:8000]

Start the backend first. Each case is a real pipeline run (several LLM calls),
so on a quota-limited key use --only. A case that fails is reported and skipped;
the rest continue. Saved reports go to the real database too, like any /diagnose.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "test_cases"


def post_case(case: dict, url: str) -> requests.Response:
    p = case["patient"]
    data = {"age": p["age"], "sex": p["sex"]}
    for k in ("clinical_text", "medical_report"):
        if case.get(k):
            data[k] = case[k]
    if case.get("history"):
        data["history"] = json.dumps(case["history"])
    if case.get("labs"):
        data["labs"] = json.dumps(case["labs"])
    files = None
    if case.get("image_path"):
        ip = ROOT / case["image_path"]
        files = {"image": (ip.name, ip.read_bytes())}
    return requests.post(f"{url}/diagnose", data=data, files=files, timeout=900)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--only", nargs="*", help="case-file stem prefixes, e.g. REAL001 CASE014")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(CASES.glob("CASE*.json")) + sorted((CASES / "real").glob("REAL*.json"))
    if args.only:
        files = [f for f in files if any(f.stem.startswith(o) for o in args.only)]
    failed = 0
    for f in files:
        t = time.time()
        try:
            r = post_case(json.loads(f.read_text(encoding="utf-8")), args.url)
        except requests.RequestException as exc:
            print(f"{f.stem}: request failed ({exc})"); failed += 1; continue
        if r.status_code != 200:
            print(f"{f.stem}: HTTP {r.status_code} {r.text[:160]}"); failed += 1; continue
        (args.out_dir / f"{f.stem}.json").write_text(json.dumps(r.json(), indent=1), encoding="utf-8")
        print(f"{f.stem}: ok in {time.time() - t:.0f}s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
