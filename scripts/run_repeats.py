"""Repeat the 31-case live run several times, each run judged, for a real run-to-run spread.

    python -m scripts.run_repeats [--rounds 3] [--arms off on] [--root eval_run/repeats]

Per (round, arm) it: copies the isolated store (KB = 67 passages, case memory empty) and a fresh SQLite file, so no
run sees another's saved cases; starts the real backend on them; posts all 31 cases (scripts/run_cases_live.py, one
retry for any case that failed); stops the backend; judges the 31 reports (backend/tests/run_llm_judge.py). Arms differ
ONLY in IMAGING_CROSS_CHECK; the caption wording is pinned to the demonstration profile (label) for both. Rounds
interleave the arms (off1, on1, off2, on2, ...) so a drift in the API over the hours falls on both arms alike.
Resumable: a run with 31 reports and a judge file is skipped. Reports and judge files land in <root>/<arm><round>/.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "eval_run" / "qdrant_live4"
PORT = 8000
N_CASES = 31


def wait_health(proc: subprocess.Popen, timeout: int = 240) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"backend exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(3)
    raise RuntimeError("backend did not become healthy")


def run_cases(out: Path, env: dict) -> int:
    cmd = [sys.executable, "scripts/run_cases_live.py", str(out)]
    subprocess.run(cmd, cwd=ROOT, env=env)
    missing = N_CASES - len(list(out.glob("*.json")))
    if missing:  # one retry of whatever failed
        have = {p.stem for p in out.glob("*.json")}
        cases = sorted((ROOT / "data/test_cases").glob("CASE*.json")) + sorted((ROOT / "data/test_cases/real").glob("REAL*.json"))
        todo = [c.stem for c in cases if c.stem not in have]
        print(f"retrying {len(todo)} case(s): {todo}", flush=True)
        subprocess.run(cmd + ["--only", *todo], cwd=ROOT, env=env)
    return N_CASES - len(list(out.glob("*.json")))


def one_run(name: str, cross_check: str, root: Path) -> None:
    run_dir = root / name
    reports = run_dir / "reports"
    if len(list(reports.glob("*.json"))) == N_CASES and (run_dir / "judge.json").exists():
        print(f"[{name}] already complete, skipping", flush=True)
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    shutil.copytree(STORE, run_dir / "qdrant")
    env = {
        **os.environ,
        "QDRANT_LOCAL_PATH": str(run_dir / "qdrant"),
        "DATABASE_URL": f"sqlite:///{(run_dir / 'run.db').as_posix()}",
        "IMAGING_CROSS_CHECK": cross_check,
        "IMAGING_FINDING_PROMPTS": "label",
        "PYTHONFAULTHANDLER": "1",
    }
    t0 = time.time()
    print(f"[{name}] starting backend (cross-check {cross_check})", flush=True)
    with (run_dir / "server.log").open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app", "--port", str(PORT)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            wait_health(server)
            failed = run_cases(reports, env)
            print(f"[{name}] cases done in {(time.time() - t0) / 60:.1f} min, {failed} missing", flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
    stem = f"repeat_judge_{name}"
    subprocess.run([sys.executable, "-m", "backend.tests.run_llm_judge", "--reports", str(reports), "--out", stem],
                   cwd=ROOT, env=env, stdout=(run_dir / "judge.log").open("w"), stderr=subprocess.STDOUT)
    for ext, dest in ((".json", "judge.json"), (".md", "judge.md"), (".csv", "judge.csv")):
        src = ROOT / "backend" / "tests" / f"{stem}{ext}"
        if src.exists():
            shutil.move(str(src), run_dir / dest)
    print(f"[{name}] finished in {(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=["off", "on"], choices=["off", "on"])
    ap.add_argument("--root", type=Path, default=ROOT / "eval_run" / "repeats")
    args = ap.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for r in range(1, args.rounds + 1):
        for arm in args.arms:
            one_run(f"{arm}{r}", arm, args.root)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
