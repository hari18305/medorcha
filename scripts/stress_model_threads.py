"""Stress test for the backend's native crash. Try to reproduce the backend's native crash (Windows access violation in torch embedding, inside the MedCPT cross-encoder, in a
ThreadPoolExecutor worker) with NO language-model calls: run the real retrieval + rerank (and the BiomedCLIP image read) over and over
the way the orchestrator does, each in a fresh asyncio.run() so each gets a new event loop and a new default executor.

    python scripts/stress_model_threads.py [LABEL] [N]     (default 100 iterations; LABEL is only printed)

Before backend/services/inference.py this crashed within about 35 iterations (access violation); with all model calls on one persistent
thread 300 iterations completed. Needs the local Qdrant store (QDRANT_LOCAL_PATH, default data/qdrant_local) and the models.
"""
import asyncio
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.agents import imaging_agent, rag_agent  # noqa: E402
from backend.models.schemas import ClinicalFindings  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
QUERIES = [
    ["headache", "dizziness"], ["chest pain", "shortness of breath"], ["fatigue", "lightheadedness"],
    ["memory difficulty", "unsteady walking"], ["persistent cough", "low-grade fever"], ["vertigo on rolling over in bed"],
]
IMG = str(ROOT / "data/images/real/real_ct_scan_10.png")


def clinical(i):
    return ClinicalFindings(demographics={}, symptoms=QUERIES[i % len(QUERIES)], history=["hypertension"], labs={}, imaging_text_findings=[])


async def one(i):
    await rag_agent.run_rag(clinical(i), None, query_override=None, image_path=None)
    if i % 4 == 0:
        await imaging_agent.analyze_image(IMG)


t0 = time.time()
for i in range(N):
    asyncio.run(one(i))  # a new loop (and default executor) each time, as orchestrator._run_sync does
    if i % 10 == 0:
        print(f"[{MODE}] {i}/{N} ok  {time.time() - t0:.0f}s", flush=True)
print(f"[{MODE}] finished {N} iterations without a crash in {time.time() - t0:.0f}s", flush=True)
