"""In-process background jobs with live progress, for POST /diagnose?async=true.

A case takes anywhere from about 10 seconds to several minutes (live evidence, LLM calls,
up to three optimizer rounds). Holding one HTTP request open that long gives the browser
nothing to show and breaks on proxies and timeouts, so the API can instead start the run in a
worker thread and let the client poll GET /diagnose/{case_id}/status.

A Job is also the progress SINK the orchestrator reports to (see orchestrator._CaseContext):
real step start/finish events, not a timer-driven animation.

Limits, stated plainly:
  * State is in this process's memory. A restart loses running jobs (the status endpoint then
    reports the case as interrupted, from the database) and is not shared between workers, so run
    one server process (uvicorn without --workers) for this to be coherent.
  * Concurrency is capped (DIAGNOSE_WORKERS, default 2): each run loads heavy models and spends
    LLM quota, so extra submissions wait in a queue and report their queue position.
  * Finished jobs are pruned after an hour or beyond 200, whichever comes first; the report itself
    is in the database and stays retrievable through GET /diagnose/{case_id}.
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from backend.services.logging import log_event

logger = logging.getLogger(__name__)

DEFAULT_ERROR = "The analysis failed on the server. Please try again."
FINISHED_TTL_SECONDS = 3600
MAX_FINISHED_JOBS = 200

QUEUED, RUNNING, COMPLETED, FAILED = "queued", "running", "completed", "failed"
PENDING, DONE, SKIPPED = "pending", "done", "skipped"  # step states (RUNNING / FAILED are shared)


class JobConflict(Exception):
    """A job for this case_id is already queued or running."""


class Job:
    """One analysis run; also the orchestrator's progress sink. Thread-safe."""

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.status = QUEUED
        self.error: str | None = None
        self.report = None  # FinalReport once completed
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self._steps: list[dict] = []
        self._lock = threading.Lock()
        self._step_started: dict[str, float] = {}

    # ---- progress sink (called from the worker thread) -------------------------------

    def _find(self, name: str) -> dict:
        for step in self._steps:
            if step["name"] == name:
                return step
        step = {"name": name, "status": PENDING, "detail": None, "duration_ms": None, "iterations": []}
        self._steps.append(step)  # e.g. the optimizer, which only exists if the run needs it
        return step

    def planned(self, names: list[str]) -> None:
        with self._lock:
            for name in names:
                self._find(name)

    def step_started(self, name: str) -> None:
        with self._lock:
            step = self._find(name)
            step["status"] = RUNNING
            self._step_started[name] = time.perf_counter()

    def step_finished(self, name: str, status: str = DONE, detail: str | None = None, duration_ms: float | None = None) -> None:
        with self._lock:
            step = self._find(name)
            step["status"] = status
            step["detail"] = detail
            if duration_ms is None and name in self._step_started:
                duration_ms = (time.perf_counter() - self._step_started[name]) * 1000
            step["duration_ms"] = round(duration_ms) if duration_ms is not None else None

    def step_skipped(self, name: str, detail: str | None = None) -> None:
        with self._lock:
            step = self._find(name)
            step["status"] = SKIPPED
            step["detail"] = detail

    def iteration(self, event: dict) -> None:
        """An optimizer round event: {"event": "iteration_started"|"iteration_done"|"iteration_failed", ...}."""
        with self._lock:
            rounds = self._find("optimizer")["iterations"]
            i = event["iteration"]
            row = next((r for r in rounds if r["iteration"] == i), None)
            if row is None:
                row = {"iteration": i, "max": event.get("max"), "status": RUNNING, "confidence": None, "strategy": None}
                rounds.append(row)
            if event["event"] == "iteration_done":
                row.update(status=DONE, confidence=event.get("confidence"), strategy=event.get("strategy"))
            elif event["event"] == "iteration_failed":
                row["status"] = FAILED

    # ---- reads ------------------------------------------------------------------------

    def snapshot(self, queue_position: int = 0) -> dict:
        with self._lock:
            steps = [{**s, "iterations": [dict(r) for r in s["iterations"]]} for s in self._steps]
        end = self.finished_at or time.time()
        return {
            "case_id": self.case_id,
            "status": self.status,
            "queue_position": queue_position if self.status == QUEUED else 0,
            "elapsed_s": round(end - self.created_at, 1),
            "error": self.error,
            "steps": steps,
            "report": self.report,
        }


class JobRegistry:
    def __init__(self, workers: int | None = None):
        if workers is None:
            try:
                workers = max(1, int(os.getenv("DIAGNOSE_WORKERS", "2")))
            except ValueError:
                workers = 2
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="diagnose")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, case_id: str, fn) -> Job:
        """Start `fn(job)` in a worker; `fn` returns the FinalReport and reports progress to `job`.
        Raises JobConflict if this case_id already has a queued or running job."""
        with self._lock:
            self._prune_locked()
            existing = self._jobs.get(case_id)
            if existing is not None and existing.status in (QUEUED, RUNNING):
                raise JobConflict(case_id)
            job = Job(case_id)
            self._jobs[case_id] = job
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn) -> None:
        job.status = RUNNING
        job.started_at = time.time()
        try:
            job.report = fn(job)
            job.status = COMPLETED
        except Exception as exc:  # noqa: BLE001 - never leak internals to the client
            logger.exception("diagnose job %s failed", job.case_id)
            log_event("api_error", endpoint="/diagnose?async", case_id=job.case_id, error=str(exc))
            job.error = DEFAULT_ERROR
            job.status = FAILED
        finally:
            job.finished_at = time.time()

    def get(self, case_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(case_id)

    def snapshot(self, case_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(case_id)
            if job is None:
                return None
            ahead = sum(1 for j in self._jobs.values() if j.status == QUEUED and j.created_at < job.created_at)
        return job.snapshot(queue_position=ahead + 1)

    def _prune_locked(self) -> None:
        now = time.time()
        finished = [j for j in self._jobs.values() if j.finished_at is not None]
        for j in finished:
            if now - j.finished_at > FINISHED_TTL_SECONDS:
                del self._jobs[j.case_id]
        finished = sorted((j for j in self._jobs.values() if j.finished_at is not None), key=lambda j: j.finished_at)
        for j in finished[: max(0, len(finished) - MAX_FINISHED_JOBS)]:
            del self._jobs[j.case_id]


registry = JobRegistry()
