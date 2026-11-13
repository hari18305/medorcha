"""Tests for backend/services/jobs.py: the Job progress sink and the worker registry."""

import threading
import time

import pytest

from backend.services import jobs
from backend.services.jobs import DEFAULT_ERROR, Job, JobConflict, JobRegistry


def _wait_for(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# --------------------------------------------------------------------------
# Job as a progress sink
# --------------------------------------------------------------------------


def test_planned_steps_start_pending_in_order():
    job = Job("c1")
    job.planned(["clinical", "imaging", "rag"])
    steps = job.snapshot()["steps"]
    assert [(s["name"], s["status"]) for s in steps] == [("clinical", "pending"), ("imaging", "pending"), ("rag", "pending")]


def test_step_lifecycle_records_status_detail_and_duration():
    job = Job("c1")
    job.planned(["clinical"])
    job.step_started("clinical")
    assert job.snapshot()["steps"][0]["status"] == "running"
    job.step_finished("clinical", "done", "2 symptom(s)", duration_ms=1234.6)
    step = job.snapshot()["steps"][0]
    assert (step["status"], step["detail"], step["duration_ms"]) == ("done", "2 symptom(s)", 1235)


def test_duration_is_measured_when_the_caller_does_not_give_one():
    job = Job("c1")
    job.step_started("rag")
    time.sleep(0.02)
    job.step_finished("rag")
    assert job.snapshot()["steps"][0]["duration_ms"] >= 15


def test_a_step_that_was_not_planned_is_appended_such_as_the_optimizer():
    job = Job("c1")
    job.planned(["fusion"])
    job.step_started("optimizer")
    assert [s["name"] for s in job.snapshot()["steps"]] == ["fusion", "optimizer"]


def test_skipped_step_keeps_the_reason():
    job = Job("c1")
    job.step_skipped("imaging", "no image attached")
    assert job.snapshot()["steps"][0] == {
        "name": "imaging", "status": "skipped", "detail": "no image attached", "duration_ms": None, "iterations": [],
    }


def test_optimizer_rounds_are_tracked_as_they_start_and_finish():
    job = Job("c1")
    job.iteration({"event": "iteration_started", "iteration": 1, "max": 3})
    (opt,) = job.snapshot()["steps"]
    assert opt["name"] == "optimizer" and opt["iterations"][0]["status"] == "running"
    job.iteration({"event": "iteration_done", "iteration": 1, "max": 3, "confidence": 0.47, "strategy": "expand"})
    job.iteration({"event": "iteration_started", "iteration": 2, "max": 3})
    job.iteration({"event": "iteration_failed", "iteration": 2, "max": 3})
    rounds = job.snapshot()["steps"][0]["iterations"]
    assert rounds[0] == {"iteration": 1, "max": 3, "status": "done", "confidence": 0.47, "strategy": "expand"}
    assert rounds[1]["status"] == "failed"


def test_a_snapshot_is_a_copy_not_a_live_view():
    job = Job("c1")
    job.planned(["clinical"])
    snap = job.snapshot()
    job.step_started("clinical")
    assert snap["steps"][0]["status"] == "pending"


def test_concurrent_progress_updates_do_not_lose_or_corrupt_steps():
    job = Job("c1")
    names = [f"s{i}" for i in range(40)]

    def work(name):
        job.step_started(name)
        job.step_finished(name, "done", name, duration_ms=1)

    threads = [threading.Thread(target=work, args=(n,)) for n in names]
    [t.start() for t in threads]
    [t.join() for t in threads]
    steps = job.snapshot()["steps"]
    assert sorted(s["name"] for s in steps) == sorted(names)
    assert all(s["status"] == "done" for s in steps)


# --------------------------------------------------------------------------
# Registry / worker
# --------------------------------------------------------------------------


def test_a_submitted_job_runs_reports_progress_and_completes():
    registry = JobRegistry(workers=1)

    def fn(job):
        job.planned(["clinical"])
        job.step_started("clinical")
        job.step_finished("clinical", "done", "ok", duration_ms=5)
        return "REPORT"

    registry.submit("c1", fn)
    assert _wait_for(lambda: registry.snapshot("c1")["status"] == "completed")
    snap = registry.snapshot("c1")
    assert snap["report"] == "REPORT" and snap["error"] is None
    assert snap["steps"][0]["status"] == "done"


def test_a_failing_job_reports_a_generic_error_and_never_the_exception_text(monkeypatch):
    monkeypatch.setattr(jobs, "log_event", lambda *a, **k: None)
    registry = JobRegistry(workers=1)

    def fn(job):
        raise RuntimeError("secret: 429 quota for billing account 12345")

    registry.submit("c1", fn)
    assert _wait_for(lambda: registry.snapshot("c1")["status"] == "failed")
    snap = registry.snapshot("c1")
    assert snap["error"] == DEFAULT_ERROR and "12345" not in str(snap)
    assert snap["report"] is None


def test_submitting_a_running_case_again_conflicts_but_a_finished_one_can_be_rerun():
    registry = JobRegistry(workers=1)
    release = threading.Event()
    registry.submit("c1", lambda job: release.wait(5) and "A")
    assert _wait_for(lambda: registry.snapshot("c1")["status"] == "running")
    with pytest.raises(JobConflict):
        registry.submit("c1", lambda job: "B")
    release.set()
    assert _wait_for(lambda: registry.snapshot("c1")["status"] == "completed")
    registry.submit("c1", lambda job: "B")
    assert _wait_for(lambda: registry.snapshot("c1")["report"] == "B")


def test_queued_jobs_report_their_position_and_run_in_order():
    registry = JobRegistry(workers=1)
    release = threading.Event()
    order = []
    registry.submit("first", lambda job: (release.wait(5), order.append("first")) and "1")
    assert _wait_for(lambda: registry.snapshot("first")["status"] == "running")
    registry.submit("second", lambda job: order.append("second") or "2")
    time.sleep(0.01)
    registry.submit("third", lambda job: order.append("third") or "3")
    assert registry.snapshot("second")["status"] == "queued"
    assert registry.snapshot("second")["queue_position"] == 1
    assert registry.snapshot("third")["queue_position"] == 2
    assert registry.snapshot("first")["queue_position"] == 0  # running, not queued
    release.set()
    assert _wait_for(lambda: registry.snapshot("third")["status"] == "completed")
    assert order == ["first", "second", "third"]


def test_unknown_case_has_no_snapshot():
    assert JobRegistry(workers=1).snapshot("nope") is None


def test_finished_jobs_expire_after_the_ttl_and_are_capped(monkeypatch):
    registry = JobRegistry(workers=1)
    registry.submit("old", lambda job: "x")
    assert _wait_for(lambda: registry.snapshot("old")["status"] == "completed")
    registry.get("old").finished_at -= jobs.FINISHED_TTL_SECONDS + 1
    registry.submit("new", lambda job: "y")  # submitting prunes
    assert registry.snapshot("old") is None

    monkeypatch.setattr(jobs, "MAX_FINISHED_JOBS", 2)
    for i in range(4):
        registry.submit(f"j{i}", lambda job: "z")
        assert _wait_for(lambda i=i: registry.snapshot(f"j{i}")["status"] == "completed")
    registry.submit("last", lambda job: "z")
    finished = [c for c in ("new", "j0", "j1", "j2", "j3") if registry.snapshot(c) is not None]
    assert len(finished) <= 2


def test_worker_count_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("DIAGNOSE_WORKERS", "3")
    assert JobRegistry()._pool._max_workers == 3
    monkeypatch.setenv("DIAGNOSE_WORKERS", "nonsense")
    assert JobRegistry()._pool._max_workers == 2
    monkeypatch.setenv("DIAGNOSE_WORKERS", "0")
    assert JobRegistry()._pool._max_workers == 1
