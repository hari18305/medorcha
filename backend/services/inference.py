"""Run every neural-network call (BioBERT, the MedCPT cross-encoder, BiomedCLIP) on ONE persistent thread.

Why. The backend died, with no Python traceback, in three long runs (Windows access violation, process exit code 139). With
`faulthandler` on, the crash was in a different torch operation each time (an embedding lookup, a GELU) inside a
ThreadPoolExecutor worker, mid-way through the MedCPT cross-encoder or BioBERT. Those workers were brand new every time: the
orchestrator runs each step with `asyncio.run()`, which creates a new event loop and a new default executor, so a retrieval ran on a
different freshly created thread on almost every call (and the async API can run two cases at once on top of that). Running only the
real retrieval and rerank code in that pattern, with no language-model calls at all, crashed within about 35 iterations
(eval_run/crash_repro.py); with all model calls on one persistent thread it did not. That points at torch's per-thread native state
(OpenMP / oneDNN) rather than at any one model, but the exact native cause is not proven; this is a workaround that was measured, not
a root-cause fix. A side effect is that model calls are serialised across concurrent cases, which costs little: they are CPU-bound
and were already competing for the same cores.

Usage: decorate any function that runs a neural network with @on_model_thread. Called from any thread it runs on the model thread
and returns its result; called from the model thread itself it just runs (so decorated functions can call each other).
"""
import functools
import threading
from concurrent.futures import ThreadPoolExecutor

_local = threading.local()


def _mark_model_thread() -> None:
    _local.on_model_thread = True


_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model", initializer=_mark_model_thread)


def on_model_thread(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if getattr(_local, "on_model_thread", False):
            return fn(*args, **kwargs)
        return _EXECUTOR.submit(fn, *args, **kwargs).result()

    return wrapper
