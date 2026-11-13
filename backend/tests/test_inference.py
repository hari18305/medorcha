import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.services.inference import on_model_thread


@on_model_thread
def _where():
    return threading.current_thread().name


@on_model_thread
def _outer():
    return _inner()  # a decorated function calling another from the model thread must not deadlock


@on_model_thread
def _inner():
    return threading.current_thread().name


@on_model_thread
def _boom():
    raise ValueError("model failure")


def test_a_decorated_call_runs_on_the_model_thread_whoever_calls_it():
    assert _where().startswith("model")
    with ThreadPoolExecutor(max_workers=4) as pool:  # four brand-new caller threads, as the orchestrator's asyncio.run() creates
        names = list(pool.map(lambda _: _where(), range(8)))
    assert set(names) == {names[0]} and names[0].startswith("model")  # ONE persistent thread, never a new one


def test_decorated_functions_can_call_each_other_without_deadlock():
    assert _outer().startswith("model")


def test_exceptions_propagate_to_the_caller():
    with pytest.raises(ValueError, match="model failure"):
        _boom()


def test_the_real_neural_entry_points_are_routed_through_it():
    from backend.agents import imaging_agent
    from backend.rag import embeddings, reranker

    for fn in (embeddings.embed_text, reranker.rerank, imaging_agent.rank_image, imaging_agent._embed_image_sync):
        assert fn.__wrapped__ is not None  # functools.wraps sets __wrapped__: the decorator is applied
