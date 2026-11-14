"""Repo-root pytest configuration.

Pins LLM_PROVIDER to "anthropic" for the whole test session, regardless
of whatever a developer's local .env has set for real (non-test) runs.

Why this matters: every test that exercises the orchestrator
(test_orchestrator.py, test_integration.py) calls the real
_get_crewai_llm(), which constructs a real crewai.LLM(...) object --
its own __new__ resolves and imports that provider's "native"
integration module even though the actual network call is mocked
elsewhere. Confirmed live: with LLM_PROVIDER=gemini active, that
import (crewai.llms.providers.gemini.completion -> google.genai)
crashes with a TypeError inside google.genai's own dynamic class
definition (`class AsyncHttpxClient(httpx.AsyncClient)`) -- but *only*
under pytest; the exact same import succeeds in a plain script or the
real running app. This smells like a pytest assertion-rewrite / import
-hook conflict with google-genai's own dynamic class construction, not
a bug in this codebase -- but a test suite that only passes for one
particular value of a developer's local, gitignored .env is a real
test-isolation gap regardless of the underlying cause. Tests should be
deterministic no matter what a developer happens to have configured
for their own local real-usage runs.
"""

import os

import pytest

os.environ["LLM_PROVIDER"] = "anthropic"
# Same reason for the two imaging switches a demo profile sets in .env: the caption tests assume the
# default captions, and with the cross-check on every analyze_image() would make a real LLM call.
# Tests that exercise either setting turn it on with monkeypatch.setenv.
os.environ["IMAGING_FINDING_PROMPTS"] = "showing"
os.environ["IMAGING_CROSS_CHECK"] = "off"


@pytest.fixture(autouse=True)
def _isolate_upload_dir(tmp_path, monkeypatch):
    """POST /diagnose saves uploaded images under routes.UPLOAD_DIR
    (data/images/uploads/). Without this, every test that uploads an image
    wrote its (tiny, fake) file into the REAL uploads folder -- littering the
    dev data a developer's own real runs live in."""
    import backend.api.routes as routes

    monkeypatch.setattr(routes, "UPLOAD_DIR", tmp_path / "uploads")
