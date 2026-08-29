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

os.environ["LLM_PROVIDER"] = "anthropic"
