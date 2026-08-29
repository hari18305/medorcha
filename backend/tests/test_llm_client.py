"""Tests for backend.services.llm_client.

Covers all five LLM_PROVIDER values. The Gemini/Mistral SDKs
(google-genai, mistralai) aren't in requirements.txt (see the module's
own docstring for why), so those two providers' actual network calls
are exercised by injecting fake modules into sys.modules rather than
by installing the real packages -- this still exercises this module's
own adapter logic (argument shaping, response-text extraction) for
real, it just doesn't hit a real Gemini/Mistral endpoint.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.services.llm_client as llm_client


@pytest.fixture(autouse=True)
def _reset_between_tests(monkeypatch):
    """Every test gets a clean provider env and an uncached client --
    otherwise the first test's cached client would leak into the next."""
    for var in [
        "LLM_PROVIDER",
        "ANTHROPIC_CLAUDE_MODEL_ID",
        "BEDROCK_CLAUDE_MODEL_ID",
        "VERTEX_CLAUDE_MODEL_ID",
        "GEMINI_MODEL_ID",
        "MISTRAL_MODEL_ID",
        "GEMINI_RPM_BUDGET",
        "MISTRAL_RPM_BUDGET",
    ]:
        monkeypatch.delenv(var, raising=False)
    llm_client.reset_client_cache()
    llm_client._recent_request_times.clear()
    yield
    llm_client.reset_client_cache()


# --------------------------------------------------------------------------
# Model name / CrewAI model string selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,expected_model,expected_crewai",
    [
        (None, "claude-sonnet-5", "anthropic/claude-sonnet-5"),  # default, LLM_PROVIDER unset
        ("anthropic", "claude-sonnet-5", "anthropic/claude-sonnet-5"),
        ("bedrock", "anthropic.claude-sonnet-5", "bedrock/anthropic.claude-sonnet-5"),
        ("vertex", "claude-sonnet-5", "vertex_ai/claude-sonnet-5"),
        ("gemini", "gemini-3.5-flash-lite", "gemini/gemini-3.5-flash-lite"),
        ("mistral", "mistral-large-latest", "mistral/mistral-large-latest"),
    ],
)
def test_default_model_and_crewai_string_per_provider(monkeypatch, provider, expected_model, expected_crewai):
    if provider is not None:
        monkeypatch.setenv("LLM_PROVIDER", provider)
    assert llm_client.get_model_name() == expected_model
    assert llm_client.get_crewai_model_string() == expected_crewai


def test_model_id_override_env_var_wins(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL_ID", "gemini-3.5-flash")
    assert llm_client.get_model_name() == "gemini-3.5-flash"
    assert llm_client.get_crewai_model_string() == "gemini/gemini-3.5-flash"


def test_empty_override_env_var_falls_back_to_default(monkeypatch):
    """Regression test: .env.example ships these vars present-but-empty
    (documented, fillable) -- an empty string must not win over the
    real default the way os.getenv(key, default) alone would allow."""
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_MODEL_ID", "")
    assert llm_client.get_model_name() == "mistral-large-latest"


# --------------------------------------------------------------------------
# get_llm_client() -- client construction per provider
# --------------------------------------------------------------------------


def test_default_provider_returns_async_anthropic():
    from anthropic import AsyncAnthropic

    assert isinstance(llm_client.get_llm_client(), AsyncAnthropic)


def test_bedrock_provider_returns_async_anthropic_bedrock(monkeypatch):
    from anthropic import AsyncAnthropicBedrock

    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    assert isinstance(llm_client.get_llm_client(), AsyncAnthropicBedrock)


def test_vertex_provider_returns_async_anthropic_vertex(monkeypatch):
    from anthropic import AsyncAnthropicVertex

    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    assert isinstance(llm_client.get_llm_client(), AsyncAnthropicVertex)


def test_gemini_and_mistral_providers_return_adapter_with_messages_create(monkeypatch):
    for provider in ("gemini", "mistral"):
        llm_client.reset_client_cache()
        monkeypatch.setenv("LLM_PROVIDER", provider)
        client = llm_client.get_llm_client()
        assert hasattr(client, "messages")
        assert hasattr(client.messages, "create")


def test_client_is_cached_across_calls():
    first = llm_client.get_llm_client()
    second = llm_client.get_llm_client()
    assert first is second


# --------------------------------------------------------------------------
# Rate limiter -- confirmed necessary live (see llm_client.py's own
# comment): a real 17-case evaluation run against Gemini's free tier
# failed 82.4% of cases with 429s until this was added. time.sleep and
# time.monotonic are monkeypatched so these tests run instantly instead
# of actually waiting up to 60s.
# --------------------------------------------------------------------------


def test_no_throttling_for_default_provider(monkeypatch):
    """anthropic (the default) has no configured RPM budget -- must
    never sleep, regardless of how many calls happen."""
    sleep_calls = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: sleep_calls.append(s))

    for _ in range(50):
        llm_client.throttle_llm_call_sync()

    assert sleep_calls == []


def test_gemini_throttles_once_budget_exhausted(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_RPM_BUDGET", "3")

    fake_now = [1000.0]
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: fake_now[0])
    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        fake_now[0] += seconds  # simulate time passing during the sleep

    monkeypatch.setattr(llm_client.time, "sleep", fake_sleep)

    # First 3 calls (the budget) should not need to sleep at all.
    for _ in range(3):
        llm_client.throttle_llm_call_sync()
    assert sleep_calls == []

    # The 4th call exceeds the budget within the same 60s window --
    # must sleep roughly until the oldest of the 3 tracked calls ages
    # out (60s after it was recorded, plus this module's 0.5s margin).
    llm_client.throttle_llm_call_sync()
    assert len(sleep_calls) == 1
    assert 59.5 <= sleep_calls[0] <= 60.5


def test_gemini_does_not_throttle_when_calls_are_spread_out(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_RPM_BUDGET", "3")

    fake_now = [1000.0]
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: fake_now[0])
    sleep_calls = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: sleep_calls.append(s))

    for _ in range(3):
        llm_client.throttle_llm_call_sync()
    # Enough real time has passed that the tracked calls have aged out
    # of the 60s window -- a 4th call should proceed immediately.
    fake_now[0] += 61
    llm_client.throttle_llm_call_sync()

    assert sleep_calls == []


def test_mistral_has_no_budget_by_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    sleep_calls = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: sleep_calls.append(s))

    for _ in range(50):
        llm_client.throttle_llm_call_sync()

    assert sleep_calls == []


def test_mistral_rpm_budget_override_is_honored(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_RPM_BUDGET", "1")

    fake_now = [1000.0]
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: fake_now[0])
    sleep_calls = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: sleep_calls.append(s))

    llm_client.throttle_llm_call_sync()
    assert sleep_calls == []
    llm_client.throttle_llm_call_sync()
    assert len(sleep_calls) == 1


# --------------------------------------------------------------------------
# Gemini adapter: request shaping + response-text extraction, against a
# fake google.genai module (real package isn't in requirements.txt).
# --------------------------------------------------------------------------


@pytest.fixture
def fake_genai_module(monkeypatch):
    """Injects a fake google.genai module so _gemini_complete's lazy
    `from google import genai` / `from google.genai import types`
    imports resolve to test doubles instead of failing with
    ModuleNotFoundError (the real package isn't installed)."""
    fake_response = MagicMock()
    fake_response.text = "fake gemini response text"

    fake_client_instance = MagicMock()
    fake_client_instance.aio.models.generate_content = AsyncMock(return_value=fake_response)

    fake_client_class = MagicMock(return_value=fake_client_instance)

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = fake_client_class

    fake_types = types.ModuleType("google.genai.types")

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_types.GenerateContentConfig = _FakeGenerateContentConfig

    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    return {"client_class": fake_client_class, "client_instance": fake_client_instance, "response": fake_response}


@pytest.mark.asyncio
async def test_gemini_complete_returns_response_text(fake_genai_module, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    text = await llm_client._gemini_complete(
        model="gemini-2.0-flash",
        max_tokens=100,
        system="You are a test system prompt.",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert text == "fake gemini response text"
    generate_content = fake_genai_module["client_instance"].aio.models.generate_content
    generate_content.assert_awaited_once()
    call_kwargs = generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.0-flash"
    assert call_kwargs["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert call_kwargs["config"].kwargs["system_instruction"] == "You are a test system prompt."
    # 100 + THINKING_TOKEN_PADDING, not the bare max_tokens -- Gemini's
    # "thinking" models spend part of the output budget on internal
    # reasoning before the visible answer; see _gemini_complete's
    # comment (confirmed live against gemini-3.6-flash, which silently
    # returned an empty/truncated response without this padding).
    assert call_kwargs["config"].kwargs["max_output_tokens"] == 100 + llm_client.THINKING_TOKEN_PADDING


@pytest.mark.asyncio
async def test_gemini_complete_maps_assistant_role_to_model(fake_genai_module, monkeypatch):
    """Gemini uses 'model' for the model's own turns, not 'assistant' --
    this is exactly the kind of thing that silently produces wrong/
    rejected requests if missed."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    await llm_client._gemini_complete(
        model="gemini-2.0-flash",
        max_tokens=100,
        system="sys",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "correction"},
        ],
    )

    contents = fake_genai_module["client_instance"].aio.models.generate_content.call_args.kwargs["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]


# --------------------------------------------------------------------------
# Mistral adapter: request shaping + response-text extraction, against a
# fake mistralai module (real package isn't in requirements.txt).
# --------------------------------------------------------------------------


@pytest.fixture
def fake_mistralai_module(monkeypatch):
    fake_message = MagicMock()
    fake_message.content = "fake mistral response text"
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]

    fake_client_instance = MagicMock()
    fake_client_instance.chat.complete_async = AsyncMock(return_value=fake_response)

    fake_client_class = MagicMock(return_value=fake_client_instance)

    fake_mistralai = types.ModuleType("mistralai")
    fake_mistralai.Mistral = fake_client_class

    monkeypatch.setitem(sys.modules, "mistralai", fake_mistralai)

    return {"client_instance": fake_client_instance, "response": fake_response}


@pytest.mark.asyncio
async def test_mistral_complete_returns_response_text(fake_mistralai_module, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")

    text = await llm_client._mistral_complete(
        model="mistral-large-latest",
        max_tokens=100,
        system="You are a test system prompt.",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert text == "fake mistral response text"
    complete_async = fake_mistralai_module["client_instance"].chat.complete_async
    complete_async.assert_awaited_once()
    call_kwargs = complete_async.call_args.kwargs
    assert call_kwargs["model"] == "mistral-large-latest"
    assert call_kwargs["max_tokens"] == 100
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "You are a test system prompt."},
        {"role": "user", "content": "hello"},
    ]


# --------------------------------------------------------------------------
# Full round-trip through the adapter's .messages.create() shim -- proves
# the Anthropic-shaped interface (.content[0].text) that every agent
# actually calls works end-to-end for the non-Anthropic providers too.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_adapter_messages_create_matches_anthropic_shape(fake_genai_module, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    client = llm_client.get_llm_client()
    response = await client.messages.create(
        model="gemini-2.0-flash",
        max_tokens=50,
        system="sys",
        messages=[{"role": "user", "content": "hi"}],
    )

    # This is the exact shape every agent's _call_model() reads --
    # response.content[0].text -- regardless of provider.
    assert response.content[0].text == "fake gemini response text"
