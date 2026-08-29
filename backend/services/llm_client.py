"""Provider-selectable LLM client factory.

Every agent that calls Claude directly (clinical, fusion, optimizer,
evidence) previously constructed its own `AsyncAnthropic()` client,
which only reads ANTHROPIC_API_KEY. This module centralizes that
construction behind one env var, LLM_PROVIDER, with two tiers of
options:

  - "anthropic" (default), "bedrock", "vertex": all three are
    genuinely Claude, just served through a different cloud's
    auth/billing instead of Anthropic's own console -- useful because
    getting an Anthropic API key needs billing setup that can be a
    real blocker under a tight deadline, while a new AWS/GCP account
    commonly comes with free trial credit.
  - "gemini", "mistral": genuinely different model vendors, offered
    as a last-resort fallback if none of the above work out in time.
    Both have low-friction free-tier API keys. Note: this is a
    materially different claim than "we used Claude" -- see the
    README's "Note on LLM provider choice" section before using this
    for anything you'll present as Claude-based work.

Set LLM_PROVIDER in .env to one of: "anthropic", "bedrock", "vertex",
"gemini", "mistral". Each provider's own credentials are read directly
from the environment by that provider's SDK client (not duplicated
here):
  - anthropic: ANTHROPIC_API_KEY
  - bedrock:   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
               (standard boto3 env vars; requires `pip install boto3`
               and one-time "Model access" approval in the Bedrock
               console for the Claude model you pick)
  - vertex:    VERTEX_PROJECT_ID, VERTEX_REGION,
               GOOGLE_APPLICATION_CREDENTIALS (path to a service
               account JSON key; requires `pip install google-auth
               google-cloud-aiplatform`)
  - gemini:    GEMINI_API_KEY (from aistudio.google.com -- a free-tier
               key, no billing/cloud account needed; requires
               `pip install google-genai`)
  - mistral:   MISTRAL_API_KEY (from console.mistral.ai -- has a free
               tier; requires `pip install mistralai`). Also needs
               `pip install litellm` for CrewAI's orchestration layer
               specifically -- "mistral" isn't one of CrewAI's
               natively-bundled providers the way anthropic/bedrock/
               gemini are (confirmed by CrewAI's own error message
               listing native providers if litellm is missing).

Model IDs differ by hosting surface, not just by model version -- see
get_model_name(). The defaults below were correct as of 2026-08; if a
provider's console shows a different exact ID for the model you have
access to, override it via the *_MODEL_ID env vars below rather than
editing this file.

Every agent still keeps its own `_get_client()` function (calling into
this module) rather than importing get_llm_client() directly
everywhere -- that's deliberate, so existing tests that monkeypatch
`<agent_module>._get_client` keep working unchanged.

Gemini and Mistral don't share the Anthropic SDK's `.messages.create()`
/ `response.content[0].text` shape, so for those two providers
get_llm_client() returns a small adapter (_AdapterClient below) that
presents that exact same shape while calling the real Gemini/Mistral
SDK underneath -- this is what lets every agent's existing call sites
(`client.messages.create(model=..., system=..., max_tokens=...,
messages=...)` / `response.content[0].text`) stay completely unchanged
regardless of which of the five providers is active.
"""

import os
import threading
import time

_client = None  # cached at module level; provider doesn't change mid-process

# See _gemini_complete()'s comment: newer Gemini models spend part of
# the output budget on internal "thinking" before the visible answer.
# 2048 was enough headroom in live testing against gemini-3.6-flash
# for this app's short prompts (extraction/reranking/refinement, not
# long-form generation) -- override GEMINI_THINKING_TOKEN_PADDING if a
# real call still comes back truncated for a longer prompt.
THINKING_TOKEN_PADDING = int(os.getenv("GEMINI_THINKING_TOKEN_PADDING") or 2048)


def get_llm_client():
    """Returns a cached client for whichever LLM_PROVIDER is
    configured, always exposing `.messages.create(model=, max_tokens=,
    system=, messages=) -> object with .content[0].text` -- callers
    never need to branch on provider."""
    global _client
    if _client is not None:
        return _client

    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "bedrock":
        from anthropic import AsyncAnthropicBedrock

        _client = AsyncAnthropicBedrock(aws_region=os.getenv("AWS_REGION") or "us-east-1")
    elif provider == "vertex":
        from anthropic import AsyncAnthropicVertex

        _client = AsyncAnthropicVertex(
            project_id=os.getenv("VERTEX_PROJECT_ID"),
            region=os.getenv("VERTEX_REGION") or "us-east5",
        )
    elif provider == "gemini":
        _client = _AdapterClient(_gemini_complete)
    elif provider == "mistral":
        _client = _AdapterClient(_mistral_complete)
    else:
        from anthropic import AsyncAnthropic

        _client = AsyncAnthropic()

    return _client


def get_model_name() -> str:
    """The model ID string to pass as `model=` in .messages.create()
    calls -- for the *directly-called* agents (clinical, fusion,
    optimizer, evidence). CrewAI's own model string is different, see
    get_crewai_model_string() below.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    # `or default`, not `os.getenv(key, default)`: .env.example ships
    # these override vars present-but-empty (so they show up as
    # documented, fillable settings), and os.getenv's own default only
    # kicks in when the key is entirely absent -- an empty string
    # would otherwise silently win and produce a blank model id.
    if provider == "bedrock":
        # No dated version suffix as of 2026-08 per AWS's own Bedrock
        # docs -- confirm against your account's actual "Model access"
        # page if this errors with a not-found/access-denied response.
        return os.getenv("BEDROCK_CLAUDE_MODEL_ID") or "anthropic.claude-sonnet-5"
    if provider == "vertex":
        return os.getenv("VERTEX_CLAUDE_MODEL_ID") or "claude-sonnet-5"
    if provider == "gemini":
        # Milestone II's design doc (Section 8) names "Gemini-2.0-
        # Flash" for this role, but that model was retired by Google
        # (confirmed live: the API itself now 404s on it and points to
        # gemini-3.6-flash as the replacement). gemini-3.6-flash itself
        # works but its free tier is capped at 5 requests/minute
        # (confirmed live: a real case run hit 429 RESOURCE_EXHAUSTED
        # partway through -- this pipeline's CrewAI orchestration layer
        # alone makes ~5-6 real calls per case, on top of the direct
        # per-agent calls, so 5/min isn't enough headroom for one case
        # to complete). gemini-3.5-flash-lite gets a much more generous
        # free-tier quota and is confirmed working -- used as the
        # default for that reason. Override via GEMINI_MODEL_ID for a
        # paid tier / a different model.
        return os.getenv("GEMINI_MODEL_ID") or "gemini-3.5-flash-lite"
    if provider == "mistral":
        return os.getenv("MISTRAL_MODEL_ID") or "mistral-large-latest"
    return os.getenv("ANTHROPIC_CLAUDE_MODEL_ID") or "claude-sonnet-5"


def get_crewai_model_string() -> str:
    """CrewAI's LLM (via its own native provider support, or LiteLLM
    for providers outside that native list) expects a provider-
    prefixed model string in a different format from the direct SDK
    calls' own `model=` value -- e.g. "bedrock/anthropic.claude-
    sonnet-5", not "anthropic.claude-sonnet-5" alone.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = get_model_name()
    if provider == "bedrock":
        return f"bedrock/{model}"
    if provider == "vertex":
        return f"vertex_ai/{model}"
    if provider == "gemini":
        return f"gemini/{model}"
    if provider == "mistral":
        # Requires `pip install litellm` -- see module docstring.
        return f"mistral/{model}"
    return f"anthropic/{model}"


def reset_client_cache() -> None:
    """Test-only: clears the cached client so a test can switch
    LLM_PROVIDER mid-session. Not used by application code."""
    global _client
    _client = None


# --------------------------------------------------------------------------
# Rate limiter -- confirmed necessary live, not speculative: a real
# 17-case evaluation run against gemini-3.5-flash-lite's free tier
# (15 requests/minute) failed 82.4% of cases with 429 RESOURCE_EXHAUSTED
# after the first case or two consumed the whole per-minute budget.
#
# This has to cover two genuinely different call paths sharing the same
# real quota: this module's own adapter functions (_gemini_complete /
# _mistral_complete, called from clinical/fusion/optimizer/evidence),
# AND CrewAI's own native-provider completion classes, which construct
# their own google.genai/mistralai client internally and are *not*
# reachable through this module's client factory at all -- so this
# can't be solved by throttling inside get_llm_client() alone.
# orchestrator.py calls throttle_llm_call_sync() directly, once per
# agent step, immediately before crew.kickoff() -- see its own comment
# for why that's the one place that sees every real call this pipeline
# makes, from either path.
#
# A simple blocking sleep (not asyncio.sleep) is deliberate: run_case()
# is fully synchronous end to end (see orchestrator.py's module
# docstring), and every LLM call in this codebase -- both the adapter
# functions here and CrewAI's own -- ultimately happens one at a time
# within that single synchronous flow, never concurrently, so there's
# nothing async to cooperate with; a plain threading.Lock is enough.
# --------------------------------------------------------------------------

_rate_limit_lock = threading.Lock()
_recent_request_times: list[float] = []


def _get_rpm_budget() -> int | None:
    """Requests-per-minute budget for the active provider, read live
    (like get_model_name(), not frozen at import) so a changed .env
    setting or a test's monkeypatch both take effect immediately.

    Deliberately under Gemini's published free-tier quota (12 vs 15)
    for margin -- multiple clock sources (this process's timer vs
    Google's own request-counting window) drifting by even a couple
    seconds at the boundary is enough to trigger a 429 right at the
    edge. None disables throttling entirely -- appropriate for
    anthropic/bedrock/vertex, whose standard tiers aren't this tightly
    capped, and left off for mistral by default too unless its own
    free tier proves similarly tight (set MISTRAL_RPM_BUDGET if so).
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "gemini":
        return int(os.getenv("GEMINI_RPM_BUDGET") or 12)
    if provider == "mistral":
        raw = os.getenv("MISTRAL_RPM_BUDGET")
        return int(raw) if raw else None
    return None


def throttle_llm_call_sync() -> None:
    """Blocks (if needed) so this process's real calls to the active
    LLM_PROVIDER stay under that provider's per-minute budget. A no-op
    for providers with no configured budget. Call once immediately
    before every real network call to the provider -- see module
    comment above for why both call paths need this.
    """
    budget = _get_rpm_budget()
    if not budget:
        return

    with _rate_limit_lock:
        now = time.monotonic()
        recent = [t for t in _recent_request_times if now - t < 60]
        if len(recent) >= budget:
            wait_seconds = 60 - (now - recent[0]) + 0.5
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            now = time.monotonic()
            recent = [t for t in _recent_request_times if now - t < 60]
        recent.append(now)
        _recent_request_times[:] = recent


# --------------------------------------------------------------------------
# Gemini / Mistral adapter -- presents the Anthropic SDK's
# .messages.create() / response.content[0].text shape over a genuinely
# different underlying SDK, so agent code never needs to know which
# provider is active. See module docstring for why.
# --------------------------------------------------------------------------


class _AnthropicShapedContentBlock:
    def __init__(self, text: str):
        self.text = text


class _AnthropicShapedResponse:
    def __init__(self, text: str):
        self.content = [_AnthropicShapedContentBlock(text)]


class _MessagesShim:
    def __init__(self, complete_fn):
        self._complete = complete_fn

    async def create(self, *, model, max_tokens, system, messages):
        text = await self._complete(model=model, max_tokens=max_tokens, system=system, messages=messages)
        return _AnthropicShapedResponse(text)


class _AdapterClient:
    """Duck-types as an AsyncAnthropic client: `.messages.create(...)`
    only -- nothing else in this codebase calls any other method on
    the client, so nothing else needs to be adapted."""

    def __init__(self, complete_fn):
        self.messages = _MessagesShim(complete_fn)


async def _gemini_complete(*, model: str, max_tokens: int, system: str, messages: list[dict]) -> str:
    from google import genai
    from google.genai import types

    throttle_llm_call_sync()

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    # Gemini's role for the model's own turns is "model", not
    # "assistant" -- everything else about the shape matches what
    # every agent already builds (role + plain-string content).
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            # Newer Gemini models spend part of the output-token budget
            # on internal "thinking" before the visible answer, on top
            # of max_tokens. Confirmed live: gemini-3.6-flash silently
            # returns an empty/truncated response (finish_reason=
            # MAX_TOKENS) at this app's normal (small, tuned for a
            # non-thinking model) max_tokens values -- thinking_budget=
            # 0 to disable it outright was rejected by this model (400
            # INVALID_ARGUMENT), so instead the visible budget is
            # padded with a flat allowance for thinking. If a call
            # still comes back truncated, raise THINKING_TOKEN_PADDING.
            max_output_tokens=max_tokens + THINKING_TOKEN_PADDING,
        ),
    )
    return response.text


async def _mistral_complete(*, model: str, max_tokens: int, system: str, messages: list[dict]) -> str:
    from mistralai import Mistral

    throttle_llm_call_sync()

    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    # Mistral's chat format is already OpenAI-style (system as a
    # regular message with role="system"), so no role remapping needed
    # beyond prepending the system prompt.
    full_messages = [{"role": "system", "content": system}, *messages]
    response = await client.chat.complete_async(model=model, messages=full_messages, max_tokens=max_tokens)
    return response.choices[0].message.content
