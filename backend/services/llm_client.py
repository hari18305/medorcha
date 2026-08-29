"""Provider-selectable Anthropic-compatible client factory.

Every agent that calls Claude directly (clinical, fusion, optimizer,
evidence) previously constructed its own `AsyncAnthropic()` client,
which only reads ANTHROPIC_API_KEY. This module centralizes that
construction behind one env var, LLM_PROVIDER, so the app can run
against Anthropic's direct API, AWS Bedrock, or Google Cloud Vertex AI
-- interchangeably, without touching each agent's code -- because all
three surfaces are still genuinely Claude, served through a different
cloud's auth/billing instead of Anthropic's own console. That matters
here specifically because getting an Anthropic API key requires
billing setup that can be a real blocker under a tight deadline, while
a new AWS/GCP account commonly comes with free trial credit.

Set LLM_PROVIDER in .env to one of: "anthropic" (default), "bedrock",
"vertex". Each provider's own extra credentials are read directly from
the environment by that provider's SDK client (not duplicated here):
  - anthropic: ANTHROPIC_API_KEY
  - bedrock:   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
               (standard boto3 env vars; requires `pip install boto3`
               and one-time "Model access" approval in the Bedrock
               console for the Claude model you pick)
  - vertex:    VERTEX_PROJECT_ID, VERTEX_REGION,
               GOOGLE_APPLICATION_CREDENTIALS (path to a service
               account JSON key; requires `pip install google-auth
               google-cloud-aiplatform`)

Model IDs differ by hosting surface, not just by model version -- see
get_model_name(). The defaults below were correct as of 2026-08; if a
provider's console shows a different exact ID for the Claude model you
have access to, override it via the *_CLAUDE_MODEL_ID env vars below
rather than editing this file.

Each agent still keeps its own `_get_client()` function (calling into
this module) rather than importing get_async_anthropic_client()
directly everywhere -- that's deliberate, so existing tests that
monkeypatch `<agent_module>._get_client` keep working unchanged.
"""

import os

_client = None  # cached at module level; provider doesn't change mid-process


def get_async_anthropic_client():
    """Returns a cached async client for whichever LLM_PROVIDER is
    configured. Same .messages.create() interface regardless of
    provider -- callers don't need to branch on provider at all."""
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
    return os.getenv("ANTHROPIC_CLAUDE_MODEL_ID") or "claude-sonnet-5"


def get_crewai_model_string() -> str:
    """CrewAI's LLM (via LiteLLM) expects a provider-prefixed model
    string in a different format from the anthropic SDK's own `model=`
    value -- e.g. "bedrock/anthropic.claude-sonnet-5", not
    "anthropic.claude-sonnet-5" alone.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = get_model_name()
    if provider == "bedrock":
        return f"bedrock/{model}"
    if provider == "vertex":
        return f"vertex_ai/{model}"
    return f"anthropic/{model}"


def reset_client_cache() -> None:
    """Test-only: clears the cached client so a test can switch
    LLM_PROVIDER mid-session. Not used by application code."""
    global _client
    _client = None
