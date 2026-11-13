"""Turns an internal exception into a short sentence that is safe to show a clinician.

Provider exceptions carry things a report should never show: raw JSON error bodies, quota
and billing details, project or key identifiers, request URLs, stack-trace fragments. They
used to be pasted into `FinalReport.conflicts`, and from there into the API response and the
UI ("Fusion reasoning failed: 429 RESOURCE_EXHAUSTED ... project ..."). The full text still
goes to the structured log (backend/services/logging.py), where it belongs; the report gets
only the category below.

The match is on keywords in the exception's type name and text, so it is a best effort: an
error that matches nothing gets the generic sentence, never its own text.
"""

import json
import re

_RULES: list[tuple[tuple[str, ...], str]] = [
    (("429", "resource_exhausted", "quota", "rate limit", "rate_limit", "too many requests"),
     "the language-model provider's request limit was reached"),
    (("401", "403", "unauthenticated", "permission_denied", "permission denied", "api key", "api_key",
      "invalid x-api-key", "authentication", "credentials"),
     "the language-model provider rejected the system's credentials"),
    (("timeout", "timed out", "deadline", "connecterror", "connection", "unreachable", "name resolution", "dns"),
     "the language-model provider could not be reached in time"),
    (("500", "502", "503", "504", "unavailable", "overloaded", "internal error", "bad gateway"),
     "the language-model provider is temporarily unavailable"),
    (("valid fusionresult json", "jsondecodeerror", "validationerror", "not a json object", "must be a",
      "missing a valid", "schema", "truncated"),
     "the language model returned an answer the system could not use"),
]

GENERIC = "an unexpected error occurred"


def describe_failure(exc: BaseException) -> str:
    """A one-clause, provider-neutral reason, e.g. "the language-model provider's request limit
    was reached". Never contains any part of the exception's own message."""
    if isinstance(exc, json.JSONDecodeError):
        return _RULES[-1][1]
    haystack = f"{type(exc).__name__} {exc}".lower()
    for keywords, reason in _RULES:
        # a bare status code is matched as a whole number, so 500 is not found inside "after 5000 ms"
        if any((re.search(rf"(?<!\d){k}(?!\d)", haystack) if k.isdigit() else k in haystack) for k in keywords):
            return reason
    return GENERIC


def reasoning_failed_message(exc: BaseException) -> str:
    """The sentence placed in the report's conflicts when the fusion step fails."""
    return (
        f"The reasoning step could not be completed ({describe_failure(exc)}), so no diagnosis is shown. "
        "A clinician must review this case; retrying later may help."
    )
