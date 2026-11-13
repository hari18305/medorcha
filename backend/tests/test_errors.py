import json

import pytest

from backend.services.errors import GENERIC, describe_failure, reasoning_failed_message

# Text that must never reach a report, whichever category it falls in.
SECRETS = ["AIzaSyDUMMYKEY123", "projects/my-secret-project-42", "https://generativelanguage.googleapis.com/v1beta/models",
           "billing", "sk-ant-api03-xyz"]


@pytest.mark.parametrize(
    "exc, expected",
    [
        (RuntimeError("429 RESOURCE_EXHAUSTED. You exceeded your current quota"), "request limit"),
        (RuntimeError("Error code: 429 - rate limit exceeded"), "request limit"),
        (RuntimeError("401 UNAUTHENTICATED: API key not valid"), "credentials"),
        (RuntimeError("403 PERMISSION_DENIED"), "credentials"),
        (TimeoutError("the read operation timed out"), "could not be reached"),
        (RuntimeError("httpx.ConnectError: connection refused"), "could not be reached"),
        (RuntimeError("503 UNAVAILABLE: the model is overloaded"), "temporarily unavailable"),
        (RuntimeError("Failed to produce a valid FusionResult JSON after retry: 'overall_confidence' must be a number"), "could not use"),
        (json.JSONDecodeError("Expecting value", "doc", 0), "could not use"),
        (RuntimeError("something nobody anticipated"), GENERIC),
    ],
)
def test_describe_failure_categories(exc, expected):
    assert expected in describe_failure(exc)


def test_a_status_code_is_matched_as_a_whole_number():
    # "5000" and "4290" must not be read as 500 or 429
    assert describe_failure(RuntimeError("waited 5000 ms then gave up")) == GENERIC
    assert describe_failure(RuntimeError("item 4290 not found")) == GENERIC


@pytest.mark.parametrize("secret", SECRETS)
def test_no_part_of_the_exception_text_reaches_the_message(secret):
    exc = RuntimeError(f"429 RESOURCE_EXHAUSTED {secret} quota exceeded for {secret}")
    message = reasoning_failed_message(exc)
    assert secret not in message and "RESOURCE_EXHAUSTED" not in message and "429" not in message


def test_the_message_says_what_to_do_and_that_no_diagnosis_is_shown():
    message = reasoning_failed_message(RuntimeError("boom"))
    assert "clinician must review" in message
    assert "no diagnosis is shown" in message
    assert "boom" not in message
