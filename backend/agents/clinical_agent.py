"""Clinical agent: extracts ClinicalFindings from case text via Claude.

Schema-constrained LLM extraction only — no custom NER model. The model
is instructed to return raw JSON matching ClinicalFindings; the response
is validated with Pydantic and, on failure, retried once with an
explicit correction message before raising.
"""

import json

from anthropic import AsyncAnthropic
from pydantic import ValidationError

from backend.models.schemas import ClinicalFindings
from backend.services.llm_client import get_llm_client, get_model_name

MODEL = get_model_name()

SYSTEM_PROMPT = """You are a clinical information extraction system.

Given a patient's free-text clinical note and/or medical report, extract \
structured findings and return ONLY a single JSON object — no prose, no \
explanation, no markdown code fences — matching exactly this shape:

{
  "demographics": {},
  "symptoms": [],
  "history": [],
  "labs": {},
  "imaging_text_findings": []
}

Field rules:
- "demographics": any age/sex/other demographic details mentioned in the \
text itself (not inferred from outside context). Empty object if none.
- "symptoms": each distinct symptom or complaint as a short string. Also \
list clinically meaningful negatives the note states, as "absence of X" \
(for example "absence of breathlessness").
- "history": past medical history, conditions, or relevant background as \
short strings. Include the recent event or exposure that explains the \
complaint, and its timing, whenever the note gives one (for example \
"collision during sport", "fall two days ago", "on warfarin"); never drop \
the mechanism or the time course.
- "labs": lab test names mapped to their reported value (as a string, \
including units if given), e.g. {"hemoglobin": "12.1 g/dL"}.
- "imaging_text_findings": any imaging findings mentioned in the TEXT \
(e.g. from a prior radiology report) — not an actual image analysis.

Normalize clinical terminology and abbreviations to standard full terms \
(for example "HTN" -> "hypertension", "DM" -> "diabetes mellitus", "SOB" \
-> "shortness of breath", "CP" -> "chest pain"). Use empty objects/arrays \
for any field with no supporting information. Return valid JSON only."""

CORRECTION_MESSAGE = (
    "That response was not valid JSON matching the required schema. "
    "Return valid JSON only — no prose, no markdown code fences, no "
    "explanation — a single JSON object with exactly the keys "
    "demographics, symptoms, history, labs, imaging_text_findings."
)

_EMPTY_FINDINGS_KWARGS = {
    "demographics": {},
    "symptoms": [],
    "history": [],
    "labs": {},
    "imaging_text_findings": [],
}


class ClinicalExtractionError(RuntimeError):
    """Raised when the model fails to produce valid ClinicalFindings JSON."""


_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        # Routed through llm_client so this agent works unchanged
        # against Anthropic direct, Bedrock, Vertex, Gemini, or Mistral
        # -- see backend/services/llm_client.py for why and how to
        # switch (LLM_PROVIDER in .env). Kept as this module's own
        # _get_client() specifically so existing tests that
        # monkeypatch clinical_agent._get_client keep working.
        _client = get_llm_client()
    return _client


def _build_user_message(clinical_text: str | None, medical_report: str | None) -> str:
    parts = []
    if clinical_text:
        parts.append(f"Clinical text:\n{clinical_text}")
    if medical_report:
        parts.append(f"Medical report:\n{medical_report}")
    return "\n\n".join(parts)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_response(text: str) -> ClinicalFindings:
    payload = json.loads(_strip_code_fences(text))
    return ClinicalFindings.model_validate(payload)


async def _call_model(client: AsyncAnthropic, messages: list[dict]) -> str:
    response = await client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


async def extract_clinical_findings(
    clinical_text: str | None, medical_report: str | None
) -> ClinicalFindings:
    if not clinical_text and not medical_report:
        return ClinicalFindings(**_EMPTY_FINDINGS_KWARGS)

    client = _get_client()
    messages = [{"role": "user", "content": _build_user_message(clinical_text, medical_report)}]

    raw_response = await _call_model(client, messages)
    try:
        return _parse_response(raw_response)
    except (json.JSONDecodeError, ValidationError) as first_error:
        messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": CORRECTION_MESSAGE})
        retry_response = await _call_model(client, messages)
        try:
            return _parse_response(retry_response)
        except (json.JSONDecodeError, ValidationError) as second_error:
            raise ClinicalExtractionError(
                "Failed to extract valid ClinicalFindings JSON after retry: "
                f"{second_error}"
            ) from second_error


def _history_key(item: str) -> str:
    """Normalised form for de-duplicating history items ("History of Hypertension" == "hypertension")."""
    key = " ".join(item.lower().split())
    for prefix in ("history of ", "h/o "):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def merge_structured_fields(
    findings: ClinicalFindings, history: list[str] | None, labs: dict | None
) -> ClinicalFindings:
    """Fold the case's structured `history` and `labs` fields into the findings.

    The API accepts and stores `history` and `labs` on every case, but the
    extractor only ever reads clinical_text and medical_report, so before this
    merge they never reached retrieval, fusion or the confidence gate (17 of the
    31 test cases carry them, e.g. CASE018's atrial fibrillation + warfarin and
    CASE011's anemia + hemoglobin). Structured values are what the clinician
    typed into a field, so they win over anything the model inferred from prose:
    labs override same-named extracted labs; history is appended after the
    extracted items, de-duplicated case-insensitively.
    """
    seen = {_history_key(item) for item in findings.history}
    merged_history = list(findings.history)
    for item in history or []:
        item = (item or "").strip()
        if item and _history_key(item) not in seen:
            seen.add(_history_key(item))
            merged_history.append(item)
    return findings.model_copy(update={"history": merged_history, "labs": {**findings.labs, **(labs or {})}})
