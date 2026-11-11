"""LLM-as-judge metrics for MedOrchestrate reports (RAGAS-style, implemented natively).

These follow the DEFINITIONS of RAGAS's faithfulness and context-precision
metrics but are NOT the ragas library: ragas drags in langchain and would risk
the CrewAI/litellm environment, and it assumes a question -> free-text answer
shape this multi-diagnosis report does not have. If you want the library's
numbers, feed it the same (case, contexts, answer) triples built here.

Metrics (all judged by an LLM; every function takes an injectable ``ask`` so
tests never call a model):

* diagnosis_correct   - does any system top-3 diagnosis semantically match the
                        answer key's acceptable differential? ('abstain' cases are
                        scored deterministically: correct = flagged for review.)
* faithfulness        - for each diagnosis, do the passages it cites actually
                        support it?  supported=1, partial=0.5, unsupported=0.
* context_precision   - RAGAS average precision over the displayed RAG passages,
                        each judged relevant/irrelevant to the case.
* unsupported_claims  - specific findings/mechanisms the report asserts that are
                        in neither the inputs nor the cited evidence.

LIMITS (state them wherever a number from here is quoted):
* The judge is an LLM; if it is the same model family that wrote the report it
  tends to favour its own style. Use JUDGE_PROVIDER to pick a different one.
* The answer key is developer-drafted, not clinician-verified.
* Small n: this is a consistency check, not clinical validation.
"""

from __future__ import annotations

import json
import re
from typing import Awaitable, Callable

Ask = Callable[[str], Awaitable[str]]

SYSTEM = (
    "You are a strict, careful evaluator of clinical decision-support output. "
    "You judge only what is shown to you; you never use outside knowledge to "
    "fill gaps in the evidence. Reply with a single JSON object and nothing else."
)

CONFIDENCE_THRESHOLD = 0.65
VERDICT_SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_json(text: str) -> dict:
    """Tolerant JSON extraction: strips code fences and surrounding prose."""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in judge reply: {text[:120]!r}")
    return json.loads(cleaned[start : end + 1])


async def ask_json(ask: Ask, prompt: str, retries: int = 1) -> dict:
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            return parse_json(await ask(prompt))
        except (ValueError, json.JSONDecodeError) as exc:
            last = exc
            prompt += "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
    raise ValueError(f"judge returned invalid JSON after retry: {last}")


def case_summary(case: dict) -> str:
    parts = []
    p = case.get("patient") or {}
    parts.append(f"Patient: {p.get('age', '?')}-year-old {p.get('sex', '?')}")
    for key, label in (("clinical_text", "Symptoms"), ("medical_report", "Report")):
        if case.get(key):
            parts.append(f"{label}: {case[key]}")
    if case.get("history"):
        parts.append(f"History: {', '.join(map(str, case['history']))}")
    if case.get("labs"):
        parts.append(f"Labs: {case['labs']}")
    return "\n".join(parts)


def imaging_summary(report: dict) -> str:
    im = report.get("imaging")
    if not im:
        return "No imaging analysed."
    if im.get("unsupported_modality"):
        return "An image was supplied but its modality is unsupported (not assessed)."
    return f"Imaging: {im.get('modality')} - findings: {', '.join(im.get('findings') or []) or 'none'}"


def split_evidence(report: dict) -> tuple[list[str], list[str]]:
    """Displayed RAG passages and live items as plain text, in display order,
    so '[RAG2]' / '[LIVE1]' citations can be resolved the way the UI shows them."""
    rag, live = [], []
    for e in report.get("evidence", []):
        if e.get("type") == "rag":
            rag.append(" ".join((e.get("text") or "").split()))
        elif e.get("type") == "live":
            live.append(" ".join(f"{e.get('title', '')}. {e.get('summary', '')}".split()))
    return rag, live


_CITE = re.compile(r"\[(RAG|LIVE)(\d+)\]\s*(.*)", re.S)


def resolve_citation(cite: str, rag: list[str], live: list[str]) -> str:
    """Text a citation points at. A tagged citation is looked up in the displayed
    evidence; if the index is out of range, fall back to the text the model wrote
    after the tag (and mark it, so the judge/reader can see it was unresolved)."""
    m = _CITE.match(cite.strip())
    if not m:
        return cite.strip()
    kind, idx, inline = m.group(1), int(m.group(2)), m.group(3).strip()
    pool = rag if kind == "RAG" else live
    if 0 < idx <= len(pool):
        return pool[idx - 1]
    return f"(unresolved citation) {inline}"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def top_names(report: dict, k: int = 3) -> list[str]:
    ranked = sorted(report.get("diagnoses", []), key=lambda d: d.get("rank", 99))
    return [d["name"] for d in ranked[:k]]


async def judge_diagnosis(ask: Ask, report: dict, key: dict) -> dict:
    """{'correct': bool, 'matched': str|None, 'method': 'judge'|'rule'}"""
    if key["kind"] == "abstain":
        flagged = bool(report.get("review_required")) or report.get("confidence", 1.0) < CONFIDENCE_THRESHOLD
        return {"correct": flagged, "matched": None, "method": "rule"}
    names = top_names(report)
    if not names:
        return {"correct": False, "matched": None, "method": "rule"}
    prompt = (
        "Acceptable diagnoses (any one of these is a correct answer):\n- "
        + "\n- ".join(key["differential"])
        + "\n\nSystem's top diagnoses:\n- "
        + "\n- ".join(names)
        + "\n\nIs any system diagnosis clinically the same condition as, or a more specific "
        "form of, an acceptable diagnosis? A broader or merely related condition does NOT count."
        '\nReply: {"match": true|false, "matched_system_diagnosis": "<name or null>", "reason": "<short>"}'
    )
    out = await ask_json(ask, prompt)
    return {"correct": bool(out.get("match")), "matched": out.get("matched_system_diagnosis"), "method": "judge"}


async def judge_faithfulness(ask: Ask, report: dict, case: dict) -> dict:
    """Per-diagnosis verdicts against the passages each diagnosis cites."""
    rag, live = split_evidence(report)
    items = []
    for d in report.get("diagnoses", []):
        cites = [resolve_citation(c, rag, live) for c in d.get("supporting_evidence", [])]
        items.append({"diagnosis": d["name"], "cited_evidence": cites or ["(none cited)"]})
    if not items:
        return {"score": None, "verdicts": []}
    prompt = (
        f"Case:\n{case_summary(case)}\n\n"
        "For each diagnosis below, judge ONLY from its cited evidence whether the evidence supports "
        "this diagnosis for this patient. 'supported' = the evidence directly backs it; 'partial' = "
        "related or only indirectly supportive; 'unsupported' = does not back it, is about something "
        "else, or nothing was cited.\n\n"
        + json.dumps(items, indent=1)
        + '\n\nReply: {"verdicts": [{"diagnosis": "...", "verdict": "supported|partial|unsupported", "reason": "<short>"}]}'
    )
    out = await ask_json(ask, prompt)
    verdicts = out.get("verdicts", [])
    scores = [VERDICT_SCORE.get(str(v.get("verdict", "")).lower()) for v in verdicts]
    scores = [s for s in scores if s is not None]
    return {"score": (sum(scores) / len(scores)) if scores else None, "verdicts": verdicts}


def average_precision(relevant: list[bool]) -> float | None:
    """RAGAS context precision: mean of precision@k over the relevant ranks."""
    if not any(relevant):
        return 0.0 if relevant else None
    hits, total = 0, 0.0
    for k, rel in enumerate(relevant, start=1):
        if rel:
            hits += 1
            total += hits / k
    return total / hits


async def judge_context_precision(ask: Ask, report: dict, case: dict) -> dict:
    rag, _ = split_evidence(report)
    if not rag:
        return {"score": None, "relevant": []}
    numbered = "\n".join(f"[{i}] {t[:600]}" for i, t in enumerate(rag, 1))
    prompt = (
        f"Case:\n{case_summary(case)}\n{imaging_summary(report)}\n\n"
        "Retrieved passages, in rank order:\n" + numbered + "\n\n"
        "For each passage, is it useful for working out this patient's diagnosis (relevant) "
        "or not? Judge usefulness to THIS case, not general medical interest."
        '\nReply: {"relevant": [true|false, ...]} with exactly one entry per passage, in order.'
    )
    out = await ask_json(ask, prompt)
    rel = [bool(x) for x in out.get("relevant", [])][: len(rag)]
    if len(rel) != len(rag):
        raise ValueError(f"judge returned {len(rel)} relevance flags for {len(rag)} passages")
    return {"score": average_precision(rel), "relevant": rel}


async def judge_unsupported_claims(ask: Ask, report: dict, case: dict) -> dict:
    rag, live = split_evidence(report)
    cited = []
    for d in report.get("diagnoses", []):
        cited += [resolve_citation(c, rag, live) for c in d.get("supporting_evidence", [])]
    prompt = (
        f"INPUTS (the only facts known about the patient):\n{case_summary(case)}\n{imaging_summary(report)}\n\n"
        "REPORT diagnoses and the evidence they cite:\n"
        + json.dumps([{"name": d["name"], "cited": d.get("supporting_evidence", [])} for d in report.get("diagnoses", [])], indent=1)
        + "\n\nCITED PASSAGES:\n"
        + "\n".join(f"- {c[:400]}" for c in cited)
        + "\n\nList any specific patient finding, event or mechanism that the report asserts as true of "
        "THIS patient but that appears in neither the INPUTS nor the cited passages (for example an "
        "invented trauma, symptom or imaging finding). Diagnoses themselves are not claims to list; "
        "only invented patient facts. If none, return an empty list."
        '\nReply: {"unsupported_claims": ["..."]}'
    )
    out = await ask_json(ask, prompt)
    claims = [str(c) for c in out.get("unsupported_claims", []) if str(c).strip()]
    return {"count": len(claims), "claims": claims}


async def judge_report(ask: Ask, report: dict, case: dict, key: dict) -> dict:
    """All four metrics for one report. A failing metric records its error and
    does not stop the others (the judge is also a flaky network call)."""
    result: dict = {"case_id": case.get("case_id")}
    for name, coro in (
        ("diagnosis", judge_diagnosis(ask, report, key)),
        ("faithfulness", judge_faithfulness(ask, report, case)),
        ("context_precision", judge_context_precision(ask, report, case)),
        ("unsupported", judge_unsupported_claims(ask, report, case)),
    ):
        try:
            result[name] = await coro
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            result[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return result
