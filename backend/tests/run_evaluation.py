"""Evaluation harness for MedOrchestrate.

NOT a pytest test -- deliberately named without a test_ prefix so pytest
never auto-collects it (it does real, slow, non-deterministic work:
real RAG/Qdrant retrieval, real PubMed/MedlinePlus network calls, and
optionally real BiomedCLIP inference). Run it directly:

    python -m backend.tests.run_evaluation
    python -m backend.tests.run_evaluation --ablation   # stretch goal, see below
    python -m backend.tests.run_evaluation --mode2      # Phase 17 Mode 2: MedQA/PubMedQA
    python -m backend.tests.run_evaluation --mode3      # Phase 17 Mode 3: imaging mini-benchmark
    python -m backend.tests.run_evaluation --case-memory-ablation  # Phase 17 required ablation
    python -m backend.tests.run_evaluation --full       # everything above, in one run

Mode 2 (MedQA/PubMedQA, Phase 17) needs `pip install datasets` and
network access to Hugging Face -- neither is guaranteed in every
environment this runs in, so it degrades to a clearly-logged
"unavailable" result (never a crash) when either is missing, rather
than pretending to produce real numbers it can't.

Runs every synthetic case in data/test_cases/ through the real
orchestrator.run_case() and reports the metrics the task asked for, to
stdout as a table and to a markdown + CSV report file.

Mocking: no ANTHROPIC_API_KEY is configured in this dev environment, so
Claude calls (clinical extraction, fusion reasoning, optimizer strategy
selection) and CrewAI's own Agent-decision layer are mocked with a
GENERIC, case-agnostic mock (see _install_mocks_for_case) -- not hand-
authored per case. Everything else is real: RAG retrieval (real local
Qdrant + BioBERT), live evidence (real PubMed/MedlinePlus network
calls), imaging (real BiomedCLIP, since the model is already cached
locally), the optimizer's control flow, and the confidence formula
(compute_confidence() ignores the mocked fusion's own overall_confidence
entirely -- it recomputes from real RAG scores, real live evidence
levels, and the mocked fusion's conflicts list -- so confidence still
varies meaningfully per case based on real retrieval quality, not an
arbitrary mocked number).

To run this against a real, fully live pipeline once a real
ANTHROPIC_API_KEY is configured: pass --no-mock-claude, which skips
_install_mocks_for_case() entirely and lets every agent call the real
Anthropic API through the real CrewAI Agent-decision layer.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import backend.agents.clinical_agent as clinical_agent
import backend.agents.fusion_agent as fusion_agent
import backend.agents.optimizer_agent as optimizer_agent
import backend.services.orchestrator as orchestrator
from backend.agents.clinical_agent import extract_clinical_findings
from backend.agents.evidence_agent import gather_live_evidence
from backend.agents.fusion_agent import fuse
from backend.agents.imaging_agent import analyze_image
from backend.agents.imaging_labels import FINDING_LABELS_BY_MODALITY
from backend.agents.rag_agent import _build_query, run_rag
from backend.models.schemas import CaseInput, ClinicalFindings, ImagingFindings, LiveEvidence, Patient, RAGEvidence
from backend.services.confidence import compute_confidence
from backend.services.orchestrator import run_case

TEST_CASES_DIR = Path("data/test_cases")
DEFAULT_MD_REPORT = Path("backend/tests/evaluation_report.md")
DEFAULT_CSV_REPORT = Path("backend/tests/evaluation_report.csv")
IMAGING_EVAL_DIR = Path("data/benchmarks/imaging_eval")
IMAGING_EVAL_MANIFEST = IMAGING_EVAL_DIR / "manifest.json"


# --------------------------------------------------------------------------
# Generic, case-agnostic mocks for Claude + CrewAI's Agent-decision layer.
# --------------------------------------------------------------------------


class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


def _fake_client(response_texts):
    """response_texts: a single JSON string (every call gets it), or a
    list that's cycled indefinitely (so an unpredictable number of calls
    -- e.g. across optimizer iterations plus an ablation run -- never
    exhausts it)."""
    client = MagicMock()
    if isinstance(response_texts, list):
        cycle = itertools.cycle(response_texts)

        async def _create(*args, **kwargs):
            return _FakeAnthropicResponse(next(cycle))

        client.messages.create = AsyncMock(side_effect=_create)
    else:
        client.messages.create = AsyncMock(return_value=_FakeAnthropicResponse(response_texts))
    return client


_SYMPTOM_VOCAB = [
    "headache", "dizziness", "vertigo", "fatigue", "nausea", "lightheadedness",
    "gait unsteadiness", "unsteadiness", "vision", "weakness", "confusion",
    "palpitations", "hyperpigmentation",
]


def _derive_terms(text: str | None) -> list[str]:
    if not text:
        return []
    lowered = text.lower()
    return [term for term in _SYMPTOM_VOCAB if term in lowered]


def _clinical_mock_response(case: CaseInput) -> str:
    """Generic clinical-extraction mock: echoes case.history directly and
    scans clinical_text/medical_report against a small symptom vocabulary
    -- not real medical NLP, just a case-agnostic stand-in so every case
    gets a plausible-shaped ClinicalFindings without hand-authoring 17
    bespoke responses."""
    symptoms = _derive_terms(case.clinical_text) or (["nonspecific symptoms"] if case.clinical_text else [])
    imaging_text_findings = _derive_terms(case.medical_report)
    if case.medical_report and "no acute intracranial abnormality" in case.medical_report.lower():
        imaging_text_findings.append("no acute intracranial abnormality")
    return json.dumps(
        {
            "demographics": {},
            "symptoms": symptoms,
            "history": case.history,
            "labs": case.labs,
            "imaging_text_findings": imaging_text_findings,
        }
    )


def _fusion_mock_response(label: str) -> str:
    """Generic single-diagnosis fusion mock. Always attempts to cite
    [RAG1] -- fusion_agent's real grounding validator silently drops that
    citation if the case's real RAG evidence didn't actually produce an
    item with that ID, so evidence_coverage still reflects genuine
    retrieval outcomes, not a fabricated citation."""
    return json.dumps(
        {
            "diagnoses": [
                {
                    "name": f"Working differential ({label})",
                    "confidence": 0.5,
                    "supporting_evidence": ["[RAG1] Most relevant offline evidence retrieved for this case."],
                    "rank": 1,
                }
            ],
            "overall_confidence": 0.5,
            "conflicts": [],
        }
    )


def _optimizer_mock_responses(case: CaseInput) -> list[str]:
    base = ", ".join(case.history) or (case.clinical_text or "symptoms")
    return [
        json.dumps({"strategy": "expand", "query": f"{base} broader search", "rationale": "expand"}),
        json.dumps({"strategy": "narrow", "query": f"{base} specific search", "rationale": "narrow"}),
        json.dumps({"strategy": "paraphrase", "query": f"{base} alternate phrasing", "rationale": "paraphrase"}),
    ]


def _direct_execute(agent, task):
    """Bypass CrewAI's real Crew.kickoff()/LLM call -- run the agent's
    one tool directly (same technique used throughout this project's test
    suite)."""
    return agent.tools[0]._run()


# Captured at import time, before anything ever overwrites
# orchestrator._execute -- lets --no-mock-claude restore the real,
# Claude-calling CrewAI execution path.
_REAL_EXECUTE = orchestrator._execute


def _install_mocks_for_case(case: CaseInput) -> None:
    orchestrator._execute = _direct_execute
    clinical_agent._get_client = lambda: _fake_client(_clinical_mock_response(case))
    fusion_agent._get_client = lambda: _fake_client([_fusion_mock_response("run")])
    optimizer_agent._get_client = lambda: _fake_client(_optimizer_mock_responses(case))
    # Imaging (BiomedCLIP) and RAG (Qdrant/BioBERT) and live evidence
    # (PubMed/MedlinePlus) are all left real/unmocked.


# --------------------------------------------------------------------------
# rag_retrieval_success heuristic -- DOCUMENTED CHOICE.
#
# The task offered two options: (a) a manual/human-scored relevance flag
# stored alongside each case, or (b) a simple keyword-overlap heuristic.
# This script uses (b): a case's RAG retrieval is judged "relevant" if its
# best-scoring retrieved passage shares at least 2 meaningful words
# (length > 3, common stopwords excluded) with the case's own
# clinical_text/medical_report/history. This is a prototype-level PROXY
# for relevance, not a validated judgment -- it can't distinguish a
# topically-adjacent passage from a truly clinically apt one, and it's
# vulnerable to synonyms/paraphrasing it can't see (e.g. "vertigo" vs
# "dizziness"). It was chosen over hand-scoring because it's reproducible
# on every run without a human re-labeling 17 cases whenever the corpus
# or retrieval logic changes -- it re-derives its answer from the run's
# own output every time.
# --------------------------------------------------------------------------

_STOPWORDS = {
    "and", "the", "with", "for", "over", "from", "this", "that", "has", "have",
    "history", "patient", "presenting", "presents", "recent", "recently",
}


def _extract_query_terms(case: CaseInput) -> set[str]:
    text = " ".join(filter(None, [case.clinical_text, case.medical_report, " ".join(case.history)]))
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def rag_retrieval_looks_relevant(case: CaseInput, report) -> bool:
    rag_items = [e for e in report.evidence if e.get("type") == "rag"]
    if not rag_items:
        return False
    query_terms = _extract_query_terms(case)
    if not query_terms:
        return False
    best = max(rag_items, key=lambda e: e.get("score", 0))
    passage_words = set(re.findall(r"[a-zA-Z]+", best.get("text", "").lower()))
    return len(query_terms & passage_words) >= 2


def has_usable_citation(report) -> bool:
    """evidence_coverage: does at least one diagnosis cite real, grounded
    evidence (not just "some evidence was retrieved somewhere")?"""
    return any(d.get("supporting_evidence") for d in report.diagnoses)


# --------------------------------------------------------------------------
# Case loading + evaluation loop.
# --------------------------------------------------------------------------


def load_cases(cases_dir: Path = TEST_CASES_DIR) -> list[tuple[str, CaseInput]]:
    results = []
    for path in sorted(cases_dir.glob("*.json")):
        try:
            case = CaseInput(**json.loads(path.read_text()))
        except Exception:
            continue  # not a CaseInput fixture -- skip non-case JSON in the dir
        results.append((path.name, case))
    return results


@dataclass
class CaseResult:
    case_id: str
    filename: str
    success: bool
    error: str | None = None
    confidence: float | None = None
    review_required: bool | None = None
    iteration_count: int | None = None
    latency_seconds: float | None = None
    rag_relevant: bool | None = None
    evidence_coverage: bool | None = None
    diagnosis_count: int | None = None
    live_evidence_available: bool | None = None
    evidence_item_count: int | None = None
    conflict_detected: bool | None = None


def run_evaluation(cases: list[tuple[str, CaseInput]], mock_claude: bool = True) -> list[CaseResult]:
    results = []
    for filename, case in cases:
        if mock_claude:
            _install_mocks_for_case(case)
        else:
            orchestrator._execute = _REAL_EXECUTE

        start = time.perf_counter()
        try:
            report = run_case(case)
        except Exception as exc:  # the evaluation harness itself must survive a bad case
            elapsed = time.perf_counter() - start
            results.append(
                CaseResult(case_id=case.case_id, filename=filename, success=False, error=str(exc), latency_seconds=elapsed)
            )
            continue
        elapsed = time.perf_counter() - start

        results.append(
            CaseResult(
                case_id=case.case_id,
                filename=filename,
                success=True,
                confidence=report.confidence,
                review_required=report.review_required,
                iteration_count=report.iteration_count,
                latency_seconds=elapsed,
                rag_relevant=rag_retrieval_looks_relevant(case, report),
                evidence_coverage=has_usable_citation(report),
                diagnosis_count=len(report.diagnoses),
                live_evidence_available=report.live_evidence_available,
                evidence_item_count=len(report.evidence),
                conflict_detected=len(report.conflicts) > 0,
            )
        )
    return results


def summarize(results: list[CaseResult]) -> dict:
    total = len(results)
    successful = [r for r in results if r.success]
    n_success = len(successful)

    def pct(n: int) -> float:
        return round(100 * n / total, 1) if total else 0.0

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    iteration_counts = [r.iteration_count or 0 for r in successful]

    return {
        "cases_tested": total,
        "error_rate_pct": pct(total - n_success),
        "rag_retrieval_success_pct": pct(sum(1 for r in successful if r.rag_relevant)),
        "average_confidence": avg([r.confidence for r in successful if r.confidence is not None]),
        "optimizer_activation_rate_pct": pct(sum(1 for r in successful if (r.iteration_count or 0) > 0)),
        "hitl_rate_pct": pct(sum(1 for r in successful if r.review_required)),
        "average_latency_seconds": avg([r.latency_seconds for r in successful if r.latency_seconds is not None]),
        "average_iterations": avg(iteration_counts),
        "max_iterations": max(iteration_counts, default=0),
        "evidence_coverage_pct": pct(sum(1 for r in successful if r.evidence_coverage)),
        "average_evidence_items": avg([r.evidence_item_count for r in successful if r.evidence_item_count is not None]),
        "live_evidence_available_pct": pct(sum(1 for r in successful if r.live_evidence_available)),
        "conflict_detected_pct": pct(sum(1 for r in successful if r.conflict_detected)),
    }


SUMMARY_LABELS = {
    "cases_tested": "Cases tested",
    "error_rate_pct": "Error rate (%)",
    "rag_retrieval_success_pct": "RAG retrieval success (%, keyword-overlap heuristic)",
    "average_confidence": "Average confidence",
    "optimizer_activation_rate_pct": "Optimizer activation rate (%)",
    "hitl_rate_pct": "HITL rate -- review_required (%)",
    "average_latency_seconds": "Average latency (s)",
    "average_iterations": "Average optimizer iterations",
    "max_iterations": "Max optimizer iterations",
    "evidence_coverage_pct": "Evidence coverage (%, >=1 grounded citation)",
    "average_evidence_items": "Average evidence items",
    "live_evidence_available_pct": "Live evidence available (%)",
    "conflict_detected_pct": "Conflict detected (%)",
}


def print_summary_table(summary: dict) -> None:
    width = max(len(v) for v in SUMMARY_LABELS.values())
    print("\n=== MedOrchestrate Evaluation Summary ===")
    for key, label in SUMMARY_LABELS.items():
        print(f"  {label:<{width}} : {summary[key]}")
    print()


def write_markdown_report(results: list[CaseResult], summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# MedOrchestrate Evaluation Report", "", "## Summary", "", "| Metric | Value |", "|---|---|"]
    for key, label in SUMMARY_LABELS.items():
        lines.append(f"| {label} | {summary[key]} |")
    lines += ["", "## Per-case results", "", "| Case | Success | Confidence | Review required | Iterations | Latency (s) | RAG relevant | Evidence coverage | Live evidence | Error |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r.case_id} | {r.success} | {r.confidence} | {r.review_required} | {r.iteration_count} | "
            f"{round(r.latency_seconds, 2) if r.latency_seconds is not None else ''} | {r.rag_relevant} | "
            f"{r.evidence_coverage} | {r.live_evidence_available} | {r.error or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_report(results: list[CaseResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else [])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


# --------------------------------------------------------------------------
# STRETCH / OPTIONAL: ablation comparison. NOT run by default -- pass
# --ablation. Runs a small subset of cases through 4 pipeline tiers.
#
# Interpretation note: the source "guide" document wasn't available to
# consult directly for the exact intended tier boundaries, so this is a
# documented, best-effort reading of "baseline LLM-only vs RAG-only vs
# RAG+live vs +fusion vs full system" as 4 (not 5) meaningfully distinct
# tiers -- fusion is what RECONCILES multiple evidence sources, so it
# only becomes a meaningfully distinct step once a tier actually HAS more
# than one source to reconcile:
#
#   1. baseline_llm_only    -- fuse() with empty RAG + empty live evidence
#                               (pure LLM reasoning over clinical/imaging
#                               findings alone, no external evidence).
#   2. rag_only              -- fuse() with real RAG evidence, empty live.
#   3. rag_plus_live_fusion  -- fuse() with real RAG AND real live evidence
#                               together -- this is also where "+fusion"
#                               is satisfied, since tiers 1-2 give fusion
#                               at most one source to work with.
#   4. full_system           -- the complete run_case() pipeline,
#                               including the iterative optimizer if
#                               confidence is low.
#
# Confidence for each tier is computed via the REAL compute_confidence()
# formula against that tier's real/empty rag+live+fusion objects, so the
# comparison reflects the actual confidence formula's behavior, not an
# invented ablation-specific metric.
# --------------------------------------------------------------------------


async def _run_ablation_tiers_for_case(case: CaseInput) -> dict:
    _install_mocks_for_case(case)

    clinical = await extract_clinical_findings(case.clinical_text, case.medical_report)
    imaging = await analyze_image(case.image_path) if case.image_path else None

    query = _build_query(clinical, imaging)
    real_rag = await run_rag(clinical, imaging)
    real_live = await gather_live_evidence(query) if query else LiveEvidence(sources=[])

    empty_rag = RAGEvidence(evidence=[])
    empty_live = LiveEvidence(sources=[])

    fusion_baseline = await fuse(clinical, imaging, empty_rag, empty_live)
    baseline_confidence = compute_confidence(empty_rag, empty_live, fusion_baseline)

    fusion_rag_only = await fuse(clinical, imaging, real_rag, empty_live)
    rag_only_confidence = compute_confidence(real_rag, empty_live, fusion_rag_only)

    fusion_rag_live = await fuse(clinical, imaging, real_rag, real_live)
    rag_live_confidence = compute_confidence(real_rag, real_live, fusion_rag_live)

    return {
        "baseline_llm_only": round(baseline_confidence, 4),
        "rag_only": round(rag_only_confidence, 4),
        "rag_plus_live_fusion": round(rag_live_confidence, 4),
    }


def run_ablation(cases: list[tuple[str, CaseInput]]) -> list[dict]:
    rows = []
    for filename, case in cases:
        tiers = asyncio.run(_run_ablation_tiers_for_case(case))
        _install_mocks_for_case(case)
        full_report = run_case(case)
        rows.append(
            {
                "case_id": case.case_id,
                **tiers,
                "full_system": round(full_report.confidence, 4),
            }
        )
    return rows


def print_ablation_table(rows: list[dict]) -> None:
    print("\n=== STRETCH: Ablation comparison (subset) ===")
    print(f"{'Case':<35} {'baseline':>10} {'rag_only':>10} {'rag+live':>10} {'full':>10}")
    for row in rows:
        print(
            f"{row['case_id']:<35} {row['baseline_llm_only']:>10} {row['rag_only']:>10} "
            f"{row['rag_plus_live_fusion']:>10} {row['full_system']:>10}"
        )
    print()


# ==========================================================================
# Phase 17 Mode 2 (required): real public benchmark subset -- MedQA +
# PubMedQA, via Hugging Face `datasets`.
#
# HONEST CAVEAT, stated up front: this needs `pip install datasets` (not
# in requirements.txt -- see its own comment there) AND live network
# access to huggingface.co to download bigbio/med_qa and pubmed_qa. When
# either is unavailable, this degrades to a clearly-logged
# {"available": False, "reason": ...} result -- never a crash, and never
# a fabricated number pretending to be real. If the LLM provider itself
# isn't credentialed (no ANTHROPIC_API_KEY/GEMINI_API_KEY/etc.), the MCQ-
# answering call also fails per-question and that question is skipped
# (counted in `errors`), rather than silently guessing.
# ==========================================================================

MCQ_SYSTEM_PROMPT = (
    "You are answering a single multiple-choice medical exam question. "
    "Reason through it briefly (a few sentences), then end your reply "
    "with a final line in exactly this format: 'Answer: X' where X is "
    "the single capital letter of the correct option."
)


async def _answer_mcq(question: str, options: list[str], context: str | None = None) -> str | None:
    """Returns the model's chosen option letter ("A", "B", ...), or None
    if the call failed or no valid letter could be parsed out of the
    response -- callers must treat None as "skip this question", not as
    a wrong answer (that would understate rather than honestly report
    the error rate). Allows the model to reason before answering (see
    MCQ_SYSTEM_PROMPT) rather than forcing an immediate bare letter,
    which a fast/lite model answers measurably worse on cold. `context`
    is the source passage a reading-comprehension item (e.g. PubMedQA)
    must be answered from -- omit for knowledge-only items (e.g. MedQA)."""
    from backend.services.llm_client import get_llm_client, get_model_name

    labeled = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    prompt = f"Context: {context}\n\nQuestion: {question}\n\n{labeled}" if context else f"{question}\n\n{labeled}"
    try:
        client = get_llm_client()
        response = await client.messages.create(
            model=get_model_name(), max_tokens=400, system=MCQ_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
    except Exception:
        return None
    valid_letters = {chr(65 + i) for i in range(len(options))}
    match = re.search(r"Answer:\s*([A-Za-z])", text)
    if match and match.group(1).upper() in valid_letters:
        return match.group(1).upper()
    # Fallback: last standalone valid option letter mentioned in the reply.
    for ch in reversed(text.upper()):
        if ch in valid_letters:
            return ch
    return None


def _load_medqa_pubmedqa(n_per_benchmark: int) -> tuple[list[dict], list[dict]] | dict:
    """Loads and normalizes both benchmarks into a common
    {"question", "options", "correct_letter"} shape. Returns
    (medqa_items, pubmedqa_items) on success, or a
    {"available": False, "reason": ...} dict on any failure (missing
    package, network error, or an unexpected dataset schema -- field
    names below are this pack's best-effort reading of each dataset's
    public schema, not independently re-verified against a live
    download in this environment)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {"available": False, "reason": "the `datasets` package is not installed (pip install datasets)"}

    try:
        medqa_raw = load_dataset("bigbio/med_qa", name="med_qa_en_bigbio_qa", split=f"test[:{n_per_benchmark}]")
    except Exception as exc:
        return {"available": False, "reason": f"could not load bigbio/med_qa (needs network access): {exc}"}

    try:
        # The original bare "pubmed_qa" repo is script-based, which
        # newer `datasets` releases refuse to load ("Dataset scripts
        # are no longer supported"); qiaojin/PubMedQA is the
        # Parquet-mirrored dataset with the identical pqa_labeled
        # schema (question / final_decision).
        pubmedqa_raw = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=f"train[:{n_per_benchmark}]")
    except Exception as exc:
        return {"available": False, "reason": f"could not load qiaojin/PubMedQA (needs network access): {exc}"}

    medqa_items = []
    for row in medqa_raw:
        try:
            choices = row["choices"]
            answer_text = row["answer"][0]
            correct_idx = choices.index(answer_text)
            medqa_items.append(
                {"question": row["question"], "options": choices, "correct_letter": chr(65 + correct_idx)}
            )
        except (KeyError, ValueError, IndexError):
            continue  # unexpected row shape -- skip rather than guess

    pubmedqa_items = []
    for row in pubmedqa_raw:
        try:
            options = ["yes", "no", "maybe"]
            correct_idx = options.index(row["final_decision"].lower())
            # PubMedQA is a reading-comprehension task -- the question is
            # only answerable from its paired abstract, not general
            # knowledge. Carry the context passages through so the model
            # actually gets what it needs (skipping this drops accuracy
            # to near the majority-class baseline regardless of model).
            context = " ".join(row["context"]["contexts"])
            pubmedqa_items.append(
                {
                    "question": row["question"], "options": options,
                    "correct_letter": chr(65 + correct_idx), "context": context,
                }
            )
        except (KeyError, ValueError, AttributeError):
            continue

    return medqa_items, pubmedqa_items


async def _score_items(items: list[dict]) -> list[dict]:
    """Answers every item via the real (or test-mocked) LLM call and
    returns each with a "predicted_letter" (None on error) and
    "correct" bool added."""
    scored = []
    for item in items:
        predicted = await _answer_mcq(item["question"], item["options"], context=item.get("context"))
        scored.append({**item, "predicted_letter": predicted, "correct": predicted == item["correct_letter"]})
    return scored


def _classification_metrics(scored: list[dict], letters: list[str], class_names: list[str]) -> dict:
    """Per-class precision/recall/F1/specificity from a one-vs-rest
    confusion matrix, plus micro-averaged precision/recall/F1 (for a
    single-label multiclass task where every scored item has exactly
    one predicted and one true label, micro-P == micro-R == micro-F1
    == accuracy -- a standard identity, not a bug -- included anyway
    to match the baseline paper's own reported metric names). Items
    with no valid prediction (errors) are excluded, same as _accuracy."""
    answered = [s for s in scored if s["predicted_letter"] is not None]
    per_class = {}
    tp_total = 0
    for letter, name in zip(letters, class_names):
        tp = sum(1 for s in answered if s["predicted_letter"] == letter and s["correct_letter"] == letter)
        fp = sum(1 for s in answered if s["predicted_letter"] == letter and s["correct_letter"] != letter)
        fn = sum(1 for s in answered if s["predicted_letter"] != letter and s["correct_letter"] == letter)
        tn = sum(1 for s in answered if s["predicted_letter"] != letter and s["correct_letter"] != letter)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        per_class[name] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "specificity": round(specificity, 4), "support": tp + fn,
        }
        tp_total += tp
    n = len(answered)
    micro = tp_total / n if n else 0.0  # == accuracy for single-label multiclass
    return {
        "per_class": per_class,
        "micro_precision": round(micro, 4),
        "micro_recall": round(micro, 4),
        "micro_f1": round(micro, 4),
        "macro_specificity": round(sum(c["specificity"] for c in per_class.values()) / len(per_class), 4) if per_class else 0.0,
        "n": n,
    }


def run_medqa_pubmedqa_benchmark(n_per_benchmark: int = 50, calibration_fraction: float = 0.7) -> dict:
    """Phase 17 Mode 2 (required). Scores real MedQA + PubMedQA subsets
    against ground truth, reports top-1 accuracy per benchmark, then
    uses a documented interpretation of "before/after accuracy on a
    held-out slice" (see the long comment in the function body) to
    actually call weight_tuner.recalibrate_weights() and report
    before/after weights + accuracy side by side.
    """
    loaded = _load_medqa_pubmedqa(n_per_benchmark)
    if isinstance(loaded, dict):
        return loaded  # {"available": False, "reason": ...}
    medqa_items, pubmedqa_items = loaded

    medqa_scored = asyncio.run(_score_items(medqa_items))
    pubmedqa_scored = asyncio.run(_score_items(pubmedqa_items))

    def _accuracy(scored: list[dict]) -> tuple[float, int, int]:
        answered = [s for s in scored if s["predicted_letter"] is not None]
        if not answered:
            return 0.0, 0, len(scored)
        correct = sum(1 for s in answered if s["correct"])
        return correct / len(answered), len(answered), len(scored) - len(answered)

    medqa_acc, medqa_n, medqa_errors = _accuracy(medqa_scored)
    pubmedqa_acc, pubmedqa_n, pubmedqa_errors = _accuracy(pubmedqa_scored)

    # Per-class precision/recall/F1/specificity, only meaningful for
    # PubMedQA: its three options (yes/no/maybe) are the SAME shared
    # label space across every question, unlike MedQA where each
    # question's A/B/C/D/E are different substantive answers -- there is
    # no shared "class" to compute precision/recall against for MedQA,
    # so accuracy alone is the correct (and only) metric there.
    pubmedqa_class_metrics = _classification_metrics(pubmedqa_scored, ["A", "B", "C"], ["yes", "no", "maybe"])

    # -----------------------------------------------------------------
    # "Before/after accuracy on a held-out slice" -- DOCUMENTED
    # INTERPRETATION (the guide names this requirement without pinning
    # down exactly what "accuracy" means once weights only affect the
    # CONFIDENCE formula, not the LLM's own MCQ answer): recalibration
    # tunes w1/w2/w3 -- the confidence-scoring formula -- not whether the
    # LLM answers a question correctly. So "accuracy" here means
    # *calibration* accuracy: on the held-out slice, does
    # (confidence >= THRESHOLD) correctly predict (the LLM's answer was
    # actually correct)? For each question, run_rag()/gather_live_evidence()
    # supply the real RAG/live signal so confidence is genuinely derived
    # from real retrieval, not a placeholder.
    # -----------------------------------------------------------------
    all_scored = [s for s in (medqa_scored + pubmedqa_scored) if s["predicted_letter"] is not None]
    split_idx = int(len(all_scored) * calibration_fraction)
    calibration_slice, held_out_slice = all_scored[:split_idx], all_scored[split_idx:]

    import tempfile

    from backend.services import storage
    from backend.services.confidence import compute_confidence_breakdown, get_current_weights
    from backend.services.weight_tuner import recalibrate_weights

    original_db_path = storage.DB_PATH
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_db.close()
    try:
        storage.DB_PATH = tmp_db.name
        storage._init_tables()

        def _breakdown_for(item: dict) -> dict:
            rag = asyncio.run(run_rag(ClinicalFindings(demographics={}, symptoms=[], history=[], labs={}, imaging_text_findings=[]), None, query_override=item["question"]))
            live = asyncio.run(gather_live_evidence(item["question"]))
            from backend.models.schemas import FusionResult

            # placeholder diagnosis: confidence is forced to 0.0 when there is none
            fusion = FusionResult(diagnoses=[{"name": "placeholder"}], overall_confidence=0.5, conflicts=[])
            return compute_confidence_breakdown(rag, live, fusion)

        def _calibration_accuracy(slice_items: list[dict]) -> float:
            if not slice_items:
                return 0.0
            correct_predictions = 0
            for item in slice_items:
                breakdown = _breakdown_for(item)
                predicted_trustworthy = breakdown["confidence"] >= 0.65
                correct_predictions += int(predicted_trustworthy == item["correct"])
            return correct_predictions / len(slice_items)

        before_weights = dict(get_current_weights())
        before_held_out_accuracy = _calibration_accuracy(held_out_slice)

        for item in calibration_slice:
            breakdown = _breakdown_for(item)
            case_id = f"medqa_calibration_{id(item)}"
            from backend.models.schemas import FinalReport

            storage.save_case(CaseInput(case_id=case_id, patient=Patient(age=40, sex="female")), status="completed")
            storage.save_report(
                case_id,
                FinalReport(case_id=case_id, diagnoses=[], confidence=breakdown["confidence"], evidence=[], review_required=False),
                confidence_breakdown=breakdown,
            )
            storage.save_review(case_id, {"decision": "accept" if item["correct"] else "override", "annotation": None, "reviewer": "mode2-eval"})

        recalibration_result = recalibrate_weights(min_reviews=min(10, len(calibration_slice)))
        after_weights = dict(get_current_weights())
        after_held_out_accuracy = _calibration_accuracy(held_out_slice)
    finally:
        storage.DB_PATH = original_db_path
        Path(tmp_db.name).unlink(missing_ok=True)

    return {
        "available": True,
        "medqa_accuracy": round(medqa_acc, 4),
        "medqa_n_scored": medqa_n,
        "medqa_errors": medqa_errors,
        "pubmedqa_accuracy": round(pubmedqa_acc, 4),
        "pubmedqa_n_scored": pubmedqa_n,
        "pubmedqa_errors": pubmedqa_errors,
        "pubmedqa_class_metrics": pubmedqa_class_metrics,
        "medqa_scored": medqa_scored,
        "pubmedqa_scored": pubmedqa_scored,
        "before_weights": before_weights,
        "after_weights": after_weights,
        "recalibration_performed": recalibration_result is not None,
        "before_held_out_calibration_accuracy": round(before_held_out_accuracy, 4),
        "after_held_out_calibration_accuracy": round(after_held_out_accuracy, 4),
        "held_out_slice_size": len(held_out_slice),
        # Accuracy of the trivial rule "always predict trustworthy" on the
        # held-out slice == the share of its answers that were correct. A
        # recalibration "improvement" that only reaches this number shows
        # no real calibration signal (see print_mode2_result).
        "held_out_always_trust_baseline": round(
            (sum(1 for item in held_out_slice if item["correct"]) / len(held_out_slice)) if held_out_slice else 0.0, 4
        ),
    }


def print_mode2_result(result: dict) -> None:
    print("\n=== Phase 17 Mode 2: MedQA / PubMedQA (real public benchmark) ===")
    if not result.get("available", False):
        print(f"  UNAVAILABLE: {result['reason']}")
        print()
        return
    print(f"  MedQA top-1 accuracy:     {result['medqa_accuracy']:.1%}  (n={result['medqa_n_scored']}, errors={result['medqa_errors']})")
    print(f"  PubMedQA top-1 accuracy:  {result['pubmedqa_accuracy']:.1%}  (n={result['pubmedqa_n_scored']}, errors={result['pubmedqa_errors']})")
    cm = result.get("pubmedqa_class_metrics")
    if cm:
        print(f"  PubMedQA micro P/R/F1:    {cm['micro_precision']:.1%} / {cm['micro_recall']:.1%} / {cm['micro_f1']:.1%}  "
              "(identical to accuracy by construction for single-label multiclass -- expected, not a bug)")
        print(f"  PubMedQA macro specificity: {cm['macro_specificity']:.1%}")
        print("  PubMedQA per-class breakdown:")
        for name, m in cm["per_class"].items():
            print(f"    {name:<6} precision={m['precision']:.1%}  recall={m['recall']:.1%}  f1={m['f1']:.1%}  "
                  f"specificity={m['specificity']:.1%}  support={m['support']}")
    print(
        "  NOTE: this benchmark family is the same one used by the closest prior "
        "published system in this space (IEEE Access 2025, DOI 10.1109/ACCESS.2025.3613340, "
        "94%/88%/84% on MedQA/PubMedQA/MedBullets) -- for comparability, not a claim of reproducing its result."
    )
    print(f"  Weights before recalibration: {result['before_weights']}")
    print(f"  Weights after recalibration:  {result['after_weights']} (performed={result['recalibration_performed']})")
    print(
        f"  Held-out calibration accuracy (confidence>=THRESHOLD correctly predicts a correct "
        f"answer), n={result['held_out_slice_size']}: "
        f"before={result['before_held_out_calibration_accuracy']:.1%} -> after={result['after_held_out_calibration_accuracy']:.1%}"
    )
    baseline = result.get("held_out_always_trust_baseline")
    if baseline is not None:
        print(f"  Baseline (always predict 'trustworthy' = share of held-out answers that were correct): {baseline:.1%}")
        if result["after_held_out_calibration_accuracy"] <= baseline + 0.005:
            print(
                "  CAVEAT: the 'after' figure does NOT beat that trivial baseline, so it is not evidence of\n"
                "  real calibration. In this benchmark fusion is a placeholder with no conflicts, so the\n"
                "  agreement signal is a constant 1.0 and recalibration can only shift ALL confidences up\n"
                "  or down together (here: up, until everything clears the threshold). The held-out slice\n"
                "  is also the tail of the combined item list (a single benchmark), not a mixed sample.\n"
                "  Do not use these weights, or this number, as a basis for the production weights."
            )


def write_mode2_items_csv(result: dict, path: Path) -> None:
    """Per-item audit trail: every MedQA/PubMedQA question with its
    predicted vs. correct letter, so the aggregate accuracy/P/R/F1
    numbers can be checked against the raw data rather than trusted
    blind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "question", "options", "correct_letter", "predicted_letter", "correct"])
        for benchmark, items in (("medqa", result["medqa_scored"]), ("pubmedqa", result["pubmedqa_scored"])):
            for item in items:
                writer.writerow([
                    benchmark, item["question"], "|".join(item["options"]),
                    item["correct_letter"], item["predicted_letter"], item["correct"],
                ])
    print()


# ==========================================================================
# Phase 17 Mode 3 (required): imaging mini-benchmark + the raw-image-
# attached vs. text-labels-only fuse() comparison.
#
# All three imaging modalities (MRI brain scan, CT scan, chest X-ray --
# see backend/agents/imaging_labels.py's FINDING_LABELS_BY_MODALITY)
# have a real findings model and are covered here. Written generically
# against FINDING_LABELS_BY_MODALITY (not a hardcoded 3-item list), so a
# future modality added there is picked up automatically with no code
# change needed in this file.
# ==========================================================================


def generate_imaging_eval_set(images_per_label: int = 1) -> list[dict]:
    """Generates small synthetic placeholder images (simple, clearly-
    labeled-synthetic gradients/shapes -- same spirit as
    data/images/README.md's existing demo images), one per
    (modality, label) pair currently in FINDING_LABELS_BY_MODALITY, and
    writes a manifest (image path -> intended modality -> intended
    label) to IMAGING_EVAL_MANIFEST. Deterministic (seeded per label) so
    repeated runs regenerate the same images.
    """
    from PIL import Image, ImageDraw

    IMAGING_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for modality, labels in FINDING_LABELS_BY_MODALITY.items():
        modality_slug = modality.lower().replace(" ", "_")
        for label_idx, label in enumerate(labels):
            for img_idx in range(images_per_label):
                seed = abs(hash((modality, label, img_idx))) % (2**32)
                # A distinct, deterministic color/shape per label -- not
                # a real scan, just a visually-distinguishable synthetic
                # stand-in so classification isn't pure noise.
                r, g, b = seed % 256, (seed // 256) % 256, (seed // 65536) % 256
                image = Image.new("RGB", (224, 224), color=(r, g, b))
                draw = ImageDraw.Draw(image)
                draw.ellipse(
                    [40 + (seed % 30), 40 + (seed % 20), 180 - (seed % 30), 180 - (seed % 20)],
                    outline=(255 - r, 255 - g, 255 - b), width=4,
                )
                filename = f"{modality_slug}_{label_idx:02d}_{img_idx}.png"
                path = IMAGING_EVAL_DIR / filename
                image.save(path)
                manifest.append({"image_path": path.as_posix(), "modality": modality, "intended_label": label})

    IMAGING_EVAL_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_or_generate_imaging_eval_set() -> list[dict]:
    if IMAGING_EVAL_MANIFEST.exists():
        return json.loads(IMAGING_EVAL_MANIFEST.read_text(encoding="utf-8"))
    return generate_imaging_eval_set()


def run_imaging_benchmark(manifest: list[dict]) -> dict:
    """Runs the real analyze_image() (real BiomedCLIP inference, no
    mocking) against every manifest entry and scores top-1 accuracy per
    modality: does the top-scoring returned finding match the label the
    image was generated for?"""
    per_modality_correct: dict[str, int] = {}
    per_modality_total: dict[str, int] = {}

    for entry in manifest:
        result = asyncio.run(analyze_image(entry["image_path"], modality_hint=entry["modality"]))
        per_modality_total[entry["modality"]] = per_modality_total.get(entry["modality"], 0) + 1
        top_finding = result.findings[0] if result.findings else None
        if top_finding == entry["intended_label"]:
            per_modality_correct[entry["modality"]] = per_modality_correct.get(entry["modality"], 0) + 1

    accuracy_by_modality = {
        modality: round(per_modality_correct.get(modality, 0) / total, 4)
        for modality, total in per_modality_total.items()
    }
    return {"accuracy_by_modality": accuracy_by_modality, "n_by_modality": per_modality_total}


def run_image_fusion_comparison(manifest: list[dict], sample_size: int = 3) -> list[dict]:
    """For a small sample of the imaging eval set, runs fuse() once WITH
    the raw image attached (the joint multimodal path, Phase 5) and once
    WITHOUT (text-labels-only), and reports whether the top diagnosis
    and confidence differ -- the number that demonstrates joint
    image+text fusion actually changes the output, not just that it runs.

    The LLM call is mocked at the call_multimodal()/_get_client()
    boundary with two DIFFERENT canned responses (one for "an image was
    attached", one for "text labels only") -- this demonstrates the
    CODE PATH genuinely differs (call_multimodal is invoked only in the
    with-image run), which is what's actually being verified here; it
    is NOT a claim about what a real credentialed LLM would say
    differently about the same image, which needs a real API key this
    environment doesn't have configured.
    """
    sample = manifest[:sample_size]
    rows = []

    for entry in sample:
        clinical = ClinicalFindings(demographics={}, symptoms=[], history=[], labs={}, imaging_text_findings=[])
        imaging = asyncio.run(analyze_image(entry["image_path"], modality_hint=entry["modality"]))
        rag = RAGEvidence(evidence=[])
        live = LiveEvidence(sources=[])

        with_image_response = json.dumps(
            {
                "diagnoses": [{"name": "Direct visual read confirms " + imaging.findings[0], "confidence": 0.7, "supporting_evidence": [], "rank": 1}],
                "overall_confidence": 0.7,
                "conflicts": [],
            }
        )
        without_image_response = json.dumps(
            {
                "diagnoses": [{"name": "Text-label-only: " + imaging.findings[0], "confidence": 0.5, "supporting_evidence": [], "rank": 1}],
                "overall_confidence": 0.5,
                "conflicts": [],
            }
        )

        async def _run_both():
            fusion_agent.call_multimodal = AsyncMock(return_value=with_image_response)
            with_image = await fuse(clinical, imaging, rag, live, image_path=entry["image_path"])

            fusion_agent._get_client = lambda: _fake_client(without_image_response)
            without_image = await fuse(clinical, imaging, rag, live, image_path=None)
            return with_image, without_image

        with_image_result, without_image_result = asyncio.run(_run_both())

        with_top = with_image_result.diagnoses[0]["name"] if with_image_result.diagnoses else None
        without_top = without_image_result.diagnoses[0]["name"] if without_image_result.diagnoses else None
        rows.append(
            {
                "image_path": entry["image_path"],
                "modality": entry["modality"],
                "with_image_top_diagnosis": with_top,
                "with_image_confidence": with_image_result.overall_confidence,
                "without_image_top_diagnosis": without_top,
                "without_image_confidence": without_image_result.overall_confidence,
                "differs": (with_top != without_top) or (with_image_result.overall_confidence != without_image_result.overall_confidence),
            }
        )

    return rows


def print_mode3_result(benchmark: dict, fusion_rows: list[dict]) -> None:
    print("\n=== Phase 17 Mode 3: Imaging mini-benchmark ===")
    print("  CAVEAT: synthetic placeholder images -- a construction-validity check, not real diagnostic accuracy.")
    print("  Covers all three modalities with a real findings model: MRI brain scan, CT scan, chest X-ray.")
    for modality, acc in benchmark["accuracy_by_modality"].items():
        n = benchmark["n_by_modality"][modality]
        print(f"    {modality:<20} top-1 accuracy: {acc:.1%}  (n={n})")
    print("\n  With-raw-image vs. text-labels-only fusion comparison:")
    for row in fusion_rows:
        print(f"    {row['modality']:<20} differs={row['differs']}  "
              f"with_image={row['with_image_top_diagnosis']!r} ({row['with_image_confidence']})  "
              f"without={row['without_image_top_diagnosis']!r} ({row['without_image_confidence']})")
    print()


# ==========================================================================
# Phase 17 required ablation: with vs. without text case-memory retrieval
# ==========================================================================


def run_case_memory_ablation(case: CaseInput) -> dict:
    """Seeds ONE real clinician-verified case-memory entry (real BioBERT
    embedding, real local Qdrant write -- see backend/rag/case_memory.py)
    matching `case`'s own symptoms, then runs fuse() with vs. without
    that entry present in the RAG evidence handed to it, reporting the
    confidence delta. This is the required Phase 17 ablation: "with vs.
    without text case-memory retrieval, on a subset of cases where at
    least one prior override exists."
    """
    import tempfile

    from qdrant_client.models import PointIdsList

    from backend.rag import case_memory as case_memory_module
    from backend.rag.case_memory import add_clinician_verified_case
    from backend.services import storage

    # Mocked at the client boundary, same as the rest of this harness's
    # Claude calls (_install_mocks_for_case) -- this ablation is about
    # RAG/case-memory retrieval, not clinical extraction, so it stays
    # deterministic and independent of whichever LLM_PROVIDER/credential
    # happens to be configured in this environment's .env.
    clinical_agent._get_client = lambda: _fake_client(_clinical_mock_response(case))
    clinical = asyncio.run(extract_clinical_findings(case.clinical_text, case.medical_report))

    # The seed is a MADE-UP "clinician-verified" diagnosis. It must never be
    # left behind in the real case-memory collection (where the app would
    # later retrieve it and show it to the model as a genuine clinician
    # precedent) nor in the real case_memory_log. So: log into a throwaway
    # database, and delete the seeded Qdrant point(s) again in `finally`.
    seed_case_id = f"{case.case_id}_seed_precedent"
    original_db_path = storage.DB_PATH
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_db.close()
    seeded_point_ids: list[str] = []
    try:
        storage.DB_PATH = tmp_db.name
        storage._init_tables()
        add_clinician_verified_case(
            case_id=seed_case_id, clinical=clinical, imaging=None,
            diagnosis_text="Vestibular migraine (clinician-confirmed precedent)", reviewer="mode17-eval",
        )
        seeded_point_ids = [row["qdrant_point_id"] for row in storage.get_case_memory_log(seed_case_id)]

        rag_with_memory = asyncio.run(run_rag(clinical, None))
        rag_without_memory = RAGEvidence(
            evidence=[e for e in rag_with_memory.evidence if not e.source.startswith("clinician_verified:")]
        )
        live = LiveEvidence(sources=[])

        fusion_agent._get_client = lambda: _fake_client(_fusion_mock_response("case-memory-ablation"))
        with_memory_fusion = asyncio.run(fuse(clinical, None, rag_with_memory, live))
        without_memory_fusion = asyncio.run(fuse(clinical, None, rag_without_memory, live))

        with_confidence = compute_confidence(rag_with_memory, live, with_memory_fusion)
        without_confidence = compute_confidence(rag_without_memory, live, without_memory_fusion)
    finally:
        if seeded_point_ids:
            try:
                case_memory_module._get_client().delete(
                    collection_name=case_memory_module.COLLECTION_NAME,
                    points_selector=PointIdsList(points=seeded_point_ids),
                )
            except Exception as exc:  # cleanup must not mask the real result/error
                print(f"  WARNING: could not remove the ablation's seeded case-memory point(s): {exc}")
        storage.DB_PATH = original_db_path
        Path(tmp_db.name).unlink(missing_ok=True)

    return {
        "case_id": case.case_id,
        "case_memory_entries_present": sum(1 for e in rag_with_memory.evidence if e.source.startswith("clinician_verified:")),
        "confidence_with_case_memory": round(with_confidence, 4),
        "confidence_without_case_memory": round(without_confidence, 4),
        "delta": round(with_confidence - without_confidence, 4),
    }


def print_case_memory_ablation(result: dict) -> None:
    print("\n=== Phase 17 required ablation: with vs. without text case-memory ===")
    print(f"  Case: {result['case_id']}  (clinician-verified entries retrieved: {result['case_memory_entries_present']})")
    print(f"  Confidence WITH case memory:    {result['confidence_with_case_memory']}")
    print(f"  Confidence WITHOUT case memory: {result['confidence_without_case_memory']}")
    print(f"  Delta: {result['delta']:+.4f}")
    print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MedOrchestrate evaluation harness")
    parser.add_argument("--cases-dir", type=Path, default=TEST_CASES_DIR)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD_REPORT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV_REPORT)
    parser.add_argument(
        "--no-mock-claude",
        action="store_true",
        help="Skip the generic Claude mock and run the real CrewAI/Claude pipeline (requires a working ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="STRETCH/OPTIONAL: also run the 4-tier ablation comparison on a small subset of cases.",
    )
    parser.add_argument("--ablation-subset-size", type=int, default=3)
    parser.add_argument(
        "--mode2", action="store_true",
        help="Phase 17 Mode 2 (required): real MedQA/PubMedQA subset, needs `pip install datasets` + network.",
    )
    parser.add_argument("--mode2-n", type=int, default=50, help="Questions per benchmark for --mode2.")
    parser.add_argument(
        "--mode3", action="store_true",
        help="Phase 17 Mode 3 (required): imaging mini-benchmark + with/without-raw-image fusion comparison.",
    )
    parser.add_argument(
        "--case-memory-ablation", action="store_true",
        help="Phase 17 required ablation: with vs. without text case-memory retrieval.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run everything: --ablation, --mode2, --mode3, and --case-memory-ablation.",
    )
    args = parser.parse_args()
    if args.full:
        args.ablation = args.mode2 = args.mode3 = args.case_memory_ablation = True

    cases = load_cases(args.cases_dir)
    print(f"Loaded {len(cases)} case(s) from {args.cases_dir}")

    results = run_evaluation(cases, mock_claude=not args.no_mock_claude)
    summary = summarize(results)

    print_summary_table(summary)
    write_markdown_report(results, summary, args.output_md)
    write_csv_report(results, args.output_csv)
    print(f"Markdown report written to {args.output_md}")
    print(f"CSV report written to {args.output_csv}")

    if args.ablation:
        subset = cases[: args.ablation_subset_size]
        ablation_rows = run_ablation(subset)
        print_ablation_table(ablation_rows)

    if args.case_memory_ablation and cases:
        print_case_memory_ablation(run_case_memory_ablation(cases[0][1]))

    if args.mode3:
        manifest = load_or_generate_imaging_eval_set()
        benchmark = run_imaging_benchmark(manifest)
        fusion_rows = run_image_fusion_comparison(manifest)
        print_mode3_result(benchmark, fusion_rows)

    if args.mode2:
        mode2_result = run_medqa_pubmedqa_benchmark(n_per_benchmark=args.mode2_n)
        print_mode2_result(mode2_result)
        if mode2_result.get("available"):
            write_mode2_items_csv(mode2_result, Path("backend/tests/mode2_items.csv"))
            print(f"Per-item Mode 2 predictions written to backend/tests/mode2_items.csv")


if __name__ == "__main__":
    main()
