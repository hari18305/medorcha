"""A real MedOrchestrate backend with every language-model, retrieval-model and network call replaced by a canned fake.

    python scripts/e2e_backend.py            # serves http://127.0.0.1:8000 with a throwaway SQLite file

Used by the browser end-to-end tests (frontend/e2e). What is REAL here: the FastAPI app, the async job registry and its progress
events, the orchestrator, the confidence formula and both input-quality rules, the optimizer loop, the review endpoints and SQLite
storage. What is FAKE: the clinical extractor (splits the text on commas), BiomedCLIP (always "chest X-ray / pneumothorax"),
retrieval and live evidence (canned items), the fusion model (a fixed differential that cites [RAG1] and [LIVE1]), the optimizer's
strategy choice, and the case-memory write on override. So it needs no API key, no model download and no network, and it tests the
boundary between the browser and the backend, not the quality of any model.

Text containing the word "ambiguous" gets weak evidence and flagged conflicts, so the real formula lands below the threshold and the
real optimizer loop runs its three rounds.
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = Path(os.getenv("E2E_DB", Path(tempfile.gettempdir()) / "medorchestrate_e2e.db"))
DB.unlink(missing_ok=True)
os.environ["DATABASE_URL"] = f"sqlite:///{DB.as_posix()}"
os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("GEMINI_API_KEY", "e2e-not-used")

import uvicorn  # noqa: E402

import backend.agents.optimizer_agent as optimizer_agent  # noqa: E402
import backend.api.routes as routes  # noqa: E402
import backend.services.orchestrator as orchestrator  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models.schemas import (  # noqa: E402
    ClinicalFindings,
    EvidenceItem,
    FusionResult,
    ImagingFindings,
    LiveEvidence,
    LiveEvidenceSource,
    RAGEvidence,
)

STEP_SECONDS = 0.45  # long enough for a browser to see each step


def _ambiguous(*parts) -> bool:
    return any("ambiguous" in str(p).lower() for p in parts if p)


async def fake_extract(clinical_text, medical_report=None):
    symptoms = [s.strip() for s in (clinical_text or "").replace(" and ", ",").split(",") if s.strip()]
    findings = [medical_report.strip()[:80]] if medical_report and medical_report.strip() else []
    await asyncio.sleep(STEP_SECONDS)
    return ClinicalFindings(demographics={}, symptoms=symptoms, history=[], labs={}, imaging_text_findings=findings)


async def fake_analyze(image_path):
    await asyncio.sleep(STEP_SECONDS)
    return ImagingFindings(modality="chest X-ray", findings=["pneumothorax"], confidence=0.9)


async def fake_rag(clinical, imaging=None, query_override=None, image_path=None, **_):
    await asyncio.sleep(STEP_SECONDS)
    weak = _ambiguous(query_override, *(clinical.symptoms if clinical else []))
    score = 0.3 if weak else 0.85
    return RAGEvidence(evidence=[
        EvidenceItem(text=f"Passage {i} on the presenting problem: severe headache can accompany very high blood pressure.",
                     source=f"MedlinePlus: Topic {i}", score=score) for i in range(1, 6)
    ])


async def fake_live(query, *_, **__):
    await asyncio.sleep(STEP_SECONDS)
    weak = _ambiguous(query)
    level = "moderate" if weak else "high"
    return LiveEvidence(sources=[
        LiveEvidenceSource(title=f"Trial of blood pressure control {i}", source="PubMed", url=f"https://example.org/{i}",
                           summary=f"Abstract {i}.", evidence_level=level, publication_date="2024") for i in range(1, 4)
    ], sources_queried=4)


async def fake_fuse(clinical, imaging, rag, live, image_path=None, **_):
    await asyncio.sleep(STEP_SECONDS)
    weak = _ambiguous(*(clinical.symptoms if clinical else []), *(live.sources[0].evidence_level == "moderate" and ["ambiguous"] or []))
    name = "Pneumothorax" if imaging is not None and not imaging.unsupported_modality else "Hypertension-related headache"
    conflicts = [f"Source {i} disagrees with the knowledge base on the first-line approach." for i in range(1, 5)] if weak else []
    return FusionResult(
        diagnoses=[
            {"name": name, "confidence": 0.7, "rank": 1, "supporting_evidence": ["The picture fits [RAG1] and is supported by trials [LIVE1]."]},
            {"name": "Tension-type headache", "confidence": 0.4, "rank": 2, "supporting_evidence": ["A common alternative [RAG2]."]},
        ],
        overall_confidence=0.7, conflicts=conflicts,
    )


async def fake_strategy(*_, **__):
    await asyncio.sleep(STEP_SECONDS)
    return {"strategy": "expand", "query": "ambiguous expanded query", "rationale": "end-to-end test"}


def _direct_execute(agent, task):
    return agent.tools[0]._run()  # skip CrewAI's own LLM call; run the step's real tool


for module in (orchestrator,):
    module.extract_clinical_findings = fake_extract
    module.analyze_image = fake_analyze
    module.run_rag = fake_rag
    module.gather_live_evidence = fake_live
    module.fuse = fake_fuse
    module._execute = _direct_execute
optimizer_agent.run_rag = fake_rag
optimizer_agent.gather_live_evidence = fake_live
optimizer_agent.fuse = fake_fuse
optimizer_agent._select_refinement_strategy = fake_strategy
routes.extract_clinical_findings = fake_extract
routes._write_case_memory_for_override = lambda *a, **k: None

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("E2E_PORT", "8000")), log_level="warning")
