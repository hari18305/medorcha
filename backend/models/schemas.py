"""Shared JSON contracts for MedOrchestrate.

Every agent and API route imports its data shapes from here rather than
redefining them, so the pipeline stays consistent end to end.
"""

from pydantic import BaseModel


class Patient(BaseModel):
    age: int
    sex: str


class CaseInput(BaseModel):
    case_id: str
    patient: Patient
    clinical_text: str | None = None
    medical_report: str | None = None
    image_path: str | None = None
    labs: dict = {}
    history: list[str] = []


class ClinicalFindings(BaseModel):
    demographics: dict
    symptoms: list[str]
    history: list[str]
    labs: dict
    imaging_text_findings: list[str]


class ImagingFindings(BaseModel):
    modality: str
    findings: list[str]
    confidence: float
    # True when the detected/hinted modality has no findings model
    # configured (see imaging_labels.FINDING_LABELS_BY_MODALITY) -- the
    # image was NOT classified against an unrelated modality's labels.
    # Downstream consumers (rag_agent, fusion_agent) must check this
    # rather than trusting `findings` at face value when it's True.
    unsupported_modality: bool = False


class EvidenceItem(BaseModel):
    text: str
    source: str
    score: float


class RAGEvidence(BaseModel):
    evidence: list[EvidenceItem]


class LiveEvidenceSource(BaseModel):
    title: str
    source: str
    url: str
    summary: str
    evidence_level: str
    publication_date: str | None = None


class LiveEvidence(BaseModel):
    sources: list[LiveEvidenceSource]
    # Additive, defaulted bookkeeping so "every source failed" can be told
    # apart from "sources answered but found nothing" (an empty `sources`
    # list alone can't). sources_queried == 0 means no live search was
    # attempted at all (e.g. an empty query).
    sources_queried: int = 0
    failed_sources: list[str] = []

    @property
    def all_sources_failed(self) -> bool:
        return self.sources_queried > 0 and len(self.failed_sources) >= self.sources_queried


class FusionResult(BaseModel):
    diagnoses: list[dict]
    overall_confidence: float
    conflicts: list[str]


class FinalReport(BaseModel):
    case_id: str
    diagnoses: list[dict]
    confidence: float
    evidence: list[dict]
    review_required: bool
    conflicts: list[str] = []
    iteration_count: int = 0
    live_evidence_available: bool = True
    # The BiomedCLIP imaging findings for this case, echoed so the clinician
    # report can show them (and render an unsupported modality honestly as
    # "not assessed"). None when the case had no image or imaging analysis
    # failed -- and for reports stored before this field existed.
    imaging: ImagingFindings | None = None
