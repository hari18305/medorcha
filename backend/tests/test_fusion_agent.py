"""Tests for backend.agents.fusion_agent.

The Anthropic API call is mocked via the mock_anthropic_client fixture
(same pattern as test_clinical_agent.py), so this suite runs without
network access. Covers: text-only input, text+image input, a case with
real evidence conflicts, empty-evidence input, citation grounding
(fabricated citations get dropped), and the retry-on-invalid-JSON path.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.agents.fusion_agent as fusion_agent
from backend.agents.fusion_agent import FusionError, _format_imaging, _should_attach_image, fuse
from backend.models.schemas import (
    ClinicalFindings,
    EvidenceItem,
    FusionResult,
    ImagingFindings,
    LiveEvidence,
    LiveEvidenceSource,
    RAGEvidence,
)


class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


@pytest.fixture
def mock_anthropic_client(monkeypatch):
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock()
    monkeypatch.setattr(fusion_agent, "_get_client", lambda: fake_client)
    return fake_client


def _canned(payload: dict) -> _FakeResponse:
    return _FakeResponse(json.dumps(payload))


CLINICAL = ClinicalFindings(
    demographics={},
    symptoms=["headache", "dizziness"],
    history=["hypertension"],
    labs={},
    imaging_text_findings=[],
)
IMAGING = ImagingFindings(
    modality="MRI brain scan", findings=["white matter hyperintensity"], confidence=0.95
)
RAG = RAGEvidence(
    evidence=[
        EvidenceItem(text="White matter hyperintensities are linked to chronic hypertension.", source="synthetic", score=0.91),
        EvidenceItem(text="Hypertensive emergency can present with severe headache and dizziness.", source="synthetic", score=0.4),
    ]
)
LIVE = LiveEvidence(
    sources=[
        LiveEvidenceSource(
            title="Vestibular migraine overview",
            source="PubMed",
            url="https://pubmed.ncbi.nlm.nih.gov/1/",
            summary="Migraine is a common cause of episodic dizziness and headache.",
            evidence_level="moderate",
            publication_date="2020",
        )
    ]
)
EMPTY_RAG = RAGEvidence(evidence=[])
EMPTY_LIVE = LiveEvidence(sources=[])


async def test_text_only_input(mock_anthropic_client):
    canned = {
        "diagnoses": [
            {
                "name": "Chronic small vessel disease related to hypertension",
                "confidence": 0.6,
                "supporting_evidence": ["[RAG1] White matter hyperintensities are linked to chronic hypertension."],
                "rank": 1,
            }
        ],
        "overall_confidence": 0.55,
        "conflicts": [],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await fuse(CLINICAL, None, RAG, EMPTY_LIVE)

    assert isinstance(result, FusionResult)
    assert len(result.diagnoses) == 1
    assert result.diagnoses[0]["name"] == "Chronic small vessel disease related to hypertension"
    assert result.diagnoses[0]["supporting_evidence"] == [
        "[RAG1] White matter hyperintensities are linked to chronic hypertension."
    ]
    assert result.overall_confidence == 0.55
    assert result.conflicts == []

    # imaging section should say "no imaging" since imaging=None
    _, kwargs = mock_anthropic_client.messages.create.call_args
    sent = kwargs["messages"][0]["content"]
    assert "no imaging performed" in sent


async def test_text_and_image_input(mock_anthropic_client):
    canned = {
        "diagnoses": [
            {
                "name": "Hypertensive cerebral small vessel disease",
                "confidence": 0.7,
                "supporting_evidence": [
                    "[RAG1] White matter hyperintensities are linked to chronic hypertension.",
                ],
                "rank": 1,
            },
            {
                "name": "Vestibular migraine",
                "confidence": 0.3,
                "supporting_evidence": ["[LIVE1] Migraine is a common cause of episodic dizziness and headache."],
                "rank": 2,
            },
        ],
        "overall_confidence": 0.6,
        "conflicts": [],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await fuse(CLINICAL, IMAGING, RAG, LIVE)

    assert len(result.diagnoses) == 2
    assert result.diagnoses[0]["rank"] == 1
    assert result.diagnoses[1]["supporting_evidence"][0].startswith("[LIVE1]")

    _, kwargs = mock_anthropic_client.messages.create.call_args
    sent = kwargs["messages"][0]["content"]
    assert "MRI brain scan" in sent
    assert "white matter hyperintensity" in sent


async def test_conflict_case_propagates_conflicts(mock_anthropic_client):
    # RAG evidence points toward a vascular/hypertensive cause; live
    # evidence points toward migraine -- a real disagreement the model
    # should flag.
    canned = {
        "diagnoses": [
            {
                "name": "Hypertensive small vessel disease",
                "confidence": 0.5,
                "supporting_evidence": ["[RAG1] White matter hyperintensities are linked to chronic hypertension."],
                "rank": 1,
            },
            {
                "name": "Vestibular migraine",
                "confidence": 0.45,
                "supporting_evidence": ["[LIVE1] Migraine is a common cause of episodic dizziness and headache."],
                "rank": 2,
            },
        ],
        "overall_confidence": 0.5,
        "conflicts": [
            "Offline evidence points toward a vascular/hypertensive cause (RAG1) while live evidence "
            "emphasizes a migrainous cause (LIVE1) for the same symptoms."
        ],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await fuse(CLINICAL, IMAGING, RAG, LIVE)

    assert len(result.conflicts) == 1
    assert "vascular" in result.conflicts[0].lower() or "migrain" in result.conflicts[0].lower()


async def test_empty_evidence_input_short_circuits_without_calling_model(mock_anthropic_client):
    empty_clinical = ClinicalFindings(
        demographics={}, symptoms=[], history=[], labs={}, imaging_text_findings=[]
    )

    result = await fuse(empty_clinical, None, EMPTY_RAG, EMPTY_LIVE)

    assert result == FusionResult(diagnoses=[], overall_confidence=0.0, conflicts=[])
    mock_anthropic_client.messages.create.assert_not_called()


async def test_no_rag_or_live_evidence_but_clinical_present_still_calls_model(mock_anthropic_client):
    canned = {
        "diagnoses": [
            {"name": "Tension-type headache", "confidence": 0.3, "supporting_evidence": [], "rank": 1}
        ],
        "overall_confidence": 0.25,
        "conflicts": [],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await fuse(CLINICAL, None, EMPTY_RAG, EMPTY_LIVE)

    mock_anthropic_client.messages.create.assert_awaited_once()
    assert result.diagnoses[0]["supporting_evidence"] == []
    assert result.overall_confidence < 0.5  # low confidence, no evidence backing


async def test_fabricated_citation_is_dropped(mock_anthropic_client):
    canned = {
        "diagnoses": [
            {
                "name": "Some diagnosis",
                "confidence": 0.5,
                "supporting_evidence": [
                    "[RAG1] White matter hyperintensities are linked to chronic hypertension.",
                    "[RAG99] This citation ID does not exist in the input.",
                    "No tag at all, just made up text.",
                ],
                "rank": 1,
            }
        ],
        "overall_confidence": 0.5,
        "conflicts": [],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await fuse(CLINICAL, None, RAG, EMPTY_LIVE)

    # only the valid [RAG1] citation survives; the fabricated ones are dropped
    assert result.diagnoses[0]["supporting_evidence"] == [
        "[RAG1] White matter hyperintensities are linked to chronic hypertension."
    ]


async def test_retries_once_on_invalid_json_then_succeeds(mock_anthropic_client):
    canned = {
        "diagnoses": [{"name": "Tension-type headache", "confidence": 0.4, "supporting_evidence": [], "rank": 1}],
        "overall_confidence": 0.4,
        "conflicts": [],
    }
    mock_anthropic_client.messages.create.side_effect = [
        _FakeResponse("Sure, here's the JSON: {not valid"),
        _canned(canned),
    ]

    result = await fuse(CLINICAL, None, EMPTY_RAG, EMPTY_LIVE)

    assert result.diagnoses[0]["name"] == "Tension-type headache"
    assert mock_anthropic_client.messages.create.await_count == 2


async def test_raises_after_two_invalid_responses(mock_anthropic_client):
    mock_anthropic_client.messages.create.side_effect = [
        _FakeResponse("not json"),
        _FakeResponse("still not json"),
    ]

    with pytest.raises(FusionError):
        await fuse(CLINICAL, None, EMPTY_RAG, EMPTY_LIVE)

    assert mock_anthropic_client.messages.create.await_count == 2


def test_format_imaging_unsupported_modality_tells_llm_not_to_use_it():
    # Must not leak the raw "no findings model available..." placeholder
    # as if it were a real finding, and must explicitly instruct the LLM
    # to disregard imaging for this case -- this is the fusion-prompt
    # side of the same fix that stops rag_agent from querying on it.
    unsupported = ImagingFindings(
        modality="chest X-ray",
        findings=["no findings model available for this modality (supported: MRI brain scan, chest X-ray)"],
        confidence=0.0,
        unsupported_modality=True,
    )

    formatted = _format_imaging(unsupported)

    assert "NOT ASSESSED" in formatted
    assert "no findings model available" not in formatted
    assert "as if no imaging were provided" in formatted


def test_format_imaging_supported_modality_unchanged():
    formatted = _format_imaging(IMAGING)

    assert "MRI brain scan" in formatted
    assert "white matter hyperintensity" in formatted
    assert "0.95" in formatted


# --------------------------------------------------------------------------
# Joint image+text fusion (Phase 5 rework, NEW in v5). call_multimodal()
# is mocked at the boundary (same "mock the client-boundary" convention
# as the rest of this suite) so no real network/provider call happens.
# --------------------------------------------------------------------------

UNSUPPORTED_IMAGING = ImagingFindings(
    modality="chest X-ray",
    findings=["no findings model available for this modality (supported: MRI brain scan, chest X-ray)"],
    confidence=0.0,
    unsupported_modality=True,
)


@pytest.fixture
def mock_call_multimodal(monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr(fusion_agent, "call_multimodal", fake)
    return fake


def test_should_attach_image_true_when_everything_aligns():
    attach, reason = _should_attach_image(IMAGING, "some/path.png", "anthropic")
    assert attach is True
    assert "MRI brain scan" in reason


def test_should_attach_image_false_no_path():
    attach, reason = _should_attach_image(IMAGING, None, "anthropic")
    assert attach is False
    assert "no image_path" in reason


def test_should_attach_image_false_no_imaging():
    attach, reason = _should_attach_image(None, "some/path.png", "anthropic")
    assert attach is False
    assert "unavailable" in reason


def test_should_attach_image_false_unsupported_modality():
    attach, reason = _should_attach_image(UNSUPPORTED_IMAGING, "some/path.png", "anthropic")
    assert attach is False
    assert "chest X-ray" in reason


def test_should_attach_image_false_provider_lacks_vision():
    attach, reason = _should_attach_image(IMAGING, "some/path.png", "mistral")
    assert attach is False
    assert "mistral" in reason


async def test_fuse_attaches_image_when_vision_capable(mock_call_multimodal, mock_anthropic_client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    canned = {
        "diagnoses": [
            {"name": "Cerebral small vessel disease", "confidence": 0.7, "supporting_evidence": [], "rank": 1}
        ],
        "overall_confidence": 0.6,
        "conflicts": [],
    }
    mock_call_multimodal.return_value = json.dumps(canned)

    result = await fuse(CLINICAL, IMAGING, RAG, EMPTY_LIVE, image_path="data/images/synthetic_demo_brain_mri_001.png")

    assert result.diagnoses[0]["name"] == "Cerebral small vessel disease"
    mock_call_multimodal.assert_awaited_once()
    call_kwargs = mock_call_multimodal.call_args
    assert call_kwargs.args[1] == "data/images/synthetic_demo_brain_mri_001.png"
    assert call_kwargs.kwargs["provider"] == "anthropic"
    # text-only client must NOT be used when the multimodal path was taken
    mock_anthropic_client.messages.create.assert_not_called()


async def test_fuse_falls_back_to_text_only_when_provider_lacks_vision(
    mock_call_multimodal, mock_anthropic_client, monkeypatch
):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    canned = {
        "diagnoses": [{"name": "Tension-type headache", "confidence": 0.4, "supporting_evidence": [], "rank": 1}],
        "overall_confidence": 0.4,
        "conflicts": [],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    result = await fuse(CLINICAL, IMAGING, RAG, EMPTY_LIVE, image_path="data/images/synthetic_demo_brain_mri_001.png")

    assert result.diagnoses[0]["name"] == "Tension-type headache"
    mock_call_multimodal.assert_not_called()
    mock_anthropic_client.messages.create.assert_awaited_once()


async def test_fuse_unsupported_modality_never_attaches_image_even_with_path(
    mock_call_multimodal, mock_anthropic_client
):
    canned = {
        "diagnoses": [{"name": "Nonspecific finding", "confidence": 0.3, "supporting_evidence": [], "rank": 1}],
        "overall_confidence": 0.3,
        "conflicts": [],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    await fuse(CLINICAL, UNSUPPORTED_IMAGING, RAG, EMPTY_LIVE, image_path="some/uploaded/chest_xray.png")

    mock_call_multimodal.assert_not_called()
    sent = mock_anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "NOT ASSESSED" in sent


async def test_fuse_case_memory_and_visual_precedent_get_distinct_framing(mock_anthropic_client):
    canned = {
        "diagnoses": [
            {"name": "Vestibular migraine", "confidence": 0.6, "supporting_evidence": ["[MEM1] prior similar case"], "rank": 1}
        ],
        "overall_confidence": 0.6,
        "conflicts": [],
    }
    mock_anthropic_client.messages.create.return_value = _canned(canned)

    case_memory = RAGEvidence(
        evidence=[EvidenceItem(text="prior similar case", source="clinician_verified:dr_lee", score=0.8)]
    )
    visual_precedent = RAGEvidence(
        evidence=[EvidenceItem(text="visually similar prior scan", source="visual_precedent:dr_lee", score=0.75)]
    )

    result = await fuse(
        CLINICAL, IMAGING, EMPTY_RAG, EMPTY_LIVE, case_memory=case_memory, visual_precedent=visual_precedent
    )

    assert result.diagnoses[0]["supporting_evidence"] == ["[MEM1] prior similar case"]
    sent = mock_anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Clinician-verified text precedent" in sent
    assert "Visually-similar past-case precedent" in sent
    assert "[VIS1]" in sent


# --------------------------------------------------------------------------
# Imaging labels are hypotheses, not findings. Regression (found live): a
# spurious secondary BiomedCLIP label ("rib fracture") was ranked at 0.85 in
# the differential on a pneumothorax film with no fracture and no trauma.
# --------------------------------------------------------------------------


def test_formatted_imaging_labels_are_presented_as_hypotheses_not_findings():
    formatted = _format_imaging(IMAGING)

    assert "Reliability" in formatted
    assert "hypotheses to check" in formatted and "not established findings" in formatted
    # one decisive label -> no claim that the classifier was torn
    assert "torn between" not in formatted


def test_formatted_imaging_with_several_labels_says_the_classifier_was_torn():
    two = ImagingFindings(modality="chest X-ray", findings=["pneumothorax", "atelectasis"], confidence=0.55)

    formatted = _format_imaging(two)

    assert "pneumothorax, atelectasis" in formatted
    assert "torn between" in formatted


def test_the_unsupported_modality_wording_is_unchanged_by_the_reliability_note():
    unsupported = ImagingFindings(
        modality="unspecified medical image", findings=["no findings model available for this modality"],
        confidence=0.0, unsupported_modality=True,
    )

    formatted = _format_imaging(unsupported)

    assert "NOT ASSESSED" in formatted and "Reliability" not in formatted


def test_both_system_prompts_forbid_ranking_a_diagnosis_on_an_imaging_label_alone():
    from backend.agents.fusion_agent import IMAGE_SYSTEM_PROMPT, SYSTEM_PROMPT

    for prompt in (SYSTEM_PROMPT, IMAGE_SYSTEM_PROMPT):
        assert "Imaging labels are hypotheses, not findings" in prompt
        assert "NEVER add a diagnosis to the differential solely because an imaging label names it" in prompt
        assert "Do not introduce a cause or mechanism" in prompt  # the 'traumatic' drift


def test_the_image_prompt_tells_the_model_not_to_rank_a_label_it_cannot_see():
    from backend.agents.fusion_agent import IMAGE_SYSTEM_PROMPT

    assert "do NOT rank a pre-filter label you cannot see in the image" in IMAGE_SYSTEM_PROMPT
