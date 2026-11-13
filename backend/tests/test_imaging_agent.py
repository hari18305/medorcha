"""Tests for backend.agents.imaging_agent.

The BiomedCLIP model load and inference are fully mocked via the
fake_model_bundle fixture below — no weights are downloaded and no real
inference runs. Fakes return real (small, hand-picked) torch tensors so
the actual cosine-similarity/softmax scoring logic in imaging_agent.py
is still exercised end to end, just against canned embeddings.
"""

import asyncio
import math
from pathlib import Path

import pytest
import torch

import backend.agents.imaging_agent as imaging_agent
from backend.agents.imaging_agent import (
    ImageNotFoundError,
    UnsupportedImageFormatError,
    analyze_image,
)
import backend.agents.imaging_labels as imaging_labels
from backend.agents.imaging_labels import FINDING_LABELS_BY_MODALITY, MODALITY_LABELS, finding_prompt

BRAIN_FINDING_LABELS = FINDING_LABELS_BY_MODALITY["MRI brain scan"]
CT_FINDING_LABELS = FINDING_LABELS_BY_MODALITY["CT scan"]
CHEST_FINDING_LABELS = FINDING_LABELS_BY_MODALITY["chest X-ray"]
ALL_FINDING_LABELS = [label for labels in FINDING_LABELS_BY_MODALITY.values() for label in labels]
from backend.models.schemas import ImagingFindings

DEMO_IMAGE_1 = Path("data/images/synthetic_demo_brain_mri_001.png")
DEMO_IMAGE_2 = Path("data/images/synthetic_demo_brain_mri_002.jpg")
DEMO_IMAGE_CT = Path("data/images/synthetic_demo_ct_scan_001.png")
DEMO_IMAGE_CHEST = Path("data/images/synthetic_demo_chest_xray_001.png")


# Finding captions (prompts) -> the label they stand for. The fake model's canned embeddings are
# keyed by LABEL, so the fake tokenizer maps a finding prompt back to its label; modality labels
# and anything else pass through unchanged. (test_finding_prompts_* check the real wording.)
_PROMPT_TO_LABEL = {
    finding_prompt(modality, label): label
    for modality, labels in FINDING_LABELS_BY_MODALITY.items()
    for label in labels
}


class _RecordingTokenizer:
    """Fake tokenizer: records what was tokenized and returns the LABEL each caption stands for."""

    def __init__(self):
        self.seen: list[list[str]] = []

    def __call__(self, texts):
        self.seen.append(list(texts))
        return [_PROMPT_TO_LABEL.get(t, t) for t in texts]


class _RecordingModel:
    """Fake BiomedCLIP model.

    encode_image always returns a fixed canned image embedding.
    encode_text looks up each label in a canned {label: embedding} map
    and stacks the results, so the real cosine-similarity/softmax math
    in imaging_agent.py runs against deterministic, known vectors.
    """

    def __init__(self, image_embedding: torch.Tensor, text_embeddings: dict[str, torch.Tensor]):
        self.image_embedding = image_embedding
        self.text_embeddings = text_embeddings
        self.encode_text_calls: list[list[str]] = []

    def encode_image(self, image_input):
        return self.image_embedding

    def encode_text(self, labels):
        self.encode_text_calls.append(list(labels))
        return torch.stack([self.text_embeddings[label] for label in labels])


def _fake_preprocess(image):
    return torch.zeros(3, 224, 224)


def _make_bundle(image_embedding: torch.Tensor, text_embeddings: dict[str, torch.Tensor]):
    model = _RecordingModel(image_embedding, text_embeddings)
    return model, _fake_preprocess, _RecordingTokenizer()


def _uniform_text_embeddings(vector: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every modality + finding label maps to the same embedding vector."""
    return {label: vector.clone() for label in MODALITY_LABELS + ALL_FINDING_LABELS}


@pytest.fixture
def dominant_finding_bundle(monkeypatch):
    """One finding/modality is a near-perfect match; everything else is orthogonal."""
    image_embedding = torch.tensor([[1.0, 0.0]])
    text_embeddings = _uniform_text_embeddings(torch.tensor([0.0, 1.0]))
    text_embeddings["MRI brain scan"] = torch.tensor([1.0, 0.0])
    text_embeddings["normal brain MRI, no acute findings"] = torch.tensor([1.0, 0.0])

    model, preprocess, tokenizer = _make_bundle(image_embedding, text_embeddings)
    monkeypatch.setattr(imaging_agent, "_get_model_bundle", lambda: (model, preprocess, tokenizer))
    return model


async def test_valid_demo_image_returns_valid_schema(dominant_finding_bundle):
    result = await analyze_image(str(DEMO_IMAGE_1))

    assert isinstance(result, ImagingFindings)
    assert result.modality == "MRI brain scan"
    assert result.findings == ["normal brain MRI, no acute findings"]
    assert result.unsupported_modality is False
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence > 0.9  # near-perfect match should score high


async def test_valid_demo_image_jpg(dominant_finding_bundle):
    result = await analyze_image(str(DEMO_IMAGE_2))

    assert isinstance(result, ImagingFindings)
    assert result.findings  # never empty


async def test_modality_hint_skips_modality_classification(dominant_finding_bundle):
    # "chest X-ray" is a supported (has a findings model) modality, so
    # finding classification still runs -- just against chest labels,
    # since the hint bypasses the (irrelevant here) modality classifier.
    result = await analyze_image(str(DEMO_IMAGE_1), modality_hint="chest X-ray")

    assert result.modality == "chest X-ray"
    assert result.unsupported_modality is False
    # only the findings classification call should have happened
    assert dominant_finding_bundle.encode_text_calls == [CHEST_FINDING_LABELS]


async def test_no_hint_classifies_modality_then_findings(dominant_finding_bundle):
    await analyze_image(str(DEMO_IMAGE_1))

    assert dominant_finding_bundle.encode_text_calls == [MODALITY_LABELS, BRAIN_FINDING_LABELS]


async def test_hint_to_unsupported_modality_skips_findings_classification(dominant_finding_bundle):
    # "unspecified medical image" has no findings model configured --
    # must not be classified against some other modality's (e.g. brain)
    # labels. (MRI, CT, and chest X-ray are all genuinely supported --
    # see FINDING_LABELS_BY_MODALITY -- this is the one deliberate
    # catch-all bucket left unsupported.)
    result = await analyze_image(str(DEMO_IMAGE_1), modality_hint="unspecified medical image")

    assert result.modality == "unspecified medical image"
    assert result.unsupported_modality is True
    assert result.confidence == 0.0
    assert len(result.findings) == 1
    assert "no findings model available for this modality" in result.findings[0]
    # no encode_text call at all -- modality hint skipped modality
    # classification, and the unsupported-modality check skipped
    # findings classification before ever calling the tokenizer/model.
    assert dominant_finding_bundle.encode_text_calls == []


async def test_detected_unsupported_modality_skips_findings_classification(monkeypatch):
    # No hint given, and the image itself scores highest against
    # "unspecified medical image" (which also has no findings model).
    image_embedding = torch.tensor([[1.0, 0.0]])
    text_embeddings = _uniform_text_embeddings(torch.tensor([0.0, 1.0]))
    text_embeddings["unspecified medical image"] = torch.tensor([1.0, 0.0])

    model, preprocess, tokenizer = _make_bundle(image_embedding, text_embeddings)
    monkeypatch.setattr(imaging_agent, "_get_model_bundle", lambda: (model, preprocess, tokenizer))

    result = await analyze_image(str(DEMO_IMAGE_1))

    assert result.modality == "unspecified medical image"
    assert result.unsupported_modality is True
    # only the modality classification call happened -- findings
    # classification was correctly skipped
    assert model.encode_text_calls == [MODALITY_LABELS]


async def test_chest_xray_modality_gets_chest_findings(monkeypatch):
    image_embedding = torch.tensor([[1.0, 0.0]])
    text_embeddings = _uniform_text_embeddings(torch.tensor([0.0, 1.0]))
    text_embeddings["chest X-ray"] = torch.tensor([1.0, 0.0])
    text_embeddings["pulmonary nodule or mass"] = torch.tensor([1.0, 0.0])

    model, preprocess, tokenizer = _make_bundle(image_embedding, text_embeddings)
    monkeypatch.setattr(imaging_agent, "_get_model_bundle", lambda: (model, preprocess, tokenizer))

    result = await analyze_image(str(DEMO_IMAGE_1))

    assert result.modality == "chest X-ray"
    assert result.unsupported_modality is False
    assert result.findings == ["pulmonary nodule or mass"]
    assert model.encode_text_calls == [MODALITY_LABELS, CHEST_FINDING_LABELS]


async def test_ct_scan_modality_gets_ct_findings(monkeypatch):
    image_embedding = torch.tensor([[1.0, 0.0]])
    text_embeddings = _uniform_text_embeddings(torch.tensor([0.0, 1.0]))
    text_embeddings["CT scan"] = torch.tensor([1.0, 0.0])
    text_embeddings["hypodense lesion"] = torch.tensor([1.0, 0.0])

    model, preprocess, tokenizer = _make_bundle(image_embedding, text_embeddings)
    monkeypatch.setattr(imaging_agent, "_get_model_bundle", lambda: (model, preprocess, tokenizer))

    result = await analyze_image(str(DEMO_IMAGE_1))

    assert result.modality == "CT scan"
    assert result.unsupported_modality is False
    assert result.findings == ["hypodense lesion"]
    assert model.encode_text_calls == [MODALITY_LABELS, CT_FINDING_LABELS]


async def test_ct_scan_modality_hint_skips_modality_classification(dominant_finding_bundle):
    # CT scan is a supported (has a findings model) modality, so finding
    # classification still runs -- just against CT labels, since the
    # hint bypasses the (irrelevant here) modality classifier.
    result = await analyze_image(str(DEMO_IMAGE_1), modality_hint="CT scan")

    assert result.modality == "CT scan"
    assert result.unsupported_modality is False
    assert dominant_finding_bundle.encode_text_calls == [CT_FINDING_LABELS]


async def test_valid_ct_demo_image_decodes_and_returns_valid_schema(dominant_finding_bundle):
    # Uses the real synthetic CT demo image file (not just a mocked
    # embedding) to confirm it's a genuinely decodable, supported image.
    result = await analyze_image(str(DEMO_IMAGE_CT), modality_hint="CT scan")

    assert isinstance(result, ImagingFindings)
    assert result.modality == "CT scan"
    assert result.unsupported_modality is False
    assert result.findings  # never empty


async def test_valid_chest_xray_demo_image_decodes_and_returns_valid_schema(dominant_finding_bundle):
    # Same real-file check as the CT one above, for the chest X-ray demo
    # image -- confirms it's a genuinely decodable, supported image.
    result = await analyze_image(str(DEMO_IMAGE_CHEST), modality_hint="chest X-ray")

    assert isinstance(result, ImagingFindings)
    assert result.modality == "chest X-ray"
    assert result.unsupported_modality is False
    assert result.findings  # never empty
    assert dominant_finding_bundle.encode_text_calls == [CHEST_FINDING_LABELS]


@pytest.mark.parametrize(
    "slug", ["brain_mri", "ct_scan", "chest_xray"]
)
def test_each_supported_modality_has_two_to_three_decodable_demo_images(slug):
    # Phase 1b: "2-3 small synthetic placeholder images per supported
    # modality in data/images/" -- guards against a modality silently
    # losing its demo images (chest X-ray had none until this was added).
    from PIL import Image

    files = sorted(Path("data/images").glob(f"synthetic_demo_{slug}_*"))
    assert 2 <= len(files) <= 3, f"{slug}: expected 2-3 demo images, found {len(files)}"
    for file in files:
        with Image.open(file) as image:
            image.verify()


async def test_missing_file_raises_typed_error(monkeypatch):
    called = False

    def _should_not_load():
        nonlocal called
        called = True
        raise AssertionError("model should not load when the file is missing")

    monkeypatch.setattr(imaging_agent, "_get_model_bundle", _should_not_load)

    with pytest.raises(ImageNotFoundError):
        await analyze_image("data/images/does_not_exist.png")

    assert called is False


async def test_unsupported_extension_raises_typed_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        imaging_agent,
        "_get_model_bundle",
        lambda: (_ for _ in ()).throw(AssertionError("model should not load")),
    )

    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("not an image")

    with pytest.raises(UnsupportedImageFormatError):
        await analyze_image(str(bad_file))


async def test_undecodable_content_with_allowed_extension_raises_typed_error(tmp_path):
    fake_png = tmp_path / "broken.png"
    fake_png.write_bytes(b"this is not real png data")

    with pytest.raises(UnsupportedImageFormatError):
        await analyze_image(str(fake_png))


async def test_findings_never_empty_when_all_scores_below_threshold(monkeypatch):
    # All labels equally (orthogonally) similar, including modality labels
    # -> "MRI brain scan" wins the modality tie (stable sort, first listed),
    # then uniform softmax over its 13 finding labels is ~1/13 ≈ 0.077,
    # below FINDING_SCORE_THRESHOLD (0.15).
    image_embedding = torch.tensor([[1.0, 0.0]])
    text_embeddings = _uniform_text_embeddings(torch.tensor([0.0, 1.0]))

    model, preprocess, tokenizer = _make_bundle(image_embedding, text_embeddings)
    monkeypatch.setattr(imaging_agent, "_get_model_bundle", lambda: (model, preprocess, tokenizer))

    result = await analyze_image(str(DEMO_IMAGE_1))

    assert result.modality == "MRI brain scan"
    assert result.findings == [BRAIN_FINDING_LABELS[0]]
    expected_score = 1.0 / len(BRAIN_FINDING_LABELS)
    assert math.isclose(result.confidence, expected_score, rel_tol=1e-4)


def _two_close_labels_bundle(monkeypatch, second_cosine: float):
    """A bundle where "white matter hyperintensity" matches the image perfectly
    and "mass lesion" matches it with cosine `second_cosine`; every other
    label points the opposite way (negligible softmax score)."""
    image_embedding = torch.tensor([[1.0, 0.0]])
    text_embeddings = _uniform_text_embeddings(torch.tensor([-1.0, 0.0]))
    text_embeddings["white matter hyperintensity"] = torch.tensor([1.0, 0.0])
    text_embeddings["mass lesion"] = torch.tensor([second_cosine, math.sqrt(1 - second_cosine**2)])
    model, preprocess, tokenizer = _make_bundle(image_embedding, text_embeddings)
    monkeypatch.setattr(imaging_agent, "_get_model_bundle", lambda: (model, preprocess, tokenizer))


async def test_a_genuinely_torn_top_two_are_both_reported_sorted_descending(monkeypatch):
    # cosine 1.0 vs 0.998 -> scores ~0.55 / ~0.45 (the second is ~82% of the
    # top): the model really is torn between the two, so both are reported.
    _two_close_labels_bundle(monkeypatch, 0.998)

    result = await analyze_image(str(DEMO_IMAGE_1))

    assert result.findings[:2] == ["white matter hyperintensity", "mass lesion"]
    assert all(result.findings[i] != result.findings[i + 1] for i in range(len(result.findings) - 1))


async def test_a_weak_secondary_label_is_dropped_even_though_it_clears_the_absolute_floor(monkeypatch):
    """Regression (found live): a real pneumothorax film scored pneumothorax
    0.68 and a spurious "rib fracture" 0.15; fusion then ranked "rib fracture"
    at 0.85. Here cosine 1.0 vs 0.99 gives ~0.73 / ~0.27: the second clears the
    0.15 floor but is only ~37% of the top, so it must NOT be reported."""
    _two_close_labels_bundle(monkeypatch, 0.99)

    result = await analyze_image(str(DEMO_IMAGE_1))

    assert result.findings == ["white matter hyperintensity"]
    assert math.isclose(result.confidence, 0.73, abs_tol=0.02)  # the top label's score is unchanged


async def test_the_relative_threshold_is_configurable(monkeypatch):
    _two_close_labels_bundle(monkeypatch, 0.99)
    monkeypatch.setattr(imaging_agent, "FINDING_RELATIVE_THRESHOLD", 0.3)  # 0.27/0.73 = 0.37 now qualifies

    result = await analyze_image(str(DEMO_IMAGE_1))

    assert result.findings == ["white matter hyperintensity", "mass lesion"]


async def test_the_absolute_floor_still_applies_on_top_of_the_relative_rule(monkeypatch):
    # relative threshold 0 would keep anything; the 0.15 floor must still drop
    # a ~0.01 label
    _two_close_labels_bundle(monkeypatch, 0.95)  # second ~ 0.007
    monkeypatch.setattr(imaging_agent, "FINDING_RELATIVE_THRESHOLD", 0.0)

    result = await analyze_image(str(DEMO_IMAGE_1))

    assert result.findings == ["white matter hyperintensity"]


async def test_empty_image_path_raises_value_error():
    with pytest.raises(ValueError):
        await analyze_image("")


# --------------------------------------------------------------------------
# Finding prompts: captions are embedded, labels are returned
# --------------------------------------------------------------------------


def test_finding_prompts_are_descriptive_captions_and_returned_names_are_still_labels(monkeypatch):
    bundle = _make_bundle(torch.tensor([[1.0, 0.0]]), _uniform_text_embeddings(torch.tensor([1.0, 0.0])))
    tokenizer = bundle[2]
    monkeypatch.setattr(imaging_agent, "_get_model_bundle", lambda: bundle)

    result = asyncio.run(imaging_agent.analyze_image(str(DEMO_IMAGE_CT), modality_hint="CT scan"))

    (finding_call,) = tokenizer.seen                                          # a hint skips modality detection
    labels = FINDING_LABELS_BY_MODALITY["CT scan"]
    assert finding_call == [finding_prompt("CT scan", lab) for lab in labels]
    assert finding_call != labels                                             # captions, not bare labels
    assert "head CT scan showing intracranial hemorrhage" in finding_call[labels.index("hemorrhage")]
    assert all(f in labels for f in result.findings)                          # what is returned is still a label


def test_modality_prompts_are_unchanged(monkeypatch):
    # The held-out experiment did not show a modality-wording gain (1 image, bar 2), so detection
    # still embeds the bare MODALITY_LABELS.
    bundle = _make_bundle(torch.tensor([[1.0, 0.0]]), _uniform_text_embeddings(torch.tensor([1.0, 0.0])))
    monkeypatch.setattr(imaging_agent, "_get_model_bundle", lambda: bundle)

    asyncio.run(imaging_agent.analyze_image(str(DEMO_IMAGE_CT)))

    assert bundle[2].seen[0] == MODALITY_LABELS


def test_every_finding_label_has_a_distinct_caption_with_its_modality():
    seen = set()
    for modality, labels in FINDING_LABELS_BY_MODALITY.items():
        for label in labels:
            prompt = finding_prompt(modality, label)
            assert prompt not in seen and prompt != label
            assert prompt.startswith(imaging_labels.MODALITY_PHRASE[modality] + " showing ")
            seen.add(prompt)


def test_label_style_env_restores_the_bare_labels(monkeypatch):
    monkeypatch.setenv("IMAGING_FINDING_PROMPTS", "label")
    assert finding_prompt("CT scan", "hemorrhage") == "hemorrhage"
    monkeypatch.delenv("IMAGING_FINDING_PROMPTS")
    assert finding_prompt("CT scan", "hemorrhage") != "hemorrhage"


def test_a_label_without_a_description_falls_back_to_the_bare_label(monkeypatch):
    monkeypatch.setitem(imaging_labels.FINDING_DESCRIPTIONS, "ultrasound", {})
    assert finding_prompt("ultrasound", "anything") == "anything"
    assert finding_prompt("CT scan", "a label added later with no description") == "a label added later with no description"


def test_production_prompts_match_the_ones_the_held_out_experiment_selected():
    # The experiment kept its own frozen copy of the wording it tested; production must say the
    # same thing, or the held-out result would not describe what actually ships.
    from backend.tests import imaging_prompt_experiment as pe

    for modality, labels in FINDING_LABELS_BY_MODALITY.items():
        for label in labels:
            assert [finding_prompt(modality, label)] == pe.FINDING_CANDIDATES["T1_showing"](modality, label)
