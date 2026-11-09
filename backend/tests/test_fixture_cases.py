"""Guards the synthetic test-case fixtures in data/test_cases/ (Phase 16).

The build spec asks for a case set covering the whole input matrix with "at
least a few per imaging modality". This does NOT run the pipeline -- it checks
the fixture files themselves, offline, so that coverage cannot silently erode
(a deleted image, a duplicate case_id, a modality that loses its cases).

Whether the REAL BiomedCLIP classifies each demo image as its intended
modality is a separate, model-dependent question: run
`python scripts/generate_demo_images.py --check` for that. The deterministic,
mocked-classifier coverage of every modality lives in test_integration.py.
"""

import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from backend.models.schemas import CaseInput

CASES_DIR = Path("data/test_cases")
CASE_FILES = sorted(CASES_DIR.glob("CASE*.json"))

# demo-image filename slug -> the modality it is designed to be classified as
SLUG_TO_MODALITY = {
    "brain_mri": "MRI brain scan",
    "ct_scan": "CT scan",
    "chest_xray": "chest X-ray",
    "unspecified": "unspecified medical image",
}


def _load(path: Path) -> CaseInput:
    return CaseInput(**json.loads(path.read_text(encoding="utf-8")))


def _intended_modality(case: CaseInput) -> str | None:
    if not case.image_path:
        return None
    name = Path(case.image_path).name
    return next((modality for slug, modality in SLUG_TO_MODALITY.items() if f"_{slug}_" in name), None)


def test_there_are_enough_cases_to_cover_the_matrix():
    assert len(CASE_FILES) >= 15  # the spec's floor ("15-20 synthetic cases")


@pytest.mark.parametrize("path", CASE_FILES, ids=[p.stem for p in CASE_FILES])
def test_each_case_file_is_a_valid_caseinput_whose_name_matches_its_id_and_image_exists(path):
    case = _load(path)

    assert case.case_id, path.name
    if case.image_path:
        image = Path(case.image_path)
        assert image.exists(), f"{path.name} points at a missing image: {case.image_path}"
        with Image.open(image) as decoded:
            decoded.verify()


def test_case_ids_are_unique_apart_from_the_two_documented_original_fixtures():
    # CASE001.json and CASE001_with_image.json deliberately share a case_id
    # (the original Phase 0 fixtures; harmless unless both are POSTed through
    # the real API -- see DEMO_READINESS.md). Nothing else may collide.
    counts = Counter(_load(p).case_id for p in CASE_FILES)
    duplicates = {case_id for case_id, n in counts.items() if n > 1}

    assert duplicates <= {"CASE001"}


def test_every_imaging_modality_has_at_least_three_cases_and_one_unsupported_case():
    per_modality = Counter(_intended_modality(_load(p)) for p in CASE_FILES)

    for modality in ("MRI brain scan", "CT scan", "chest X-ray"):
        assert per_modality[modality] >= 3, f"only {per_modality[modality]} case(s) for {modality}"
    # the spec's "plus one unsupported-modality test"
    assert per_modality["unspecified medical image"] >= 1


@pytest.mark.parametrize("modality", ["MRI brain scan", "CT scan", "chest X-ray"])
def test_each_supported_modality_has_text_plus_image_and_image_only_cases(modality):
    cases = [c for c in (_load(p) for p in CASE_FILES) if _intended_modality(c) == modality]
    has_text = lambda c: bool((c.clinical_text or "").strip() or (c.medical_report or "").strip())

    assert any(has_text(c) for c in cases), f"no text+image case for {modality}"
    assert any(not has_text(c) for c in cases), f"no image-only case for {modality}"


def test_the_input_combinations_the_matrix_requires_are_all_present():
    cases = [_load(p) for p in CASE_FILES]

    assert any(c.clinical_text and not c.medical_report and not c.image_path for c in cases), "text only"
    assert any(c.medical_report and not c.clinical_text and not c.image_path for c in cases), "report only"
    assert any(c.clinical_text and c.medical_report for c in cases), "text + report"
    assert any(c.image_path and not c.clinical_text and not c.medical_report for c in cases), "image only"
    assert any(c.image_path and (c.clinical_text or c.medical_report) for c in cases), "text/report + image"
    assert any(c.labs for c in cases), "populated labs"
