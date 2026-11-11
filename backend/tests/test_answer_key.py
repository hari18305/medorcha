"""Guards for data/test_cases/answer_key.json.

The key is a developer-drafted expected-diagnosis list used by the LLM-based
evaluation. These tests only keep it in step with the fixtures; they say
nothing about whether the key is clinically right (it is not clinician-verified).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "test_cases"
KEY = json.loads((ROOT / "answer_key.json").read_text(encoding="utf-8"))


def _stems(folder: Path, pattern: str) -> set[str]:
    return {p.stem for p in folder.glob(pattern)}


def test_every_synthetic_case_has_an_entry_and_no_stale_entries():
    assert set(KEY["cases"]) == _stems(ROOT, "CASE*.json")


def test_every_real_case_has_an_entry_and_no_stale_entries():
    assert set(KEY["real_cases"]) == _stems(ROOT / "real", "REAL*.json")


def test_entries_are_well_formed():
    for group in ("cases", "real_cases"):
        for name, entry in KEY[group].items():
            assert entry["kind"] in ("differential", "abstain"), name
            if entry["kind"] == "differential":
                assert entry["differential"], name
                assert all(d == d.strip().lower() for d in entry["differential"]), name
            else:
                assert "differential" not in entry, name


def test_key_declares_it_is_not_clinician_verified():
    # Guard against the draft status silently being dropped from the file.
    prov = KEY["_provenance"]
    assert "NOT clinician-verified" in prov["status"]
    assert prov["verified_by"] is None


def test_image_only_and_empty_cases_expect_abstention():
    # No text/report means nothing to diagnose from, unless the image is real
    # and shows a recognisable finding (REAL007).
    for group, folder in (("cases", ROOT), ("real_cases", ROOT / "real")):
        for name, entry in KEY[group].items():
            case = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
            empty_text = not (case.get("clinical_text") or case.get("medical_report"))
            synthetic_img = "synthetic_demo_" in (case.get("image_path") or "")
            if empty_text and (synthetic_img or not case.get("image_path")):
                assert entry["kind"] == "abstain", name
