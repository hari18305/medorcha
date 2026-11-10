"""Developer tool: classify the REAL images in data/images/real/ with the real
BiomedCLIP and report how each is classified.

    python scripts/check_real_images.py

The intended modality comes from the filename (real_ct_scan_*, real_brain_mri_*,
real_chest_xray_*). Exits non-zero if any image is classified as a different
modality. This is a plausibility check on a handful of hand-picked public
images -- NOT an accuracy measurement (see data/images/real/ATTRIBUTION.md).
"""

import asyncio
import sys
from pathlib import Path

REAL_DIR = Path("data/images/real")
SLUG_TO_MODALITY = {"ct_scan": "CT scan", "brain_mri": "MRI brain scan", "chest_xray": "chest X-ray"}


def intended_modality(name: str) -> str | None:
    return next((m for slug, m in SLUG_TO_MODALITY.items() if f"_{slug}_" in name), None)


async def main() -> int:
    sys.path.insert(0, ".")
    from backend.agents.imaging_agent import analyze_image

    mismatches = 0
    print(f"{'image':<36} {'intended':<16} {'classified as':<26} conf  top findings")
    for path in sorted(p for p in REAL_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")):
        want = intended_modality(path.name)
        result = await analyze_image(str(path))
        ok = want is None or result.modality == want
        mismatches += not ok
        print(f"{path.name:<36} {want or '?':<16} {result.modality:<26} {result.confidence:4.2f}  {result.findings[:2]}  {'OK' if ok else 'MISMATCH'}")
    return mismatches


if __name__ == "__main__":
    failed = asyncio.run(main())
    sys.exit(f"{failed} image(s) classified as a different modality" if failed else 0)
