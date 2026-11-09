"""Developer tool: (re)generates the synthetic demo images in data/images/.

    python scripts/generate_demo_images.py           # write the images
    python scripts/generate_demo_images.py --check   # write, then classify each
                                                     # with the REAL BiomedCLIP and
                                                     # fail if one is not classified
                                                     # as intended

These are SYNTHETIC placeholder images drawn with numpy/Pillow -- schematic
anatomy (skull ring + ventricles for CT, lungs + ribs + heart for chest X-ray,
CSF/gray-matter contrast for MRI). They are NOT real scans, not derived from
any patient data, and not clinically meaningful. They exist so the imaging
pipeline can be exercised end to end.

Why a recipe per file: BiomedCLIP's zero-shot modality guess is sensitive to
tonal distribution, not anatomy. Flat shapes (the original placeholders) were
classified essentially at random -- a brain MRI placeholder came back as "CT
scan". So each recipe below applies the histogram treatment that, measured
against the real model, makes the image classify as its INTENDED modality, and
--check re-verifies that. A passing --check says the FIXTURE routes through the
intended code path; it says nothing about BiomedCLIP's real-world accuracy.

Everything is deterministic (fixed seeds), so the output is reproducible.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

OUT_DIR = Path("data/images")
W = 224


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _grid():
    y, x = np.mgrid[0:W, 0:W].astype(np.float32)
    return (x - W / 2) / (W / 2), (y - W / 2) / (W / 2)


def _smooth_noise(rng, sigma):
    n = rng.normal(0, 1, (W, W)).astype(np.float32)
    img = Image.fromarray(((n - n.min()) / np.ptp(n) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
    return np.asarray(img, np.float32)


def _finish(arr, noise, blur, rng):
    arr = arr + rng.normal(0, noise, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return img.convert("RGB")


def head_ct(seed):
    """Axial head CT: bright skull ring, mid-gray brain, dark ventricles, falx."""
    rng = np.random.default_rng(seed)
    x, y = _grid()
    a, b = 0.80 + rng.uniform(-0.03, 0.03), 0.90 + rng.uniform(-0.03, 0.03)
    r = np.sqrt((x / a) ** 2 + (y / b) ** 2)
    arr = np.zeros((W, W), np.float32)
    arr[r < 1.0] = 235
    arr[r < 0.90] = 105 + rng.normal(0, 3)
    tex = _smooth_noise(rng, 5)
    arr[r < 0.90] += (tex[r < 0.90] - 128) * 0.25
    for sx in (-1, 1):
        arr[(((x - sx * 0.16) / 0.11) ** 2 + ((y + 0.05) / 0.28) ** 2 < 1) & (r < 0.9)] = 55
    arr[(np.abs(x) < 0.012) & (r < 0.88)] += 18
    return _finish(arr, noise=7, blur=1.0, rng=rng)


def chest_xray(seed):
    """PA chest radiograph: dark lungs with vascular markings, ribs, clavicles, heart, spine."""
    rng = np.random.default_rng(seed)
    x, y = _grid()
    arr = np.full((W, W), 40, np.float32)
    body = (np.abs(x) < 0.92) & (y > -0.95)
    arr[body] = 105
    for sx in (-1, 1):
        arr[((x - sx * 0.47) / 0.36) ** 2 + ((y + 0.02) / 0.68) ** 2 < 1] = 50 + rng.normal(0, 2)
    vess = _smooth_noise(rng, 2.2)
    lung = arr < 70
    arr[lung] += (vess[lung] - 128) * 0.35
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")
    d = ImageDraw.Draw(img)
    for i in range(7):
        yy = 38 + i * 21
        for sx in (-1, 1):
            cx = W / 2 + sx * 52
            d.arc([cx - 62, yy - 18, cx + 62, yy + 30], 195 if sx < 0 else 270, 270 if sx < 0 else 345, fill=150, width=3)
    d.line([(24, 30), (W / 2, 46), (W - 24, 30)], fill=170, width=4)
    arr = np.asarray(img, np.float32)
    arr[(np.abs(x) < 0.07) & body] += 55
    arr[((x + 0.12) / 0.33) ** 2 + ((y - 0.38) / 0.30) ** 2 < 1] = 140
    arr[(y > 0.72 + 0.10 * np.cos(x * 3.0)) & body] = 128
    return _finish(arr, noise=6, blur=1.2, rng=rng)


def brain_mri(seed):
    """T2-weighted-like axial brain MRI: dark skull, bright CSF, gray/white matter, bright ventricles."""
    rng = np.random.default_rng(seed)
    x, y = _grid()
    a, b = 0.78 + rng.uniform(-0.03, 0.03), 0.88 + rng.uniform(-0.03, 0.03)
    r = np.sqrt((x / a) ** 2 + (y / b) ** 2)
    arr = np.zeros((W, W), np.float32)
    arr[r < 1.0] = 70
    arr[r < 0.93] = 25
    arr[r < 0.88] = 205
    arr[r < 0.82] = 120
    tex = _smooth_noise(rng, 3.5)
    arr[r < 0.82] += (tex[r < 0.82] - 128) * 0.55
    for sx in (-1, 1):
        arr[(((x - sx * 0.15) / 0.10) ** 2 + ((y + 0.04) / 0.26) ** 2 < 1) & (r < 0.8)] = 215
    return _finish(arr, noise=5, blur=1.0, rng=rng)


def abstract_photo(seed):
    """A smooth colourful abstract image -- deliberately NOT any supported
    medical modality, so the imaging agent should report it as unsupported."""
    rng = np.random.default_rng(seed)
    chans = [(_smooth_noise(rng, 18) * 0.9 + 12) for _ in range(3)]
    arr = np.stack(chans, axis=-1)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    for _ in range(5):
        cx, cy, rad = rng.integers(30, 194), rng.integers(30, 194), rng.integers(15, 45)
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=tuple(int(v) for v in rng.integers(40, 255, 3)))
    return img.filter(ImageFilter.GaussianBlur(2))


# ---------------------------------------------------------------------------
# Tonal treatments (what actually steers the classifier)
# ---------------------------------------------------------------------------


def _equalize(im):
    return ImageOps.equalize(im.convert("L")).convert("RGB")


def _equalize_gamma(g):
    def apply(im):
        a = np.asarray(_equalize(im).convert("L"), np.float32) / 255
        return Image.fromarray((255 * a**g).astype(np.uint8)).convert("RGB")

    return apply


def _identity(im):
    return im


# (filename, intended modality, generator, seed, treatment)
RECIPES = [
    ("synthetic_demo_brain_mri_001.png", "MRI brain scan", brain_mri, 1, _equalize),
    ("synthetic_demo_brain_mri_002.jpg", "MRI brain scan", brain_mri, 2, _equalize),
    ("synthetic_demo_ct_scan_001.png", "CT scan", head_ct, 1, _identity),
    ("synthetic_demo_ct_scan_002.png", "CT scan", head_ct, 2, _identity),
    ("synthetic_demo_ct_scan_003.jpg", "CT scan", head_ct, 3, _identity),
    ("synthetic_demo_chest_xray_001.png", "chest X-ray", chest_xray, 1, _equalize_gamma(0.7)),
    ("synthetic_demo_chest_xray_002.png", "chest X-ray", chest_xray, 2, _equalize_gamma(0.7)),
    ("synthetic_demo_chest_xray_003.jpg", "chest X-ray", chest_xray, 4, _equalize_gamma(0.7)),
    ("synthetic_demo_unspecified_001.png", "unspecified medical image", abstract_photo, 1, _identity),
]


def write_images() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, _modality, generate, seed, treat in RECIPES:
        image = treat(generate(seed)).convert("RGB")
        path = OUT_DIR / filename
        if path.suffix == ".jpg":
            image.save(path, quality=92)
        else:
            image.save(path)
        print(f"wrote {path}")


async def check() -> int:
    sys.path.insert(0, ".")
    from backend.agents.imaging_agent import analyze_image

    failures = 0
    print(f"\n{'image':<40} {'intended':<26} {'classified as':<26} conf")
    for filename, intended, *_ in RECIPES:
        result = await analyze_image(str(OUT_DIR / filename))
        ok = result.modality == intended
        failures += not ok
        print(f"{filename:<40} {intended:<26} {result.modality:<26} {result.confidence:4.2f}  {'OK' if ok else 'MISMATCH'}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="classify each image with the real BiomedCLIP")
    args = parser.parse_args()
    write_images()
    if args.check:
        failed = asyncio.run(check())
        if failed:
            sys.exit(f"{failed} image(s) were not classified as intended")
        print("\nAll images classified as intended.")


if __name__ == "__main__":
    main()
