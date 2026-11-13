"""Build the real-image imaging benchmark from open-licence Wikimedia Commons files.

    python -m scripts.build_imaging_benchmark            # download + write manifest/attribution
    python -m scripts.build_imaging_benchmark --check    # verify the files on disk match the manifest

Output (committed): data/benchmarks/imaging_real/{images/, manifest.json, ATTRIBUTION.md}

WHAT THIS IS. A small set of REAL radiology images (chest X-ray, head CT, brain
MRI) with a finding label each, used to measure how BiomedCLIP's zero-shot
classification (backend/agents/imaging_agent.py) actually behaves. It replaces the
earlier benchmark, which scored the classifier on generated gradient-and-ellipse
pictures whose "intended label" was arbitrary (near chance by construction).

WHAT THE LABELS ARE. Each label comes from the file's own Wikimedia Commons
category / description, which is usually a case report or a radiologist's teaching
caption. They are NOT re-read by a radiologist for this project, and the developer
who picked them is not one. Treat them as "labelled by the uploader".

KNOWN WEAKNESSES (also recorded in the manifest and the results):
  * Small: a few images per finding, so every per-finding number has a wide
    confidence interval. A handful of findings (MRI atrophy) had no usable image and
    are not covered. There is exactly ONE normal head-CT subject.
  * Possible training-data overlap: BiomedCLIP was trained on PubMed Central
    figure-caption pairs. Images taken from journal articles are flagged
    (`possible_pretraining_overlap`) by a keyword heuristic on the file's own
    description; the heuristic can miss.
  * Selection bias: images on Commons are chosen to be clear, typical teaching
    examples. Real clinical images are harder, so accuracy here is optimistic.
  * Images are downscaled (<= 768 px wide) by Wikimedia's thumbnail service;
    changes are stated in ATTRIBUTION.md as CC BY / CC BY-SA require.

The selection below is an explicit list of exact Commons titles, not a search, so a
rebuild gives the same benchmark.
"""

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

import httpx

OUT_DIR = Path("data/benchmarks/imaging_real")
IMG_DIR = OUT_DIR / "images"
MANIFEST = OUT_DIR / "manifest.json"
ATTRIBUTION = OUT_DIR / "ATTRIBUTION.md"
API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia's robot policy requires an identifying User-Agent with a contact.
UA = {"User-Agent": "MedOrchestrate-benchmark/0.1 (university coursework prototype; https://github.com/hari18305/medorcha)"}
MAX_WIDTH = 768
ALLOWED_LICENSE = re.compile(r"^(CC0|CC BY( |-)?(SA )?[\d.]*|Public domain)", re.I)
# Heuristic: the file's own description/credit says it comes from a journal article.
OVERLAP_HINT = re.compile(r"journal|case report|\bpmc\b|doi|plos|\bbmc\b|cureus|open access|et al|article|figure \d", re.I)

CXR, CT, MRI = "chest X-ray", "CT scan", "MRI brain scan"
NORMAL_CXR = "normal chest X-ray, no acute findings"
NORMAL_CT = "normal head CT, no acute findings"
NORMAL_MRI = "normal brain MRI, no acute findings"

# (Commons title, modality, accepted labels (first = primary), group)
# `group` = one patient/series; the scorer counts a group once. Default: the title.
SELECTION = [
    # ---- chest X-ray ---------------------------------------------------------
    ("Chest X-ray 2346.jpg", CXR, [NORMAL_CXR], None),
    ("Chest Xray PA 3-8-2010.png", CXR, [NORMAL_CXR], None),
    ("Chest.png", CXR, [NORMAL_CXR], None),
    ("05-Spontanpneumothorax.jpg", CXR, ["pneumothorax"], None),
    ("09-01-Pneumothorax.png", CXR, ["pneumothorax"], None),
    ("Deep sulcus sign 59jm - Roe - 001.jpg", CXR, ["pneumothorax"], None),
    ("Effusion.png", CXR, ["pleural effusion"], None),
    ("Effusionhalf.PNG", CXR, ["pleural effusion"], None),
    ("Bilateral Pleural Effusion.jpg", CXR, ["pleural effusion"], None),
    ("Cardiomegalia.JPG", CXR, ["cardiomegaly"], None),
    ("Cardiomegally.PNG", CXR, ["cardiomegaly"], None),
    ("Enfermedad Mitral.JPG", CXR, ["cardiomegaly"], None),
    ("Aspiration pneumonia201711-3264.jpg", CXR, ["consolidation"], None),
    ("Chest radiograph in influensa and H influenzae, posteroanterior.jpg", CXR, ["consolidation"], None),
    ("Chest X-ray in influenza and Haemophilus influenzae.jpg", CXR, ["consolidation"], None),
    ("16-01-Lungenoedem.png", CXR, ["pulmonary edema"], None),
    ("AP portable CXR of a patient in acute pulmonary oedema.jpg", CXR, ["pulmonary edema"], None),
    ("Lungenoedem nach NSTEMI 73W - CR ap - 001.jpg", CXR, ["pulmonary edema"], None),
    ("LungCa.png", CXR, ["pulmonary nodule or mass"], None),
    ("LungCancer-Xray-01.jpg", CXR, ["pulmonary nodule or mass"], None),
    ("Plattenepithelkarzinom der Lunge rechter Oberlappen 76W - CR pa - 001.jpg", CXR, ["pulmonary nodule or mass"], None),
    ("Emphysema2008.jpg", CXR, ["hyperinflation consistent with emphysema"], None),
    ("BullousEmphysema.png", CXR, ["hyperinflation consistent with emphysema"], None),
    ("Unterlappenatelektase rechts pa.jpg", CXR, ["atelectasis"], None),
    ("Oberlappenatelektase links pa.jpg", CXR, ["atelectasis"], None),
    ("Atelectasia.JPG", CXR, ["atelectasis"], None),
    ("Rib fracture 2347.jpg", CXR, ["rib fracture"], None),
    ("Гемоторакс, переломы рёбер.jpg", CXR, ["rib fracture", "pleural effusion"], None),
    ("Rippenserienfraktur Röntgen Thorax.jpg", CXR, ["rib fracture"], None),
    ("HilarAdenopathyNo.png", CXR, ["hilar or mediastinal widening"], None),
    # ---- head CT -------------------------------------------------------------
    # normal head CT: only the three Uppsala slices (existing, below). Haggstrom's normal series
    # was dropped: each slice carries a screenshot of the viewer's sagittal reference.
    ("Trauma subdural.jpg", CT, ["hemorrhage"], None),
    ("BilateralSubDur.png", CT, ["hemorrhage"], None),
    ("Intra Parenchymal Bleed with Edema.jpg", CT, ["hemorrhage"], None),
    ("Intracerebral hemorrage (CT scan).jpg", CT, ["hemorrhage"], None),
    ("Frontale Kontusionsblutung als Contre coup 01.png", CT, ["hemorrhage"], None),
    ("CT Brain MCA Infarct.jpg", CT, ["acute infarct", "hypodense lesion"], None),
    ("Right MCA and PCA territories infarct.jpg", CT, ["acute infarct", "hypodense lesion"], None),
    ("POCITKWojder.jpg", CT, ["acute infarct", "hypodense lesion"], None),
    ("Hydrocephalus.jpg", CT, ["ventricular enlargement"], None),
    ("NPH CT 20.png", CT, ["ventricular enlargement"], "dilmen_nph_ct"),
    ("BrainCa.png", CT, ["mass lesion"], None),
    ("Medulloblastom CT.jpg", CT, ["mass lesion"], None),
    ("Tectal plate glioma CT.jpg", CT, ["mass lesion"], None),
    ("FrontalSinusitis.png", CT, ["sinus abnormality"], None),
    ("Chronische Sinusitis maxillaris mit Ebonisierung des Knochens 92W - 001.jpg", CT, ["sinus abnormality"], None),
    # ---- brain MRI -----------------------------------------------------------
    ("Brain MRI t2 142301.png", MRI, [NORMAL_MRI, "no significant abnormality"], None),
    ("DWI MRI of a cerebral infarction.jpg", MRI, ["acute infarct"], None),
    ("LACIMRIflairWojder.jpg", MRI, ["acute infarct", "chronic infarct"], None),
    ("MRI Cerebellum Hemorrhage.jpg", MRI, ["hemorrhage"], None),
    ("Frontale Kontusionsblutung MRT T1 01.png", MRI, ["hemorrhage"], None),
    ("Glioblastoma - MR coronal with contrast.jpg", MRI, ["mass lesion"], None),
    ("Glioblastoma multiforme - MRT T1 ax.jpg", MRI, ["mass lesion"], None),
    ("Huge Meningioma.jpg", MRI, ["mass lesion"], None),
    ("Butterfly glioblastoma.png", MRI, ["mass lesion"], None),
    ("MSMRI.png", MRI, ["white matter hyperintensity"], None),
    ("NPH MRI 010.png", MRI, ["ventricular enlargement"], "dilmen_nph_mri"),
]

# The six real images already in the repo (data/images/real, see its ATTRIBUTION.md).
EXISTING = [
    ("data/images/real/real_chest_xray_normal_pa.jpg", CXR, [NORMAL_CXR], None, "CC0 1.0", False),
    ("data/images/real/real_chest_xray_pneumothorax.png", CXR, ["pneumothorax"], None, "CC BY 4.0", True),  # figure carries arrows
    ("data/images/real/real_ct_scan_06.png", CT, [NORMAL_CT], "uppsala_normal_ct", "CC0 1.0", False),
    ("data/images/real/real_ct_scan_10.png", CT, [NORMAL_CT], "uppsala_normal_ct", "CC0 1.0", False),
    ("data/images/real/real_ct_scan_14.png", CT, [NORMAL_CT], "uppsala_normal_ct", "CC0 1.0", False),
    ("data/images/real/real_brain_mri_t1.png", MRI, [NORMAL_MRI, "no significant abnormality"], None, "Public domain", False),
]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(s or ""))).strip()


def fetch_metadata(titles: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with httpx.Client(headers=UA, timeout=40, follow_redirects=True) as c:
        for i in range(0, len(titles), 20):
            batch = titles[i:i + 20]
            r = c.get(API, params={
                "action": "query", "format": "json", "titles": "|".join("File:" + t for t in batch),
                "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": MAX_WIDTH,
                "iiextmetadatafilter": "LicenseShortName|Artist|ImageDescription|Credit"})
            r.raise_for_status()
            time.sleep(0.6)
            q = r.json()["query"]
            norm = {n["to"]: n["from"] for n in q.get("normalized", [])}
            for page in q["pages"].values():
                title = page["title"][5:]
                if "imageinfo" not in page:
                    raise SystemExit(f"Commons has no file titled {title!r} (renamed or deleted?)")
                ii, md = page["imageinfo"][0], page["imageinfo"][0].get("extmetadata", {})
                get = lambda k: _clean((md.get(k) or {}).get("value", ""))  # noqa: E731
                out[norm.get("File:" + title, "File:" + title)[5:]] = {
                    "title": title, "license": get("LicenseShortName"), "artist": get("Artist")[:120],
                    "description": get("ImageDescription")[:300], "credit": get("Credit")[:200],
                    "mime": ii["mime"], "orig_w": ii["width"], "orig_h": ii["height"],
                    "url": ii.get("thumburl") or ii["url"], "page": "https://commons.wikimedia.org/wiki/File:" + title.replace(" ", "_")}
    return out


def _slug(i: int, modality: str, label: str, ext: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", f"{modality}_{label}".lower()).strip("_")
    return f"{i:02d}_{base}{ext}"


def build() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    meta = fetch_metadata([t for t, *_ in SELECTION])
    entries = []
    with httpx.Client(headers=UA, timeout=60, follow_redirects=True) as c:
        for i, (title, modality, labels, group) in enumerate(SELECTION, 1):
            m = meta[title]
            if not ALLOWED_LICENSE.match(m["license"]):
                raise SystemExit(f"{title}: licence {m['license']!r} is not an allowed open licence")
            ext = ".jpg" if m["mime"] == "image/jpeg" else ".png"
            path = IMG_DIR / _slug(i, modality, labels[0], ext)
            if not path.exists():
                r = c.get(m["url"])
                r.raise_for_status()
                path.write_bytes(r.content)
                time.sleep(0.6)
            overlap = bool(OVERLAP_HINT.search(" ".join((m["description"], m["credit"], m["artist"]))))
            entries.append({
                "file": path.as_posix(), "modality": modality, "labels": labels, "group": group or title,
                "source": "wikimedia_commons", "commons_title": title, "page": m["page"], "license": m["license"],
                "artist": m["artist"], "description": m["description"], "original_size": [m["orig_w"], m["orig_h"]],
                "possible_pretraining_overlap": overlap, "changes": f"downscaled to <= {MAX_WIDTH}px wide by Wikimedia's thumbnail service",
            })
    for path, modality, labels, group, lic, overlap in EXISTING:
        entries.append({
            "file": path, "modality": modality, "labels": labels, "group": group or path,
            "source": "data/images/real (see its ATTRIBUTION.md)", "license": lic,
            "possible_pretraining_overlap": overlap,
            "overlap_note": "from an open-access PubMed Central article" if overlap else "",
            "has_annotation": path.endswith("pneumothorax.png"),  # arrows on the film (a journal figure)
        })
    manifest = {
        "_provenance": {
            "labels": "taken from each file's Commons category/description; NOT radiologist-verified for this project",
            "selection": "explicit list in scripts/build_imaging_benchmark.py (clear teaching examples: optimistic vs. real practice)",
            "overlap": "possible_pretraining_overlap is a keyword heuristic on the file's own description; it can miss",
            "built": time.strftime("%Y-%m-%d"),
        },
        "images": entries,
    }
    keep = {Path(e["file"]).name for e in entries if e["source"] == "wikimedia_commons"}
    for stale in IMG_DIR.iterdir():
        if stale.name not in keep:
            stale.unlink()
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_attribution(entries)
    print(f"{len(entries)} images -> {MANIFEST}")


def write_attribution(entries: list[dict]) -> None:
    lines = [
        "# Imaging benchmark: image sources and licences", "",
        "Real images from Wikimedia Commons (plus the six already in `data/images/real/`), used only as test inputs.",
        "They are real people's scans, unidentified. None of this is a clinical dataset and none of it validates the system.", "",
        "Each file keeps its own licence. **CC BY and CC BY-SA require this attribution, the licence, and a statement of",
        "changes**: every Commons image here was **downscaled to at most 768 px wide by Wikimedia's thumbnail service**",
        "(no other change). CC BY-SA images stay under CC BY-SA. Licences were read from each file's Commons page.", "",
        "| File | Commons page | Author / credit | Licence |", "|---|---|---|---|",
    ]
    for e in entries:
        if e["source"] != "wikimedia_commons":
            continue
        who = (e["artist"] or "see Commons page").replace("|", "/")
        lines.append(f"| `{Path(e['file']).name}` | [{e['commons_title']}]({e['page']}) | {who} | {e['license']} |")
    lines += ["", "The six images from `data/images/real/` are credited in that folder's `ATTRIBUTION.md`.", ""]
    ATTRIBUTION.write_text("\n".join(lines), encoding="utf-8")


def check() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing = [e["file"] for e in manifest["images"] if not Path(e["file"]).exists()]
    for f in missing:
        print("MISSING", f)
    print(f"{len(manifest['images'])} manifest entries, {len(missing)} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.exit(check() if args.check else build())
