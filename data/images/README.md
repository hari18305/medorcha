# Synthetic demo images

The `synthetic_demo_*` files in this directory are **programmatically
generated placeholder graphics** — schematic anatomy drawn with
numpy/Pillow — used only to exercise the imaging pipeline end to end:

| Files | Intended modality | What is drawn |
|---|---|---|
| `synthetic_demo_brain_mri_001/002` | MRI brain scan | skull, bright CSF, gray/white-matter texture, bright ventricles |
| `synthetic_demo_ct_scan_001/002/003` | CT scan | bright skull ring, mid-gray brain, dark ventricles, falx |
| `synthetic_demo_chest_xray_001/002/003` | chest X-ray | dark lung fields with vessel markings, ribs, clavicles, heart, spine |
| `synthetic_demo_unspecified_001` | none (unsupported) | a smooth abstract colour image — deliberately *not* a medical modality, so the imaging agent reports "unsupported modality" |

They are **not real patient data**, not real scans, and not derived from
any real imaging study. Do not treat their filenames or content as
clinically meaningful.

## How they are made, and why that matters

`python scripts/generate_demo_images.py` regenerates all of them
deterministically (fixed seeds); `--check` then classifies each with the
**real** BiomedCLIP and fails if one is not classified as its intended
modality.

The earlier placeholders were flat shapes, and the real classifier
labelled them essentially at random — the brain-MRI placeholder came back
as **"CT scan"**, and two of the three chest X-rays as "unspecified" or CT.
So any test case using them did not exercise the modality its filename
claimed. BiomedCLIP's zero-shot modality guess turned out to depend on an
image's tonal distribution far more than on anatomy, so each recipe applies
the histogram treatment (equalisation, with a gamma for chest X-rays) that
makes the image classify correctly. The grainy look of the MRI and chest
images is a side-effect of that equalisation.

**What a passing `--check` does and does not mean:** it shows these
*fixtures* route through their intended code path (the right label set, the
right retrieval filters). It says nothing about BiomedCLIP's accuracy on real
scans, which is unmeasured here. Real public-dataset images would be needed
for that.
