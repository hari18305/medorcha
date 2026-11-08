# Imaging mini-benchmark (Phase 17 Mode 3)

The images in this directory are **programmatically generated placeholder
graphics** (solid color + a simple outlined ellipse, seeded per intended
label) produced by
`backend.tests.run_evaluation.generate_imaging_eval_set()`. They are
**not real patient data, not real scans**, and not derived from any real
imaging study — same spirit as `data/images/README.md`'s existing demo
images, just one per label in `FINDING_LABELS_BY_MODALITY` instead of a
handful of general demo images.

`manifest.json` records, for each image, the modality and finding label
it was generated for. `run_imaging_benchmark()` scores
`analyze_image()`'s top-1 output against that manifest per modality;
`run_image_fusion_comparison()` uses a small sample of these images to
demonstrate that `fuse()`'s output genuinely differs when the raw image
is attached vs. text-labels-only (Phase 5's joint multimodal fusion).

**All three imaging modalities are covered** — MRI brain scan, CT scan,
and chest X-ray, each with its own real findings model (see
`backend/agents/imaging_labels.py`'s `FINDING_LABELS_BY_MODALITY`). Both
the generator and the benchmark are written generically against that
dict, so a future fourth modality would be picked up automatically, with
no code changes needed here.

**Accuracy numbers from this set are a synthetic construction-validity
check, not a claim of real diagnostic accuracy on real scans** — these
images are simple colored shapes, not radiology images, so BiomedCLIP's
top-1 accuracy against them says only "does the pipeline mechanism run
and score correctly end to end," not "how good is this at reading a
real scan."

Regenerate with: `python -c "from backend.tests.run_evaluation import generate_imaging_eval_set; generate_imaging_eval_set()"`
