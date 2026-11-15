# Evaluation and benchmarks

How the system was measured, and how to repeat it. Read the caveats before quoting any number: these are small,
developer-labelled sets that characterise the components, not clinical accuracy.

## Evaluation harness (`backend/tests/run_evaluation.py`)

```bash
python -m backend.tests.run_evaluation              # Mode 1: synthetic in-house cases
python -m backend.tests.run_evaluation --mode2       # Mode 2: real MedQA/PubMedQA subset (needs `pip install datasets` + network)
python -m backend.tests.run_evaluation --mode3       # Mode 3: real-image imaging benchmark + with/without-raw-image fusion comparison
python -m backend.tests.run_evaluation --case-memory-ablation  # required ablation: with vs. without text case memory
python -m backend.tests.run_evaluation --full        # everything above in one run
```

## Imaging benchmark (real images)

```bash
python -m scripts.build_imaging_benchmark    # (re)download the ~56 open-licence Commons images + manifest + attribution
python -m backend.tests.imaging_benchmark    # score BiomedCLIP on them -> data/benchmarks/imaging_real/RESULTS.md
```

62 real chest X-rays, head CTs and brain MRIs with a finding label each (the 56 are
downloaded from Wikimedia Commons under CC0 / CC BY / CC BY-SA / public domain, the
other 6 are the images in `data/images/real/`). It replaced an earlier benchmark that
scored the classifier on generated gradient pictures with an arbitrary "intended label"
(chance by construction). Results carry Wilson 95% intervals and chance / majority
baselines; **read the caveats at the top of [`RESULTS.md`](../data/benchmarks/imaging_real/RESULTS.md) before quoting a number**: a few
images per finding, labels from the uploader and not a radiologist, clear teaching
examples (optimistic), possible overlap with BiomedCLIP's PubMed Central training data. It
characterises one zero-shot model, not the system's diagnostic accuracy.
