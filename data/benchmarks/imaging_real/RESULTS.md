# Imaging benchmark results (real images)

BiomedCLIP zero-shot (`imaging_agent.rank_image`), 62 real images in 60 groups (manifest built 2026-11-13). Produced by `python -m backend.tests.imaging_benchmark`.

## Read this first

- **Small.** A few images per finding, so every per-finding figure has a wide confidence interval (shown). MRI atrophy and CT midline shift had no usable image and are not covered; MRI white-matter hyperintensity has one.
- **Labels are the uploader's, not a radiologist's.** They come from each file's Wikimedia Commons category or caption. The developer who selected them is not a radiologist, and nobody re-read the films for this project.
- **Optimistic.** Commons holds clear, typical teaching examples; real clinical films are harder and messier. Accuracy here is an upper bound on what to expect, not an estimate of it.
- **Possible training overlap.** BiomedCLIP was trained on PubMed Central figure-caption pairs. Images from journal articles are flagged by a keyword heuristic on their own description, which can miss. See the 'excluding' rows.
- **Not independent.** Slices of one patient/series share a `group`; the group-collapsed figure counts each once. There is exactly ONE normal head-CT subject (three slices), so CT specificity is close to meaningless.
- **Not a clinical validation.** It characterises one zero-shot model's behaviour on a hand-picked set; it says nothing about the whole system's diagnostic accuracy.

## Headline (proportion, count, Wilson 95% interval)

| | images | modality detected | top-1 finding | top-3 | end-to-end top-1 | normal-vs-abnormal: sensitivity | specificity |
|---|---|---|---|---|---|---|---|
| **all** | 62 | 94% (58/62; 85%-97%) | 42% (26/62; 30%-54%) | 65% (40/62; 52%-75%) | 40% (25/62; 29%-53%) | 91% (48/53; 80%-96%) | 78% (7/9; 45%-94%) |
| MRI brain scan | 12 | 100% (12/12; 76%-100%) | 25% (3/12; 9%-53%) | 58% (7/12; 32%-81%) | 25% (3/12; 9%-53%) | 100% (10/10; 72%-100%) | 50% (1/2; 9%-91%) |
| CT scan | 18 | 78% (14/18; 55%-91%) | 33% (6/18; 16%-56%) | 56% (10/18; 34%-75%) | 28% (5/18; 12%-51%) | 73% (11/15; 48%-89%) | 100% (3/3; 44%-100%) |
| chest X-ray | 32 | 100% (32/32; 89%-100%) | 53% (17/32; 36%-69%) | 72% (23/32; 55%-84%) | 53% (17/32; 36%-69%) | 96% (27/28; 82%-99%) | 75% (3/4; 30%-95%) |

Top-1 / top-3 use the TRUE modality's label set (isolating the findings model); end-to-end uses the modality the pipeline detects, so a wrong modality counts as wrong.

## Against baselines

| | top-1 | chance top-1 | always-the-commonest-label | top-3 | chance top-3 | group-collapsed top-1 |
|---|---|---|---|---|---|---|
| **all** | 42% (26/62; 30%-54%) | 10% | 21% | 65% (40/62; 52%-75%) | 28% | 40% |
| MRI brain scan | 25% (3/12; 9%-53%) | 10% | 33% | 58% (7/12; 32%-81%) | 28% | 25% |
| CT scan | 33% (6/18; 16%-56%) | 12% | 28% | 56% (10/18; 34%-75%) | 34% | 25% |
| chest X-ray | 53% (17/32; 36%-69%) | 9% | 12% | 72% (23/32; 55%-84%) | 26% | 53% |

## What the pipeline would actually pass downstream

| | reported findings contain the truth | also reports a wrong finding | mean findings reported |
|---|---|---|---|
| **all** | 45% (28/62; 33%-57%) | 63% (39/62; 50%-74%) | 1.19 |
| MRI brain scan | 25% (3/12; 9%-53%) | 83% (10/12; 55%-95%) | 1.17 |
| CT scan | 39% (7/18; 20%-61%) | 67% (12/18; 44%-84%) | 1.06 |
| chest X-ray | 56% (18/32; 39%-72%) | 53% (17/32; 36%-69%) | 1.28 |

## Sensitivity to the weaknesses above (top-1)

| subset | images | top-1 | modality detected |
|---|---|---|---|
| all | 62 | 42% (26/62; 30%-54%) | 94% (58/62; 85%-97%) |
| excluding possible pretraining overlap | 60 | 42% (25/60; 30%-54%) | 93% (56/60; 84%-97%) |
| excluding annotated films | 61 | 41% (25/61; 30%-54%) | 93% (57/61; 84%-97%) |
| excluding both | 60 | 42% (25/60; 30%-54%) | 93% (56/60; 84%-97%) |

## Per finding

| modality | finding | n | top-1 | top-3 | most common wrong top-1 |
|---|---|---|---|---|---|
| CT scan | acute infarct | 3 | 2/3 | 3/3 | ventricular enlargement x1 |
| CT scan | hemorrhage | 5 | 0/5 | 2/5 | acute infarct x4; normal head CT, no acute findings x1 |
| CT scan | mass lesion | 3 | 0/3 | 0/3 | ventricular enlargement x1; acute infarct x1 |
| CT scan | normal head CT, no acute findings | 3 | 3/3 | 3/3 | - |
| CT scan | sinus abnormality | 2 | 0/2 | 0/2 | normal head CT, no acute findings x1; hypodense lesion x1 |
| CT scan | ventricular enlargement | 2 | 1/2 | 2/2 | normal head CT, no acute findings x1 |
| MRI brain scan | acute infarct | 2 | 1/2 | 2/2 | white matter hyperintensity x1 |
| MRI brain scan | hemorrhage | 2 | 0/2 | 0/2 | cerebral atrophy x2 |
| MRI brain scan | mass lesion | 4 | 0/4 | 2/4 | acute infarct x3; chronic infarct x1 |
| MRI brain scan | normal brain MRI, no acute findings | 2 | 1/2 | 2/2 | cerebral atrophy x1 |
| MRI brain scan | ventricular enlargement | 1 | 0/1 | 0/1 | acute infarct x1 |
| MRI brain scan | white matter hyperintensity | 1 | 1/1 | 1/1 | - |
| chest X-ray | atelectasis | 3 | 2/3 | 3/3 | pleural effusion x1 |
| chest X-ray | cardiomegaly | 3 | 2/3 | 2/3 | atelectasis x1 |
| chest X-ray | consolidation | 3 | 0/3 | 0/3 | pulmonary edema x3 |
| chest X-ray | hilar or mediastinal widening | 1 | 1/1 | 1/1 | - |
| chest X-ray | hyperinflation consistent with emphysema | 2 | 0/2 | 0/2 | rib fracture x1; atelectasis x1 |
| chest X-ray | normal chest X-ray, no acute findings | 4 | 3/4 | 4/4 | hilar or mediastinal widening x1 |
| chest X-ray | pleural effusion | 3 | 2/3 | 3/3 | atelectasis x1 |
| chest X-ray | pneumothorax | 4 | 3/4 | 3/4 | cardiomegaly x1 |
| chest X-ray | pulmonary edema | 3 | 3/3 | 3/3 | - |
| chest X-ray | pulmonary nodule or mass | 3 | 0/3 | 3/3 | atelectasis x1; pulmonary edema x1 |
| chest X-ray | rib fracture | 3 | 1/3 | 1/3 | no significant abnormality x1; pneumothorax x1 |

## Modality detection (rows = truth, columns = detected)

- **chest X-ray**: chest X-ray x32
- **CT scan**: CT scan x14, MRI brain scan x4
- **MRI brain scan**: MRI brain scan x12
