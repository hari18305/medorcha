# Imaging benchmark results (real images)

BiomedCLIP zero-shot (`imaging_agent.rank_image`), 79 real images in 73 groups (manifest built 2026-11-13). Produced by `python -m backend.tests.imaging_benchmark`.

## Read this first

- **Small.** A few images per finding, so every per-finding figure has a wide confidence interval (shown). MRI atrophy and CT midline shift had no usable image and are not covered; MRI white-matter hyperintensity has one.
- **Labels are the uploader's, not a radiologist's.** They come from each file's Wikimedia Commons category or caption. The developer who selected them is not a radiologist, and nobody re-read the films for this project.
- **Optimistic.** Commons holds clear, typical teaching examples; real clinical films are harder and messier. Accuracy here is an upper bound on what to expect, not an estimate of it.
- **Possible training overlap.** BiomedCLIP was trained on PubMed Central figure-caption pairs. Images from journal articles are flagged by a keyword heuristic on their own description, which can miss. See the 'excluding' rows.
- **Not independent.** Slices of one patient/series share a `group`; the group-collapsed figure counts each once. There is exactly ONE normal head-CT subject (five slices), so CT specificity is close to meaningless.
- **The finding prompts were tuned on this set.** The wording was chosen on the DEV half and judged once on the TEST half (PROMPT_EXPERIMENT.md), so the all-image figures are partly in-sample. Quote the TEST-half row (see 'Sensitivity to the weaknesses') as the unbiased read; it is ~25 images. The 17-image EXTENSION set (EXTENSION.md) was added afterwards and never used for tuning; it did **not** reproduce the captions' advantage (3 gained, 4 lost against bare labels).
- **Not a clinical validation.** It characterises one zero-shot model's behaviour on a hand-picked set; it says nothing about the whole system's diagnostic accuracy.

## Headline (proportion, count, Wilson 95% interval)

| | images | modality detected | top-1 finding | top-3 | end-to-end top-1 | normal-vs-abnormal: sensitivity | specificity |
|---|---|---|---|---|---|---|---|
| **all** | 79 | 94% (74/79; 86%-97%) | 57% (45/79; 46%-67%) | 81% (64/79; 71%-88%) | 53% (42/79; 42%-64%) | 95% (59/62; 87%-98%) | 65% (11/17; 41%-83%) |
| MRI brain scan | 20 | 100% (20/20; 84%-100%) | 60% (12/20; 39%-78%) | 85% (17/20; 64%-95%) | 60% (12/20; 39%-78%) | 94% (15/16; 72%-99%) | 50% (2/4; 15%-85%) |
| CT scan | 23 | 78% (18/23; 58%-90%) | 48% (11/23; 29%-67%) | 83% (19/23; 63%-93%) | 35% (8/23; 19%-55%) | 89% (16/18; 67%-97%) | 40% (2/5; 12%-77%) |
| chest X-ray | 36 | 100% (36/36; 90%-100%) | 61% (22/36; 45%-75%) | 78% (28/36; 62%-88%) | 61% (22/36; 45%-75%) | 100% (28/28; 88%-100%) | 88% (7/8; 53%-98%) |

Top-1 / top-3 use the TRUE modality's label set (isolating the findings model); end-to-end uses the modality the pipeline detects, so a wrong modality counts as wrong.

## Against baselines

| | top-1 | chance top-1 | always-the-commonest-label | top-3 | chance top-3 | group-collapsed top-1 |
|---|---|---|---|---|---|---|
| **all** | 57% (45/79; 46%-67%) | 10% | 23% | 81% (64/79; 71%-88%) | 29% | 59% |
| MRI brain scan | 60% (12/20; 39%-78%) | 10% | 20% | 85% (17/20; 64%-95%) | 28% | 63% |
| CT scan | 48% (11/23; 29%-67%) | 13% | 26% | 83% (19/23; 63%-93%) | 36% | 49% |
| chest X-ray | 61% (22/36; 45%-75%) | 9% | 22% | 78% (28/36; 62%-88%) | 26% | 61% |

## What the pipeline would actually pass downstream

| | reported findings contain the truth | also reports a wrong finding | mean findings reported |
|---|---|---|---|
| **all** | 61% (48/79; 50%-71%) | 49% (39/79; 39%-60%) | 1.25 |
| MRI brain scan | 70% (14/20; 48%-85%) | 50% (10/20; 30%-70%) | 1.45 |
| CT scan | 48% (11/23; 29%-67%) | 52% (12/23; 33%-71%) | 1.17 |
| chest X-ray | 64% (23/36; 48%-78%) | 47% (17/36; 32%-63%) | 1.19 |

## Sensitivity to the weaknesses above (top-1)

| subset | images | top-1 | modality detected |
|---|---|---|---|
| all | 79 | 57% (45/79; 46%-67%) | 94% (74/79; 86%-97%) |
| excluding possible pretraining overlap | 76 | 55% (42/76; 44%-66%) | 93% (71/76; 86%-97%) |
| excluding annotated films | 78 | 56% (44/78; 45%-67%) | 94% (73/78; 86%-97%) |
| excluding both | 76 | 55% (42/76; 44%-66%) | 93% (71/76; 86%-97%) |
| ORIGINAL 62 images (the set the prompts were tuned on) | 62 | 56% (35/62; 44%-68%) | 94% (58/62; 85%-97%) |
| EXTENSION set (added after the prompt choice, never used for tuning) | 17 | 59% (10/17; 36%-78%) | 94% (16/17; 73%-99%) |
| TEST half: held out from the prompt choice | 25 | 48% (12/25; 30%-67%) | 96% (24/25; 80%-99%) |
| DEV half: used to choose the prompt (optimistic) | 37 | 62% (23/37; 46%-76%) | 92% (34/37; 79%-97%) |

## Per finding

| modality | finding | n | top-1 | top-3 | most common wrong top-1 |
|---|---|---|---|---|---|
| CT scan | acute infarct | 4 | 1/4 | 4/4 | midline shift x2; ventricular enlargement x1 |
| CT scan | hemorrhage | 5 | 3/5 | 4/5 | midline shift x1; calcification x1 |
| CT scan | mass lesion | 3 | 0/3 | 1/3 | hypodense lesion x1; ventricular enlargement x1 |
| CT scan | midline shift | 2 | 2/2 | 2/2 | - |
| CT scan | normal head CT, no acute findings | 5 | 2/5 | 5/5 | sinus abnormality x2; mass lesion x1 |
| CT scan | sinus abnormality | 2 | 2/2 | 2/2 | - |
| CT scan | ventricular enlargement | 2 | 1/2 | 1/2 | normal head CT, no acute findings x1 |
| MRI brain scan | acute infarct | 2 | 2/2 | 2/2 | - |
| MRI brain scan | chronic infarct | 1 | 0/1 | 1/1 | sinus abnormality x1 |
| MRI brain scan | edema | 1 | 1/1 | 1/1 | - |
| MRI brain scan | hemorrhage | 2 | 1/2 | 1/2 | cerebral atrophy x1 |
| MRI brain scan | mass lesion | 4 | 4/4 | 4/4 | - |
| MRI brain scan | normal brain MRI, no acute findings | 4 | 2/4 | 4/4 | sinus abnormality x1; cerebral atrophy x1 |
| MRI brain scan | vascular malformation | 3 | 1/3 | 2/3 | hemorrhage x1; normal brain MRI, no acute findings x1 |
| MRI brain scan | ventricular enlargement | 1 | 0/1 | 0/1 | hemorrhage x1 |
| MRI brain scan | white matter hyperintensity | 2 | 1/2 | 2/2 | cerebral atrophy x1 |
| chest X-ray | atelectasis | 3 | 0/3 | 1/3 | pleural effusion x2; hyperinflation consistent with emphysema x1 |
| chest X-ray | cardiomegaly | 3 | 3/3 | 3/3 | - |
| chest X-ray | consolidation | 3 | 0/3 | 3/3 | pulmonary nodule or mass x2; pulmonary edema x1 |
| chest X-ray | hilar or mediastinal widening | 1 | 1/1 | 1/1 | - |
| chest X-ray | hyperinflation consistent with emphysema | 2 | 2/2 | 2/2 | - |
| chest X-ray | normal chest X-ray, no acute findings | 8 | 3/8 | 4/8 | no significant abnormality x4; hyperinflation consistent with emphysema x1 |
| chest X-ray | pleural effusion | 3 | 3/3 | 3/3 | - |
| chest X-ray | pneumothorax | 4 | 3/4 | 3/4 | cardiomegaly x1 |
| chest X-ray | pulmonary edema | 3 | 3/3 | 3/3 | - |
| chest X-ray | pulmonary nodule or mass | 3 | 2/3 | 3/3 | hilar or mediastinal widening x1 |
| chest X-ray | rib fracture | 3 | 2/3 | 2/3 | pneumothorax x1 |

## Modality detection (rows = truth, columns = detected)

- **chest X-ray**: chest X-ray x36
- **CT scan**: CT scan x18, MRI brain scan x5
- **MRI brain scan**: MRI brain scan x20
