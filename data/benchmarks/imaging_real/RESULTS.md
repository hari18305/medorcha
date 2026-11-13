# Imaging benchmark results (real images)

BiomedCLIP zero-shot (`imaging_agent.rank_image`), 88 real images in 81 groups (manifest built 2026-11-13). Produced by `python -m backend.tests.imaging_benchmark`.

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
| **all** | 88 | 94% (83/88; 87%-98%) | 57% (50/88; 46%-67%) | 82% (72/88; 72%-88%) | 53% (47/88; 43%-63%) | 94% (63/67; 86%-98%) | 57% (12/21; 37%-76%) |
| MRI brain scan | 23 | 100% (23/23; 86%-100%) | 61% (14/23; 41%-78%) | 87% (20/23; 68%-95%) | 61% (14/23; 41%-78%) | 94% (16/17; 73%-99%) | 50% (3/6; 19%-81%) |
| CT scan | 28 | 82% (23/28; 64%-92%) | 50% (14/28; 33%-67%) | 86% (24/28; 69%-94%) | 39% (11/28; 24%-58%) | 86% (19/22; 67%-95%) | 33% (2/6; 10%-70%) |
| chest X-ray | 37 | 100% (37/37; 91%-100%) | 59% (22/37; 43%-74%) | 76% (28/37; 60%-87%) | 59% (22/37; 43%-74%) | 100% (28/28; 88%-100%) | 78% (7/9; 45%-94%) |

Top-1 / top-3 use the TRUE modality's label set (isolating the findings model); end-to-end uses the modality the pipeline detects, so a wrong modality counts as wrong.

## Against baselines

| | top-1 | chance top-1 | always-the-commonest-label | top-3 | chance top-3 | group-collapsed top-1 |
|---|---|---|---|---|---|---|
| **all** | 57% (50/88; 46%-67%) | 10% | 24% | 82% (72/88; 72%-88%) | 30% | 59% |
| MRI brain scan | 61% (14/23; 41%-78%) | 10% | 26% | 87% (20/23; 68%-95%) | 30% | 63% |
| CT scan | 50% (14/28; 33%-67%) | 12% | 21% | 86% (24/28; 69%-94%) | 35% | 52% |
| chest X-ray | 59% (22/37; 43%-74%) | 9% | 24% | 76% (28/37; 60%-87%) | 26% | 61% |

## What the pipeline would actually pass downstream

| | reported findings contain the truth | also reports a wrong finding | mean findings reported |
|---|---|---|---|
| **all** | 61% (54/88; 51%-71%) | 50% (44/88; 40%-60%) | 1.26 |
| MRI brain scan | 70% (16/23; 49%-84%) | 48% (11/23; 29%-67%) | 1.39 |
| CT scan | 54% (15/28; 36%-70%) | 54% (15/28; 36%-70%) | 1.25 |
| chest X-ray | 62% (23/37; 46%-76%) | 49% (18/37; 33%-64%) | 1.19 |

## Sensitivity to the weaknesses above (top-1)

| subset | images | top-1 | modality detected |
|---|---|---|---|
| all | 88 | 57% (50/88; 46%-67%) | 94% (83/88; 87%-98%) |
| excluding possible pretraining overlap | 85 | 55% (47/85; 45%-65%) | 94% (80/85; 87%-97%) |
| excluding annotated films | 87 | 56% (49/87; 46%-66%) | 94% (82/87; 87%-98%) |
| excluding both | 85 | 55% (47/85; 45%-65%) | 94% (80/85; 87%-97%) |
| ORIGINAL 62 images (the set the prompts were tuned on) | 62 | 56% (35/62; 44%-68%) | 94% (58/62; 85%-97%) |
| EXTENSION set (added after the prompt choice, never used for tuning) | 17 | 59% (10/17; 36%-78%) | 94% (16/17; 73%-99%) |
| EXTENSION 2 set (added later still; plan fixed before scoring) | 9 | 56% (5/9; 27%-81%) | 100% (9/9; 70%-100%) |
| TEST half: held out from the prompt choice | 25 | 48% (12/25; 30%-67%) | 96% (24/25; 80%-99%) |
| DEV half: used to choose the prompt (optimistic) | 37 | 62% (23/37; 46%-76%) | 92% (34/37; 79%-97%) |

## Per finding

| modality | finding | n | top-1 | top-3 | most common wrong top-1 |
|---|---|---|---|---|---|
| CT scan | acute infarct | 4 | 1/4 | 4/4 | midline shift x2; ventricular enlargement x1 |
| CT scan | calcification | 4 | 3/4 | 4/4 | no significant abnormality x1 |
| CT scan | hemorrhage | 5 | 3/5 | 4/5 | midline shift x1; calcification x1 |
| CT scan | mass lesion | 3 | 0/3 | 1/3 | hypodense lesion x1; ventricular enlargement x1 |
| CT scan | midline shift | 2 | 2/2 | 2/2 | - |
| CT scan | normal head CT, no acute findings | 6 | 2/6 | 6/6 | sinus abnormality x3; mass lesion x1 |
| CT scan | sinus abnormality | 2 | 2/2 | 2/2 | - |
| CT scan | ventricular enlargement | 2 | 1/2 | 1/2 | normal head CT, no acute findings x1 |
| MRI brain scan | acute infarct | 2 | 2/2 | 2/2 | - |
| MRI brain scan | cerebral atrophy | 1 | 1/1 | 1/1 | - |
| MRI brain scan | chronic infarct | 1 | 0/1 | 1/1 | sinus abnormality x1 |
| MRI brain scan | edema | 1 | 1/1 | 1/1 | - |
| MRI brain scan | hemorrhage | 2 | 1/2 | 1/2 | cerebral atrophy x1 |
| MRI brain scan | mass lesion | 4 | 4/4 | 4/4 | - |
| MRI brain scan | normal brain MRI, no acute findings | 6 | 3/6 | 6/6 | cerebral atrophy x2; sinus abnormality x1 |
| MRI brain scan | vascular malformation | 3 | 1/3 | 2/3 | hemorrhage x1; normal brain MRI, no acute findings x1 |
| MRI brain scan | ventricular enlargement | 1 | 0/1 | 0/1 | hemorrhage x1 |
| MRI brain scan | white matter hyperintensity | 2 | 1/2 | 2/2 | cerebral atrophy x1 |
| chest X-ray | atelectasis | 3 | 0/3 | 1/3 | pleural effusion x2; hyperinflation consistent with emphysema x1 |
| chest X-ray | cardiomegaly | 3 | 3/3 | 3/3 | - |
| chest X-ray | consolidation | 3 | 0/3 | 3/3 | pulmonary nodule or mass x2; pulmonary edema x1 |
| chest X-ray | hilar or mediastinal widening | 1 | 1/1 | 1/1 | - |
| chest X-ray | hyperinflation consistent with emphysema | 2 | 2/2 | 2/2 | - |
| chest X-ray | normal chest X-ray, no acute findings | 9 | 3/9 | 4/9 | no significant abnormality x4; hyperinflation consistent with emphysema x1 |
| chest X-ray | pleural effusion | 3 | 3/3 | 3/3 | - |
| chest X-ray | pneumothorax | 4 | 3/4 | 3/4 | cardiomegaly x1 |
| chest X-ray | pulmonary edema | 3 | 3/3 | 3/3 | - |
| chest X-ray | pulmonary nodule or mass | 3 | 2/3 | 3/3 | hilar or mediastinal widening x1 |
| chest X-ray | rib fracture | 3 | 2/3 | 2/3 | pneumothorax x1 |

## Modality detection (rows = truth, columns = detected)

- **chest X-ray**: chest X-ray x37
- **CT scan**: CT scan x23, MRI brain scan x5
- **MRI brain scan**: MRI brain scan x23
