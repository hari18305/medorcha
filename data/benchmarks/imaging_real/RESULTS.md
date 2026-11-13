# Imaging benchmark results (real images)

BiomedCLIP zero-shot (`imaging_agent.rank_image`), 62 real images in 60 groups (manifest built 2026-11-13). Produced by `python -m backend.tests.imaging_benchmark`.

## Read this first

- **Small.** A few images per finding, so every per-finding figure has a wide confidence interval (shown). MRI atrophy and CT midline shift had no usable image and are not covered; MRI white-matter hyperintensity has one.
- **Labels are the uploader's, not a radiologist's.** They come from each file's Wikimedia Commons category or caption. The developer who selected them is not a radiologist, and nobody re-read the films for this project.
- **Optimistic.** Commons holds clear, typical teaching examples; real clinical films are harder and messier. Accuracy here is an upper bound on what to expect, not an estimate of it.
- **Possible training overlap.** BiomedCLIP was trained on PubMed Central figure-caption pairs. Images from journal articles are flagged by a keyword heuristic on their own description, which can miss. See the 'excluding' rows.
- **Not independent.** Slices of one patient/series share a `group`; the group-collapsed figure counts each once. There is exactly ONE normal head-CT subject (three slices), so CT specificity is close to meaningless.
- **The finding prompts were tuned on this set.** The wording was chosen on the DEV half and judged once on the TEST half (PROMPT_EXPERIMENT.md), so the all-image figures are partly in-sample. Quote the TEST-half row (see 'Sensitivity to the weaknesses') as the unbiased read; it is ~25 images.
- **Not a clinical validation.** It characterises one zero-shot model's behaviour on a hand-picked set; it says nothing about the whole system's diagnostic accuracy.

## Headline (proportion, count, Wilson 95% interval)

| | images | modality detected | top-1 finding | top-3 | end-to-end top-1 | normal-vs-abnormal: sensitivity | specificity |
|---|---|---|---|---|---|---|---|
| **all** | 62 | 94% (58/62; 85%-97%) | 56% (35/62; 44%-68%) | 79% (49/62; 67%-87%) | 53% (33/62; 41%-65%) | 96% (51/53; 87%-99%) | 56% (5/9; 27%-81%) |
| MRI brain scan | 12 | 100% (12/12; 76%-100%) | 58% (7/12; 32%-81%) | 83% (10/12; 55%-95%) | 58% (7/12; 32%-81%) | 100% (10/10; 72%-100%) | 0% (0/2; 0%-66%) |
| CT scan | 18 | 78% (14/18; 55%-91%) | 44% (8/18; 25%-66%) | 78% (14/18; 55%-91%) | 33% (6/18; 16%-56%) | 87% (13/15; 62%-96%) | 33% (1/3; 6%-79%) |
| chest X-ray | 32 | 100% (32/32; 89%-100%) | 62% (20/32; 45%-77%) | 78% (25/32; 61%-89%) | 62% (20/32; 45%-77%) | 100% (28/28; 88%-100%) | 100% (4/4; 51%-100%) |

Top-1 / top-3 use the TRUE modality's label set (isolating the findings model); end-to-end uses the modality the pipeline detects, so a wrong modality counts as wrong.

## Against baselines

| | top-1 | chance top-1 | always-the-commonest-label | top-3 | chance top-3 | group-collapsed top-1 |
|---|---|---|---|---|---|---|
| **all** | 56% (35/62; 44%-68%) | 10% | 21% | 79% (49/62; 67%-87%) | 28% | 57% |
| MRI brain scan | 58% (7/12; 32%-81%) | 10% | 33% | 83% (10/12; 55%-95%) | 28% | 58% |
| CT scan | 44% (8/18; 25%-66%) | 12% | 28% | 78% (14/18; 55%-91%) | 34% | 46% |
| chest X-ray | 62% (20/32; 45%-77%) | 9% | 12% | 78% (25/32; 61%-89%) | 26% | 62% |

## What the pipeline would actually pass downstream

| | reported findings contain the truth | also reports a wrong finding | mean findings reported |
|---|---|---|---|
| **all** | 60% (37/62; 47%-71%) | 48% (30/62; 36%-61%) | 1.19 |
| MRI brain scan | 67% (8/12; 39%-86%) | 42% (5/12; 19%-68%) | 1.08 |
| CT scan | 44% (8/18; 25%-66%) | 56% (10/18; 34%-75%) | 1.22 |
| chest X-ray | 66% (21/32; 48%-80%) | 47% (15/32; 31%-64%) | 1.22 |

## Sensitivity to the weaknesses above (top-1)

| subset | images | top-1 | modality detected |
|---|---|---|---|
| all | 62 | 56% (35/62; 44%-68%) | 94% (58/62; 85%-97%) |
| excluding possible pretraining overlap | 60 | 55% (33/60; 42%-67%) | 93% (56/60; 84%-97%) |
| excluding annotated films | 61 | 56% (34/61; 43%-67%) | 93% (57/61; 84%-97%) |
| excluding both | 60 | 55% (33/60; 42%-67%) | 93% (56/60; 84%-97%) |
| TEST half: held out from the prompt choice | 25 | 48% (12/25; 30%-67%) | 96% (24/25; 80%-99%) |
| DEV half: used to choose the prompt (optimistic) | 37 | 62% (23/37; 46%-76%) | 92% (34/37; 79%-97%) |

## Per finding

| modality | finding | n | top-1 | top-3 | most common wrong top-1 |
|---|---|---|---|---|---|
| CT scan | acute infarct | 3 | 1/3 | 3/3 | midline shift x1; ventricular enlargement x1 |
| CT scan | hemorrhage | 5 | 3/5 | 4/5 | midline shift x1; calcification x1 |
| CT scan | mass lesion | 3 | 0/3 | 1/3 | hypodense lesion x1; ventricular enlargement x1 |
| CT scan | normal head CT, no acute findings | 3 | 1/3 | 3/3 | sinus abnormality x2 |
| CT scan | sinus abnormality | 2 | 2/2 | 2/2 | - |
| CT scan | ventricular enlargement | 2 | 1/2 | 1/2 | normal head CT, no acute findings x1 |
| MRI brain scan | acute infarct | 2 | 2/2 | 2/2 | - |
| MRI brain scan | hemorrhage | 2 | 1/2 | 1/2 | cerebral atrophy x1 |
| MRI brain scan | mass lesion | 4 | 4/4 | 4/4 | - |
| MRI brain scan | normal brain MRI, no acute findings | 2 | 0/2 | 2/2 | sinus abnormality x1; cerebral atrophy x1 |
| MRI brain scan | ventricular enlargement | 1 | 0/1 | 0/1 | hemorrhage x1 |
| MRI brain scan | white matter hyperintensity | 1 | 0/1 | 1/1 | cerebral atrophy x1 |
| chest X-ray | atelectasis | 3 | 0/3 | 1/3 | pleural effusion x2; hyperinflation consistent with emphysema x1 |
| chest X-ray | cardiomegaly | 3 | 3/3 | 3/3 | - |
| chest X-ray | consolidation | 3 | 0/3 | 3/3 | pulmonary nodule or mass x2; pulmonary edema x1 |
| chest X-ray | hilar or mediastinal widening | 1 | 1/1 | 1/1 | - |
| chest X-ray | hyperinflation consistent with emphysema | 2 | 2/2 | 2/2 | - |
| chest X-ray | normal chest X-ray, no acute findings | 4 | 1/4 | 1/4 | no significant abnormality x3 |
| chest X-ray | pleural effusion | 3 | 3/3 | 3/3 | - |
| chest X-ray | pneumothorax | 4 | 3/4 | 3/4 | cardiomegaly x1 |
| chest X-ray | pulmonary edema | 3 | 3/3 | 3/3 | - |
| chest X-ray | pulmonary nodule or mass | 3 | 2/3 | 3/3 | hilar or mediastinal widening x1 |
| chest X-ray | rib fracture | 3 | 2/3 | 2/3 | pneumothorax x1 |

## Modality detection (rows = truth, columns = detected)

- **chest X-ray**: chest X-ray x32
- **CT scan**: CT scan x14, MRI brain scan x4
- **MRI brain scan**: MRI brain scan x12
