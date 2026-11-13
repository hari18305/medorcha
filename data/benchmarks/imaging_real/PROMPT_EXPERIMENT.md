# Prompt-wording experiment (held-out)

Split by group, seed 20261113: DEV 37 images / 35 groups, TEST 25 / 25. Candidates, split, selection rule and adoption rule were fixed in `imaging_prompt_experiment.py` before any run. A change is adopted only if it gains >= 2 top-1 images on TEST and does not lower MRR.

## DEV (used to choose)

### Finding prompts (true modality given)

| candidate | top-1 | top-3 | MRR |
|---|---|---|---|
| T0_raw_label | 19/37 | 25/37 | 0.6318 |
| T1_showing **<- chosen** | 23/37 | 32/37 | 0.7571 |
| T2_this_is | 15/37 | 32/37 | 0.6424 |
| T3_description_only | 17/37 | 28/37 | 0.6374 |
| T4_ensemble | 21/37 | 33/37 | 0.7298 |

### Modality prompts

| candidate | detected correctly |
|---|---|
| M0_raw_label | 34/37 |
| M1_descriptive | 36/37 |
| M2_this_is **<- chosen** | 37/37 |
| M3_ensemble | 37/37 |

## TEST (held out, looked at once)

### Finding prompts
Baseline `T0_raw_label` vs chosen `T1_showing`.

| | top-1 | top-3 | MRR |
|---|---|---|---|
| baseline | 7/25 | 15 | 0.4896 |
| chosen | 12/25 | 17 | 0.6115 |

Paired top-1: 7 images gained, 2 lost, McNemar exact p = 0.1797.

**Decision: ADOPT** - test top-1 7 -> 12 (gain 5, bar 2); MRR 0.4896 -> 0.6115; McNemar p=0.1797.

### Modality prompts
Baseline `M0_raw_label` vs chosen `M2_this_is`.

Test correct: baseline 24/25, chosen 25/25; paired 1 gained / 0 lost, p = 1.0.

**Decision: DO NOT ADOPT** - test modality correct 24 -> 25 of 25 (gain 1, bar 2); McNemar p=1.0.


## Limits

- I had seen full-set failure patterns before writing the descriptions, so candidates are informed by them (not tuned against a run).
- ~30 test images: only a large effect can clear the bar; 'not adopted' mostly means 'not shown to help'.
- Descriptions are a non-radiologist's; labels are the uploader's (see RESULTS.md).
