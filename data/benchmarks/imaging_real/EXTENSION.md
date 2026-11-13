# Extension set (17 images added 2026-11-13) - plan, written before any of them was scored

The first 62 images were used to choose the finding captions ("... showing ...", see PROMPT_EXPERIMENT.md); the
held-out half of that experiment was spent when it was run. These 17 images were added afterwards and have not been
scored by anything. They give (a) a second, fresh test of the adopted captions, and (b) coverage of gaps.

**What was added.** 4 normal chest X-rays; CT: 2 midline-shift films, 1 more infarct, 2 more slices of the one normal
head-CT subject; MRI: 3 slices of one vascular malformation, 1 chronic infarct, 1 edema (PRES), 1 white-matter
hyperintensity, 2 normal T2 axial. Selection used only each file's own Commons description and a visual check that the
image is clean (no arrows, panels or patient text). Labels are the uploader's, as before; none is radiologist-verified.

**Still not covered:** cerebral atrophy (MRI) and calcification (CT) - every usable candidate was a multi-panel composite
or carried arrows. There is still exactly ONE normal head-CT subject.

**Fixed before scoring.**
- Primary read: top-1 finding accuracy (true modality given) of the current system (`IMAGING_FINDING_PROMPTS=showing`)
  on the 17 images, with its Wilson interval and the chance baseline.
- Secondary: the same images with the bare labels (`IMAGING_FINDING_PROMPTS=label`) and the paired comparison
  (images gained / lost, exact McNemar); modality detection; specificity on the 8 normal images; per-image misses reported.
- No decision depends on it: nothing is re-tuned from these results, and the prompt experiment stays frozen on the
  original 62 (the loader's `only="original"`). With n=17 an interval is about +-22 points wide.
- Prior expectation (a guess, stated so it can be wrong): top-1 near the 56% seen on the original set; specificity is the
  number to watch, because the original experiment's cost was normals being called "sinus abnormality".

---

# Result (written after scoring; the plan above was committed first, in e3c4974)

Scored with the unchanged system. Nothing was re-tuned.

| 17 extension images | current captions (`showing`) | bare labels (`label`) |
|---|---|---|
| top-1 | 10/17 = 59% (Wilson 36-78%) | 11/17 = 65% (41-83%) |
| top-3 | 15/17 | 12/17 |
| specificity (8 normal images) | 6/8 | 8/8 |
| sensitivity (9 abnormal images) | 8/9 | 6/9 |
| modality detected | 16/17 | 16/17 |

Chance is 11% and always predicting the commonest label gets 53% (the 9 normal-film labels dominate), so top-1 is not
impressive here. **Paired, the captions gained 3 images and lost 4 (exact McNemar p = 1.0): on fresh images the captions
are no better than the bare labels.** The advantage seen on the original set (TEST half 12/25 against 7/25, p = 0.18) was
not reproduced. Neither result is significant; n = 17 cannot show a difference of this size either way.

What does hold up is the **trade-off**. Pooled over all 79 images, the captions give 45/79 against 37/79 top-1 and
sensitivity 59/62 against 54/62, but specificity 11/17 against 15/17: they find more abnormalities and also call more
normal films abnormal (a normal chest X-ray as "hyperinflation", a normal sagittal head CT as "mass lesion"). The pooled
top-1 difference (21 gained, 13 lost, p = 0.23) mixes in the tuning set and is not an unbiased estimate; only the 17 are.

Misses with the current captions: a normal chest film read as emphysema and another as "no significant abnormality" (a
label with no chest-film images); a CT infarct read as midline shift; one normal sagittal CT read as mass lesion; one CT
(subfalcine herniation) detected as MRI; the vascular malformation 1/3 (a GRE slice read as hemorrhage, a coronal T1 slice as
normal); the chronic lacunar infarct, whose lesion is subtle, read as "sinus abnormality" at a low score (0.24). Edema (PRES)
was correct (that image is flagged as possible training overlap).

Caveats specific to this set: two extension images (the normal head-CT slices) come from the same subject as the original
normal head CTs, so they are not fully independent of the tuning set; three images are slices of one malformation (one
group); labels are the uploader's, not a radiologist's; and the PRES image is flagged for possible pretraining overlap.

**Decision.** The captions stay (the pre-registered rule was applied once and this set is not a re-run of it), but the claim
is now weaker: "more sensitive, less specific, net top-1 gain not confirmed on fresh images". `IMAGING_FINDING_PROMPTS=label`
remains the one-line way back, and which wording is better for this project depends on whether a missed abnormality or a
false alarm is the costlier error, which is a clinical judgement, not something these 79 images settle.
