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
