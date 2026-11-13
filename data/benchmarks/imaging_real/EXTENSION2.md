# Extension 2 (9 images added 2026-11-13) - plan, written before any of them was scored

Added after the first extension (EXTENSION.md) had been scored. The system is unchanged; nothing is tuned.

**What was added, and why.** The first 79 images had no CT calcification and no MRI cerebral atrophy, only one
normal head-CT subject (and two slices of it in the extension) and few normal films.
- CT calcification: 4 images (two basal-ganglia calcifications of the elderly, an extensive hyperparathyroid pattern,
  a Sturge-Weber film).
- MRI cerebral atrophy: 1 image (T2 axial, ex-vacuo ventricles and widened sulci; labels: cerebral atrophy, or
  ventricular enlargement).
- Normals: a second head-CT subject (1), two normal brain MRIs (axial T2, sagittal T1), one lateral chest film
  (the same subject as an existing PA film, so the same group).

**What is still missing.** There is only ONE atrophy image and no CT atrophy at all: the CT label set has no atrophy
label, and every other candidate on Commons was a multi-panel composite, annotated with arrows, or the wrong kind of
atrophy (cerebellar). A single image cannot say anything about atrophy; it is there so the label is not empty. The
"second normal head-CT subject" is one slice. Selection used only each file's own Commons description plus a visual
check that the image is clean. Labels are the uploader's, not a radiologist's. Two calcification films are described as
normal age-related calcification of the globus pallidus; they are still labelled "calcification" because that is what is
visible.

**Fixed before scoring.** Primary read: top-1 of the current captions on the 9 images, with the Wilson interval and the
paired bare-label comparison (images gained / lost). Secondary: specificity on the 4 normal images, and whether each
calcification film gets "calcification" as top-1 or top-3. No decision depends on it and nothing is re-tuned; with n = 9
an interval is about +-30 points wide. Prior expectation (a guess, stated so it can be wrong): calcification is a small
bright-spot finding that a whole-image zero-shot model often misses, so most of the 4 will not be top-1.
