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

---

# Result (written after scoring; the plan above was committed first, in 5c1e5c6)

Scored with the unchanged system; nothing re-tuned.

| 9 extension-2 images | current captions | bare labels |
|---|---|---|
| top-1 | 5/9 | 3/9 |
| top-3 | 8/9 | 5/9 |
| CT calcification (4 films) top-1 | 3/4 | 0/4 |
| cerebral atrophy (1 film) top-1 | 1/1 | 1/1 |
| normal films called normal (4) | 1/4 | 2/4 |
| abnormal films flagged (5) | 4/5 | 3/5 |

Paired, the captions gained 3 images and lost 1 (exact McNemar p = 0.63; not significant at n = 9). My stated expectation
was that most calcification films would NOT be top-1; it was wrong, with captions three of the four were (the bare labels
got none). One image of calcification-by-caption is a real signal worth a larger set, not a result: 4 films, two of them
from one 80+-year-old pattern.

The normals went badly again, the same way as before: the normal head-CT slice through the orbits and sinuses was read as
"sinus abnormality" at 0.98, a normal sagittal MRI as cerebral atrophy, and the lateral chest film (a view the label set was
never written for) as pneumothorax, with the truth ranked 10th.

**Pooled over the two fresh sets (26 images):** captions 15/26 top-1 against 14/26 for bare labels (6 gained, 5 lost, p = 1.0);
sensitivity 12/14 against 9/14; specificity 7/12 against 10/12. So the pattern in EXTENSION.md holds: the captions flag more
abnormal films and also call more normal films abnormal, and a net top-1 advantage is still not shown on images that took
no part in choosing them. Still missing: more than one atrophy film, any CT atrophy, and a second normal head-CT subject
who is not a slice through the orbits.

---

# Second search for the remaining gaps (2026-11-14): nothing usable found

A further search for single-panel, unannotated, openly licensed images for the gaps listed above (more cerebral-atrophy MRI,
any atrophy CT, a second normal head-CT subject, more normal films) added **no images**, so there is no extension 3. What was
found and why it was rejected: Alzheimer / Pick / LATE / DLB atrophy MRIs are multi-panel composites or carry boxes and arrows;
the two other atrophy CTs and MRIs found earlier are composites or annotated; the second "normal" brain CT on Commons is a
montage of the same Uppsala subject already in the set; the Antwerp "healthy lungs" film (Public domain) has the patient's name
and date of birth printed on it, and an AP normal film carries a date overlay and a partial ID, so neither was used; an MS
DIR-sequence MRI is circled and arrowed. The set stays at 88 images and the gaps stay as stated in EXTENSION2.md.
