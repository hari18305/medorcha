# Task cards

**Print only Part A.** Part B is for the facilitator. The case texts are made up; the images are open-licence teaching images kept in
`demo_assets/` on the project laptop (that folder is not in git). Pre-fill each case before handing the card over, so that
the task is about reading and understanding the result, not typing.

Case sets (see `README.md` for which participant gets which set first):
- **Set 1:** Case 1 (chest X-ray) then Case 5 (vague complaint)
- **Set 2:** Case 3 (brain MRI) then Case 4 (normal chest X-ray)

---

# Part A: participant cards

## Task Card 1 (first case of the round)
The case is already entered. Press **Analyze case**, wait for the report, then tell us out loud:
1. What is the system's leading diagnosis?
2. Does the system say a clinician must review this result? How can you tell?
3. What evidence supports the leading diagnosis? Show us where you would check it.

## Task Card 2 (second case of the round)
The case is already entered. Press **Analyze case**, wait for the report, then tell us out loud:
1. What is the system's leading diagnosis?
2. Does the system say a clinician must review this result? Why, or why not?
3. What, if anything, does the report say about the image (if there is one)?
4. What would you do next with this result?

## Extra question for the normal-film case only (Case 4)
5. Does the report mention anything that is **normal** or **absent** in this patient? Where do you see it?

---

# Part B: facilitator only

Wording to type into the form is exactly as below. The "seen in rehearsal" results come from earlier runs (Gemini,
demonstration settings). The language model varies from run to run, so wording and near-threshold decisions can differ. Judge each answer against what the
**report on screen actually says**, not against this list.

## Case 1: chest X-ray, pneumothorax
Image `1_chest_xray_pneumothorax.png`. Age 24, Male.
Complaint: "24-year-old male, otherwise healthy, tall and slim. Sudden sharp chest pain and shortness of breath that started this morning while resting. The pain gets worse when taking a deep breath. No fever, no cough, no recent injury or trauma, no known lung disease."
Past history: none. Labs: none.
Seen in rehearsal: read "pneumothorax", second reading agrees; primary spontaneous pneumothorax first; no review.

## Case 5: vague complaint (no image)
Age 45, Female. Complaint: "Generally unwell, mild fatigue, occasional lightheadedness." Past history: none. Labs: none.
Seen in rehearsal: score capped at 60 and review required, with the note about non-specific input.

## Case 3: brain MRI, contested image read
Image `3_brain_mri_t1.png`. Age 72, Female.
Complaint: "Gradual memory difficulty and unsteady walking over several months, with a new urgency to pass urine. Family says she shuffles and takes small steps. No sudden onset, no weakness on one side, no seizures."
Past history: hypertension. Labs: none.
Seen in rehearsal: the automated image read and the second reading disagree, so the image is shown as contested and left out of the differential; normal pressure hydrocephalus first. The score sits near the review line, so review may or may not be required. Either is fine.

## Case 4: normal chest X-ray, with negatives
Image `4_chest_xray_normal.jpg`. Age 21, Female.
Complaint: "Left-sided chest pain after a collision during sport, worse when pressing on the chest wall and with movement. No breathlessness, no cough, no fever."
Past history: none. Labs: none.
Seen in rehearsal: normal film read; chest wall contusion or musculoskeletal pain first, rib fracture second; usually no review (can fall either side).

## What counts as task success
| Task | Success if the participant... |
|---|---|
| Leading diagnosis | names the diagnosis ranked first on screen |
| Review needed? | says yes/no matching the report **and** points to where the screen says it |
| Evidence | finds a cited source and opens or reads it |
| Image statement | correctly paraphrases what the image section says (including "contested", "inconclusive" or "normal") |
| Normal or absent findings (Case 4) | locates the normal film and at least one stated negative |

Mark each as **done**, **done with help or hesitation**, or **not done** on the observation sheet. Do not give hints.
