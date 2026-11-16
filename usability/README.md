# Clinician usability session kit

A small, formative usability study of the MedOrchestrate report screen with real clinicians, using synthetic cases only.
It asks two questions: **can a clinician understand what the result and the score mean without knowing how the system works,
and does the redesigned screen (`s8`) help more than the original (`master`)?**

This is a usability and comprehension study. It is **not** clinical validation and must not be described as one.

## What is in this folder

| File | Use |
|---|---|
| `01_consent_and_information_sheet.md` | Read or hand to each participant before starting |
| `02_session_script.md` | What the facilitator says and does, minute by minute |
| `03_task_cards.md` | The cases and tasks the participant works through (give one card at a time) |
| `04_comprehension_questions.md` | Questions after each round, with the answer key (facilitator only) |
| `05_sus_and_feedback_form.md` | The System Usability Scale (SUS) and a few short feedback questions |
| `06_observation_sheet.md` | The notetaker's sheet |
| `results_template.csv` | One row per participant per screen; copy to `data/responses.csv` |
| `score_usability.py` | Computes SUS and comprehension results from the CSV |

Participant data goes in `usability/data/` (ignored by git, so no participant answers are ever committed).

## Who and how many
- 3 to 5 clinicians (junior doctors, residents or a senior clinician). Record role and years of experience, not names.
- A small number is enough to find the main problems; it is not enough for statistical claims. Report it as formative, n = 5 at most.
- Ask your guide for an introduction to clinicians, and ask whether your department needs any approval for a short session with
  synthetic cases and anonymous participants.

## Two versions, side by side
Each clinician sees both versions. Both can run at once against the **same backend** (the new backend also works with the old screen):

- **Version A, original screen** (`master`): http://localhost:3000
- **Version B, redesigned screen** (`s8`): http://localhost:5173

The backend only accepts these two ports, which is why they are used.

One-time setup of the original screen in a second folder (from `D:\medorcha`):
```bash
git worktree add ../medorcha-old master
```
```bash
cd ../medorcha-old/frontend
```
```bash
npm install
```

Each session day, start three things:
```bash
# terminal 1, in D:\medorcha (on branch s8): the backend
venv\Scripts\activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
```bash
# terminal 2, in D:\medorcha\frontend: Version B (redesigned)
npm run dev
```
```bash
# terminal 3, in D:\medorcha-old\frontend: Version A (original)
npm run dev -- --port 3000
```
Then run **one warm-up case** before the first participant (the first case after a start is the slowest, about a minute).
Keep the laptop plugged in, awake and online (the language-model service and the literature sources need internet).

## Order of use (counterbalanced)
So that the screen is not confused with the case or with practice, rotate who sees what first. With two case sets
(Set 1 = Cases 1 and 5, Set 2 = Cases 3 and 4, see `03_task_cards.md`):

| Participant | Round 1 | Round 2 |
|---|---|---|
| P1 | A with Set 1 | B with Set 2 |
| P2 | B with Set 1 | A with Set 2 |
| P3 | A with Set 2 | B with Set 1 |
| P4 | B with Set 2 | A with Set 1 |
| P5 | as P1 | |

## Plan
1. Prepare: read all documents, print task cards and forms, set up both versions. (half a day)
2. Pilot with one friend (not a clinician) to catch broken tasks and timing. Fix, then reset any counts. (1 hour)
3. Run 3 to 5 sessions, about 35 to 40 minutes each. Two people: one facilitator, one notetaker.
4. Enter the data in `data/responses.csv`, run `python usability/score_usability.py`, and write up the top problems.

## What to report
- Participants: number, roles, experience range.
- SUS mean for each version (with the individual scores; with so few people do not quote a confidence interval as if it were precise).
- The share of correct answers to each comprehension question for each version.
- The top 3 to 5 problems seen, with a short quote or observation each, and what you changed.
- Limits: very small sample, synthetic cases, a prototype, same people judging both versions, not a clinical validation.

## Notes on data and ethics
- Use only the supplied synthetic or open-licence cases. Never enter real patient information.
- Participants are identified only as P1, P2 and so on. Do not record names, faces or voices unless a participant agrees in writing.
- Clinician "accept or override" decisions made during a session are clinician reviews of **synthetic** cases. They are not outcomes on
  real patients, so do not describe them as evidence for tuning clinical weights.
