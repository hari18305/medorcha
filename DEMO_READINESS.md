# MedOrchestrate — Demo Readiness

Pre-demo pass against the build guide's Demo Day Checklist, run for real
in this repo on 2026-08-28. Each item below was actually executed, not
assumed — see "Evidence" for how.

## Checklist

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Text-only case works end-to-end | ✅ (pipeline mechanics) | `test_integration.py` (mocked boundary) + real eval run, CASE002 |
| 2 | Image-only case works as far as the prototype supports | ✅ | `test_integration.py` + real eval run, CASE003 (degrades to `review_required=True` when imaging/RAG signal is thin — correct behavior, not a bug) |
| 3 | Text + image case works | ✅ | Real eval run, CASE004, CASE016 |
| 4 | Clinical Agent returns validated JSON | ✅ | 7 tests, `test_clinical_agent.py` |
| 5 | Imaging Agent returns validated JSON | ✅ | 10 tests; BiomedCLIP loaded and ran for real during the eval run (see Known Gaps for one real failure mode observed) |
| 6 | Qdrant contains biomedical vectors + metadata | ✅ (local store) / ⚠️ Docker needs re-ingest | `medorchestrate_kb` is now **30 points, 768-dim, 100% real citable content** (20 MedlinePlus + 10 PubMed excerpts, 0 synthetic) — verified directly against the local embedded store on 2026-11-08 and via `backend/tests/test_ingest.py`. This replaced the earlier 18–23 all-`synthetic` passages (the build spec requires ≥80% real). The dockerized `qdrant` volume still holds the OLD corpus until you run `docker compose exec backend python -m backend.rag.ingest` (ingest now rebuilds the collection cleanly). The case-memory collection is unchanged. |
| 7 | RAG returns cited source passages | ✅ | Real eval run: RAG relevance true for 14/17 cases (keyword-overlap heuristic) |
| 8 | Evidence Agent: live when available, degrades gracefully when not | ✅ | All **four** spec sources now implemented — PubMed, MedlinePlus, Europe PMC, ClinicalTrials.gov — run concurrently, each with a 10s per-request timeout + 1 retry and a 30s per-source deadline; verified against the real APIs on 2026-11-09 (16 merged items, 3.4s). A failed source is isolated and recorded (`LiveEvidence.failed_sources`); the report says "live evidence unavailable" only if all four fail (tests: `test_evidence_agent.py`, `test_integration.py`). Near-duplicate titles across PubMed/Europe PMC are dropped. **Caveat:** every source gets the same free-text symptom query, so result relevance is loose — ClinicalTrials.gov in particular returned off-topic trials in the live check. Each live item also counts toward the S_web confidence signal by its evidence level (a registered trial defaults to "moderate"), so confidence scores will shift versus the earlier two-source evaluation runs; re-run Mode 1 before quoting those numbers. |
| 9 | Fusion detects conflicts | ✅ | `test_fusion_agent.py`; conflict-detection path exercised in `test_integration.py` |
| 10 | Confidence computed, threshold (0.65) respected | ✅ | Real eval run: every case with confidence < 0.65 after 3 optimizer iterations shows `review_required=True` (CASE003, 006, 008, 013, 014); every case ≥0.65 shows `False` |
| 11 | Optimizer runs at most 3 iterations, never more | ✅ | Real eval run: max iterations across all 17 cases = 3, never exceeded (also asserted directly in `test_optimizer_agent.py`) |
| 12 | HITL triggered for unresolved low confidence | ✅ | Real eval run: HITL rate 29.4% (5/17 cases), all genuinely still below threshold after 3 iterations |
| 13 | Final report renders correctly in React | ✅ | Verified in-browser against the dockerized frontend |
| 14 | All FastAPI endpoints work (health, diagnose, diagnose/{id}, review/{id}) | ✅ | `test_api.py` (13 tests) + manual `/health` and `/diagnose` checks against the running Docker stack |
| 15 | Only synthetic data used anywhere in the demo path | ✅ | `data/images/README.md` and `data/test_cases/README.md` both explicitly label everything synthetic/de-identified; spot-checked |
| 16 | No secrets committed | ✅ | `git log --all -p` grepped for key patterns — only hit is a fake placeholder (`sk-ant-super-secret-value`) in `test_logging.py`, used to test the log-redaction function itself |
| 17 | Basic retries/timeouts on Qdrant/PubMed/Claude | ✅ | Confirmed in code (`evidence_agent.py`: `MAX_ATTEMPTS=2`, `HTTP_TIMEOUT_SECONDS`); exercised for real during the eval run |
| 18 | `docker compose up` starts the full demo cleanly | ✅ | Built and started for real: all three services `healthy`/`Up`, verified after a full teardown+restart too (Qdrant data persisted via the named volume) |
| 19 | At least 10-20 test cases in `data/test_cases/` | ✅ | **24** present (17 original + 7 CT / chest X-ray / unsupported-modality cases added 2026-11; guarded by `test_fixture_cases.py`). The eval figures quoted elsewhere in this file are from the original 17. |
| 20 | 3-5 minute demo script rehearsed end-to-end, once, out loud | ✅ Steps 1-5 driven live 2026-08-31 against the real running app (not just read) — header framing, real strong-case submission (confidence 0.656, matches the original proven run), full report rendering with ranked differentials/evidence, and the low-confidence optimizer+HITL path (confidence 0.40, 3 iterations, correct review-required routing) all confirmed working. Two real script bugs found and fixed in the process (see step 4 below). **Still worth it**: read the narration aloud once yourself so the timing/wording feels natural — the mechanics are proven, the delivery is still yours to practice. |

## The one real gap — CLOSED 2026-08-29

~~No real Claude API call has ever completed in this project or this
repo.~~ **Resolved.** A full, real, non-mocked `orchestrator.run_case()`
run completed successfully via `LLM_PROVIDER=gemini` (a free-tier
Gemini key — see `backend/services/llm_client.py`'s provider switch;
this closes the gap functionally, though it's Gemini's reasoning, not
Claude's, unless/until a Claude-serving path — Anthropic direct,
Bedrock, or Vertex — is what actually gets used for the real demo; see
the README's "Note on LLM provider choice"):

- Case: `CASE002_text_only` ("recurrent headache and dizziness over
  the past month, history of hypertension", 52F).
- Real clinical extraction, real BioBERT/Qdrant RAG retrieval, real
  PubMed live evidence, real fusion reasoning, all through genuine
  CrewAI orchestration (not a monkeypatched `_execute()`).
- **Confidence 0.656** — above the 0.65 threshold, optimizer correctly
  skipped, `review_required=False`.
- Three ranked, clinically sensible differentials: vestibular migraine
  (0.65) → hypertension-related headache (0.45) → vertebrobasilar TIA
  (0.35), every `supporting_evidence` string tracing back to a real
  `[RAGn]`/`[LIVEn]` citation with real cited text, exactly per
  `fusion_agent.py`'s no-fabricated-citations contract.
- 34.2s end to end.

Two real bugs surfaced and got fixed along the way (not found by
inspection — found by actually running it): the Milestone-doc-named
`gemini-2.0-flash` is retired (Google's own API 404 confirmed it,
pointing to a successor); and `gemini-3.6-flash`'s free tier is capped
at 5 requests/minute, not enough for this pipeline's ~8-10 real LLM
calls per case (CrewAI's orchestration-decision layer alone makes
~5-6, one per agent step) — `gemini-3.5-flash-lite` has a materially
more generous free quota and is what actually produced the run above.

**Still worth doing before the actual demo:** run at least one real
case through a **Claude-serving** path (Anthropic direct, Bedrock, or
Vertex) specifically, if the plan is to present this as Claude-based
work — the mechanics are now proven end-to-end on this architecture,
but Claude's own reasoning quality on this exact prompt set hasn't
been observed yet, only Gemini's.

### Update 2026-08-30: a real 4-case subset, and an honest rate-limit finding

Ran a second, larger real batch (`CASE002`, `CASE003_image_only`,
`CASE007_high_confidence_candidate`, `CASE008_low_confidence_candidate`)
after adding a rate limiter (`throttle_llm_call_sync()` in
`llm_client.py`) to `.env`'s free-tier `GEMINI_RPM_BUDGET=12`. Result:
**2 of 4 succeeded genuinely, 2 failed on 429s despite the limiter**:

- `CASE003_image_only`: **SUCCESS**, 183.4s, confidence 0.574 — full
  3-iteration optimizer path exercised for real (strategies: narrow →
  paraphrase → expand), correctly landed on `review_required=True`
  since confidence stayed below 0.65 after all 3 iterations.
- `CASE008_low_confidence_candidate`: **SUCCESS**, 125.2s, confidence
  0.533 — same full optimizer path, same correct HITL routing.
- `CASE002_text_only`, `CASE007_high_confidence_candidate`: **FAILED**
  on 429 RESOURCE_EXHAUSTED.

Root cause of the failures, precisely: **CrewAI has its own internal
retry-on-error logic** (`agent/core.py`'s `_handle_execution_error` →
`execute_task` → `_handle_execution_error` → ...) that makes fresh real
API calls when a step errors — each retry is a real network call this
project's rate limiter never sees, because the limiter only gates entry
into `crew.kickoff()` once per step, not CrewAI's own internal retries
inside a single `kickoff()`. The evaluation-run logs show 3+ separate
real 429/success attempts for a single logical "fusion" step a few
seconds apart, confirming this. Not fixable without either a paid tier
(trivially solves it — 15 req/min is a free-tier-only cap) or patching
CrewAI's own internals (out of scope for this timeline).

**Practical read for the actual demo**: three separate real, genuine
successes now exist across two sessions (this one plus the original
CASE002 run), including two that exercised the full 3-iteration
optimizer path correctly. A single interactive demo case — submitted
by a human, with natural pauses for narration — is far less likely to
trigger this than a scripted back-to-back batch was. If you want to
eliminate the risk entirely rather than just reduce it, any paid tier
(Gemini pay-as-you-go, or a working Anthropic/Bedrock/Vertex key) removes
this ceiling outright.

**Docker status as of tonight (2026-08-30)**: the running container had
`google-genai` live-patched in (`pip install` + restart, not baked into
the image) and was verified healthy and working before this session
ended. A `docker compose build` to bake it into the image properly
(now that `.dockerignore` and `requirements.txt` are both fixed, see
git log) was interrupted mid-`pip install` — genuinely stuck for 30+
minutes with zero CPU/network activity, a different failure mode than
the earlier `.dockerignore` build-context hang, cause not further
diagnosed tonight. Docker Desktop and WSL2 were shut down cleanly
rather than left hung. **Next session: `docker compose up -d --build`
should work cleanly now** (both real bugs from tonight are fixed) —
if the build stalls again at the `pip install` step specifically with
no progress for several minutes, that's a fresh issue, not a repeat of
either bug already fixed here.

## Update 2026-11-09: case memory (Phase 13), weight recalibration (Phase 14), and an open confidence regression

**Phase 13 — exercised for real, three bugs fixed.** A full round trip with the
real models (real FastAPI `POST /review`, BioBERT, BiomedCLIP, cross-encoder, a
temporary local Qdrant store; only the clinical-extraction LLM call stubbed)
showed: an override on an image case wrote one `text` and one `visual`
`case_memory_log` row; a new case with a similar complaint and a similar image
then received **both** a `clinician_verified:` and a `visual_precedent:` item
in its `RAGEvidence`; an unrelated chest-pain case received neither; a
text-only override wrote only a `text` row. Bugs found and fixed:

1. **Text case memory discarded the clinician's diagnosis** — retrieval
   returned only the past case's symptoms, so the model was told "a similar
   case existed" but never what the clinician concluded (a test even asserted
   it). Precedent text now carries "Clinician-verified diagnosis: …".
2. **No relevance cutoff** — every query got its 3 nearest past cases however
   unrelated. Now: text ≥ 0.6 BioBERT cosine (measured: similar ≈ 0.75–0.81,
   unrelated ≈ 0.2–0.3); visual ≥ 0.85, an *unvalidated* default (synthetic
   images can't calibrate it).
3. **Visual precedents ignored modality** — a brain-MRI override could come
   back as "visually similar" to a chest X-ray. Now filtered to the same
   modality (before the top-k cut).

Also: the Mode 17 ablation harness seeded a made-up "clinician-confirmed"
diagnosis into the real case-memory store and never removed it; it now cleans
up after itself. **Existing leftovers in the real store** (4 points with
case_id `CASE001_seed_precedent`, reviewer `mode17-eval`) are still there and
should be deleted — they would now be shown to the model as real precedents.
Caveat from the round trip: BiomedCLIP's zero-shot modality guess on the
synthetic MRI placeholder was "CT scan", so those placeholder images aren't a
reliable test of modality handling.

**Phase 14 — not runnable on real data yet.** The real database has **3
qualifying outcomes of the 10 required**, so `python -m backend.services.weight_tuner`
correctly does nothing (it now says so and how many more are needed). Fixed:
the gate counted *reviews*, so one case reviewed ten times could satisfy it; it
now counts distinct cases (latest accept/override each). The earlier Mode 2
"held-out calibration 13.3% → 86.7%" was **misleading**: 86.7% is exactly the
share of held-out answers that were correct, i.e. the trivial rule "always
trust" (fusion is a placeholder there, so agreement is constant and
recalibration just raised every confidence). The harness now prints that
baseline and a caveat; do not treat those weights as production weights.
"Weight recalibration run at least once on real outcomes" is therefore **not
met** and can only be met by collecting ≥10 genuine clinician reviews.

**Confidence regression from the real knowledge base (Phase 2) — fixed in the
score scale; Mode 1 still needs re-running.** With the new real
MedlinePlus/PubMed corpus the cross-encoder's raw logits for symptom-list
queries sit around −4 to −10, which a plain sigmoid flattened to ≈ 0.00: mean
S_rag across the test cases was **0.039**, capping confidence at 0.6 — below the
0.65 threshold — for nearly every case. Measured alternatives: BioBERT
similarity as S_rag separates on-/off-topic queries poorly (0.29–0.58 vs
0.21–0.33) and still can't reach the threshold, so it was rejected; the raw
cross-encoder ordering *is* informative (on-topic mean logit −6.5 vs off-topic
−9.7). The reranker now reports `sigmoid((logit + 7) / 2)` — a monotonic
rescale (ranking unchanged) — giving mean S_rag **0.54** (strong specific cases
0.64–0.72; thin or poorly covered ones 0.16–0.39 so they still route to
review; off-topic queries 0.15–0.29). **Caveats:** the two constants were set by
hand from 19 queries on the current 30-passage corpus — a prototype
calibration, not a validated one; re-check with
`python -m backend.rag.calibrate_reranker` whenever the corpus or model
changes. The earlier Mode 1 numbers (average confidence 0.74, optimizer 53%,
HITL 24%) came from the old synthetic corpus and do not describe the current
system; **re-run Mode 1** (real Gemini calls) before quoting any figures.

## Update 2026-11-09: Phase 16 — modality coverage

- **The placeholder demo images were being misclassified.** With the real
  BiomedCLIP the old flat-shape placeholders were labelled essentially at
  random: the brain-MRI image came back as "CT scan", two chest X-rays as
  "unspecified"/CT, one CT as "unspecified". Any earlier real run on those
  images therefore did not exercise the modality its filename claimed. They
  are replaced by structured synthetic images (same filenames) that the real
  classifier maps to the intended modality, plus an unsupported-modality image;
  `python scripts/generate_demo_images.py --check` re-verifies this.
  **Limit:** that shows the *fixtures* take the intended code path; BiomedCLIP's
  accuracy on real scans is still unmeasured.
- **Test cases:** 24 (was 17) — at least 3 per supported modality plus the
  unsupported-modality case. The spec suggests 15-20; the surplus is the missing
  coverage. Mode 1 now runs 24 cases, so its figures are not comparable with
  the earlier 17-case runs.
- **Integration tests added** for every previously missing matrix row: each
  modality (MRI / CT / chest X-ray) with and without text, with the image
  attached to fusion's LLM call; vision forced off (text-only fallback, logged
  with the reason); an unsupported-modality image end to end (no findings
  classification, placeholder kept out of the RAG query and the fusion prompt,
  no image attached); override -> text + visual case memory -> a later similar
  case retrieves both (and a different-modality / unrelated image gets none);
  weight recalibration through the real pipeline (no-op below 10 reviewed
  cases, then a refit that changes the next run's confidence); and a rating
  never changing the accept/override label.

## Update 2026-11-10: first real end-to-end run of the current system

One real case (28M, "chest pain and shortness of breath", plus a real chest
X-ray) submitted through the real React UI to the real backend with real Gemini
calls, BiomedCLIP, Qdrant and all four live evidence sources.

**Run 1 failed (HTTP 500) and exposed four bugs** — none visible to the mocked
tests:
1. `.env` selected `gemini-3.6-flash`, whose free tier is **20 requests per
   day**; one case used it up (plus several 503 "high demand" errors).
2. When CrewAI's own orchestration LLM call failed (429), the exception
   escaped `run_case()` and became a 500 instead of a degraded, flagged report.
   Fixed: `_run_step` logs it and continues with safe defaults; a
   below-threshold confidence can no longer come back as `review_required=False`.
3. The rate limiter was a blocking `time.sleep()` inside `async def` calls,
   freezing the whole event loop (no timeout could fire; the other evidence
   sources stalled). Fixed: it now runs in a worker thread.
4. PubMed's per-abstract LLM summaries queued behind that limiter and blew past
   the 30s source deadline, discarding PubMed even though its HTTP calls
   succeeded. Fixed: summaries are now opt-in (`EVIDENCE_LLM_SUMMARIES`, off by
   default), and when on they run concurrently under a 12s budget.
5. ClinicalTrials.gov answers `400 "Too complicated query"` above 13 words, and
   the optimizer's refined queries run to 15–20. Fixed: capped at 12 terms, and
   client errors are no longer retried.

**Run 2** (same case, `GEMINI_MODEL_ID=gemini-3.5-flash-lite` for that process
only) **completed: HTTP 200 in 72s, 12 Gemini calls, every step `success`.**
All four evidence sources answered (11 live items); `fuse()` logged
"multimodal path (attaching raw image (chest X-ray) alongside text for provider
'gemini')" on every fusion pass — i.e. **the raw image is attached to the LLM
call in a live run**. First-pass confidence 0.623 (< 0.65) triggered the
optimizer: 0.617, 0.458, 0.660 → final 0.66, `review_required=False`. The
reranker's calibrated relevance spread over the real knowledge base was sensible
(98%, 88%, 77%, 47%, 31%).

**What this does NOT show:** the top diagnosis (pneumothorax, 82%) simply agrees
with BiomedCLIP's label, and no image-vs-BiomedCLIP conflict was raised. The
teammate's mid-review deck shows a different reading of what appears to be the
same image (a large gas-filled structure in the left lower chest read as a
possible gastric bubble / diaphragmatic hernia rather than pneumothorax), so the
model's visual read varies by model and is not validated — one run proves the
wiring, not the quality, and this is decision support for clinician review only.
Confidence landed only just above the threshold, after three optimizer passes.

**Action for you:** unset `GEMINI_MODEL_ID` in `.env` (or set it to
`gemini-3.5-flash-lite`). As written, `gemini-3.6-flash` allows ~1–2 cases/day
on the free tier. Mode 1 over 24 cases needs a paid tier or several days of
quota.

## Update 2026-11-10: real scan images

Six real, openly licensed images (3 head CT slices, 1 brain MRI, 2 chest
X-rays: CC0 / public domain / CC BY 4.0, from Wikimedia Commons, each licence
read from its primary file page) are now in `data/images/real/`, with per-file
credits in `ATTRIBUTION.md`, seven `REAL00x` cases in `data/test_cases/real/`,
and `scripts/check_real_images.py`. **The real BiomedCLIP classifies all six as
the intended modality** (the old flat placeholders: 3 of 8), with plausible
readings (normal head CT, normal chest X-ray, pneumothorax on the pneumothorax
film). Limits: six hand-picked, well-known images are a plausibility check, not
an accuracy measurement; the pneumothorax figure comes from a PubMed Central
article BiomedCLIP may have seen in training; and there is no single-image
abnormal CT or MRI (all CT/MRI slices are normal-looking anatomy). For a demo,
prefer these over the synthetic images; the synthetic ones remain for the
automated tests.

## Update 2026-11-10 (later): a real image through the full live pipeline

Case `REAL006` — the real CC BY pneumothorax chest X-ray (panel A) plus the
vignette "sudden sharp chest pain and shortness of breath that started this
morning" (24M) — through the real UI, backend, Gemini (`gemini-3.5-flash-lite`),
BiomedCLIP, Qdrant and all four evidence sources: HTTP 200 in about 2.3 minutes,
12 Gemini calls, every agent step `success`, `fuse(): multimodal path` on every
fusion pass.

**Result.** BiomedCLIP: chest X-ray, findings `pneumothorax` and `rib fracture`
(conf 0.68). Fusion ranked **Pneumothorax 0.95** first — correct for this film.
Overall confidence 0.627 on the first pass; three optimizer passes reached
0.588 / 0.600 / 0.618, best 0.627 < 0.65, so `review_required=True` and the UI
routed straight to the review screen, as designed. The reranker's relevance
spread was again sensible (0.92 / 0.87 / 0.74 / 0.25 / 0.17).

**What went wrong (fixed below).** Fusion also ranked **Rib
Fracture at 0.85**. There is no rib fracture on this film and the vignette has
no trauma; the label came from BiomedCLIP's *secondary* finding (anything above
the 0.15 softmax threshold is passed on). The "supporting evidence" for it is a
generic knowledge-base passage and an unrelated PubMed item about aortic injury,
and the optimizer's refined queries then drifted to "traumatic pneumothorax
associated with acute rib fracture". The fusion prompt tells the model to treat
BiomedCLIP's labels as a secondary sanity check and to flag disagreement, but
here it adopted a spurious low-confidence label with high stated confidence and
raised no conflict. The deterministic confidence formula correctly ignored the
LLM's 0.85/0.95 self-reported numbers, which is why overall confidence stayed
below threshold and the case was flagged for review — but a clinician would see
a wrong differential entry at 85%. Candidate fixes (not done): pass only the top
finding (or findings above a much higher threshold) downstream; tell the model
explicitly that secondary labels are frequently spurious; and require a
differential entry to be supported by evidence that actually mentions it.

**Fix (2026-11-10) and live verification.** Measured on the real images, the
spurious `rib fracture` scored 0.15 (0.1514 over the 0.15 floor) against 0.68 for
the real finding (22% as strong); a second real pneumothorax film's secondary
scored 0.14 and was dropped, so the old rule was a coin flip at the floor.
Because the scores are a softmax (they sum to 1), a decisive top label leaves
little for the rest. `analyze_image` now reports a secondary label only if it
clears the 0.15 floor AND is at least 60% as strong as the top one
(`FINDING_RELATIVE_THRESHOLD`, default 0.6; a prototype value from a handful of
images, not validated): genuine two-way ambiguity (0.83–0.86 of the top) is kept,
noise (0.0–0.53) is dropped. The fusion prompts now state that imaging labels are
hypotheses, forbid ranking a diagnosis on an imaging label alone, forbid
inventing a mechanism such as trauma, and tell the image-attached model not to
rank a pre-filter label it cannot see. **Re-running the identical case live:**
findings `['pneumothorax']` only; differential Pneumothorax 0.95 (cited to a
collapsed-lung passage and a pneumothorax protocol paper), then Pleural Effusion
0.20; no rib fracture, no "traumatic" drift; first-pass confidence 0.684 so no
optimizer and no review flag; 11 Gemini calls, every step `success`.

Caveats: still one image, one run (LLM output varies between runs; the 0.684 is
only just over the 0.65 threshold). The tradeoff is real — softmax scores make
co-occurring findings compete, so a genuine second finding that is much weaker
than the first is now dropped from BiomedCLIP's labels (the attached-image model
still sees the image itself).

## Update 2026-11-12: LLM-judged evaluation, answer key, simulated recalibration demo

**Added (tooling only; no pipeline code changed):**
- `data/test_cases/answer_key.json` - expected-diagnosis key for all 30 cases
  (acceptable differential, or `abstain` for inputs too vague to diagnose).
  **Developer-drafted, NOT clinician-verified**; the file says so and a test
  fails if that label is removed. Treat any score computed from it as a
  consistency check until a clinician has reviewed it.
- `backend/tests/llm_judge.py` + `run_llm_judge.py` - RAGAS-style metrics
  (faithfulness, context precision, diagnosis correctness, unsupported claims)
  implemented natively, not the `ragas` library (it would pull langchain into
  the CrewAI environment). It judges *saved* reports, so judging can be re-run
  with a different judge without re-running the pipeline.
  `scripts/run_cases_live.py` produces the saved reports. 26 unit tests with a
  fake judge; four scoring mutations were each caught.
- `scripts/demo_recalibration_simulated.py` - shows the recalibration
  *mechanism* using 3 real reviews + 7 reviews written by the developer
  (labelled `SIMULATED-developer`) in a throwaway DB. The real DB is verified
  byte-identical afterwards. **The resulting weights only reflect a rule chosen
  in the script; they are not learned from clinicians and must not be quoted as such.**

**Smoke test (2 real cases, Gemini judging Gemini - i.e. self-grading):**
diagnosis in answer key 2/2; faithfulness 0.50; context precision 0.50; 0
unsupported claims. n=2, not a result. Two things it exposed:
1. **Knowledge-base coverage gap.** REAL004 (memory loss + unsteady gait) got
   0/5 relevant passages: the 30-passage KB has nothing on dementia,
   Alzheimer's or hydrocephalus. The metric found a real hole.
2. **Judge inconsistency.** For that same case the judge rated the WMH passage
   "not relevant" under context precision yet "supports" a diagnosis under
   faithfulness. LLM judges are noisy; do not over-read small differences.

**Live CT/MRI runs (REAL001 head CT, REAL004 brain MRI):** both completed with
no failed step and the multimodal path. REAL004 took 505 s (3 optimizer
iterations) - too slow to run live in a demo. Europe PMC timed out on both;
the other three live sources answered.

**Case-memory contamination (found by the REAL001 run):** its top diagnosis
(TIA) came from a stored override by `Dr. Demo` (2026-09-08) in the *real*
case-memory store - a demo artefact, not model reasoning. Same store holds two
empty-text `Dr. Lee` points. Remove before any recorded demo or evaluation run.

## Update 2026-11-12 (later): knowledge base 30 -> 42 passages

Added 12 verbatim MedlinePlus / PubMed passages for answer-key conditions that
had nothing behind them (Alzheimer disease, dementia, vascular cognitive
impairment, normal pressure hydrocephalus, hydrocephalus, idiopathic
intracranial hypertension, subarachnoid / hemorrhagic stroke, traumatic brain
injury, lung cancer, tuberculosis, chest injuries / rib fracture). The builder
now defaults to add-only (`--rebuild` re-fetches everything); the 30 existing
passages are byte-identical. This exceeds the spec's "20-30" and says so in
`data/knowledge_base/README.md`.

**Regression check** (old vs new KB, 12 answer-key queries, in-memory Qdrant,
real BioBERT + reranker): a relevant passage in the top 5 for **9/12 -> 11/12**
queries, and no query got worse. The memory-loss + gait case went from no
relevant passage to Dementia ranked 1st. Head injury on warfarin went from none
to Subarachnoid hemorrhage ranked 1st. Small sample, hand-chosen "relevant"
sets (mine, not a clinician's): evidence the gap closed, not a retrieval benchmark.

**CORRECTION / FIXED (same day):** the paragraph that stood here called the
missing chronic-subdural retrieval "a dense-retrieval weakness, not missing
content". That diagnosis was wrong. Root cause was in the KB builder: it
appended "Causes" before "Symptoms" and cut at ~300 words, so in 10 of 15
encyclopedia articles the Symptoms section was cut off or dropped. The subdural
passage described acute head-injury mechanics and never mentioned headache,
confusion or balance/walking problems, so a patient presenting with exactly
those could not match it (and widening the stage-1 candidate pool to all 42
passages changed nothing, which is what pointed at the content).

Fix: Symptoms now comes first (still verbatim), passages may run to 340 words
(`ingest.CHUNK_SIZE_TOKENS` 300 -> 350 so one passage stays one chunk), whole KB
rebuilt, new `--refresh DOC_ID...` builder option, and tests that fail if
Symptoms stops coming first. Measured on 14 answer-key queries (committed KB vs
fixed KB, real BioBERT + reranker, in-memory Qdrant): relevant passage in the
top 5 for **12/14 -> 14/14**, top-1 unchanged at 8. CASE009 with the query the
pipeline really builds: subdural not retrieved -> **rank 2**.

Costs, not hidden: two queries ranked worse - the memory-loss/gait case 1st ->
2nd (still found), and the chest-pain-after-sport case 1st -> 5th (the Chest
Injuries passage is still not in the stage-1 top 10 for it). The cross-encoder
is volatile: a passage's score can swing a lot when ~40 words are added. 14
hand-picked queries with relevant-sets chosen by the developer; not a benchmark.

## Update 2026-11-13: 31-case live run, a confidence bug, and the judge run that did not score

All 31 test cases (24 synthetic `CASE*`, 7 real-image `REAL*`) ran through the
live pipeline (Gemini `gemini-3.5-flash-lite`) in an **isolated copy** of the
store (`eval_run/`, gitignored: its own SQLite DB and a Qdrant copy with the 3
stale case-memory points removed, KB = 42). The real DB and real case memory
were not touched. The backend died once mid-run (cause unknown, no traceback);
it was restarted and the remaining cases re-run, so the run is two sessions.

**Pipeline facts (from the saved reports, no LLM judging involved):** 31/31
returned a report. Mean confidence 0.69; `review_required` on 3/31; optimizer
iterations 0 x17, 1 x11, 2 x1, 3 x2; live evidence present on 28/31.
**CASE007 is not a valid result**: Gemini returned 429 (quota) during fusion, so
it has no diagnoses and confidence 0.00 - it needs a re-run.

**Bug found and fixed: an empty case scored 0.79 and skipped review.**
`CASE013` (no input at all) came back with zero diagnoses, confidence 0.79 and
`review_required=False`. Cause: `compute_confidence` scored only evidence
quality; the optimizer's refined query retrieved strong passages, fusion still
produced no diagnosis, and 0.4*S_rag + 0.3*S_web + 0.3*A cleared 0.65. A result
with no diagnosis now gets confidence 0.0 (`confidence.py`), with a regression
test. The saved CASE013 report predates the fix.

**Abstention was mostly wrong - now gated for structurally empty input.** The
answer key marks 6 cases as "abstain" (`CASE003`, `CASE008`, `CASE013`,
`CASE017`, `CASE021`, `REAL002`); the correct behaviour is low confidence / human
review. Before the fix only `CASE021` did that (1/6). The others returned
0.67-0.69 with a confident diagnosis (e.g. "Ischemic Stroke" for an image-only
case). The breakdowns show why: agreement is 1.00 for nearly every case (a free
0.30), S_web sits at 0.4-0.8 whatever the input, and retrieval always returns
something, so C lands near 0.65 regardless of how little the case contained.

Fix: an input-sufficiency gate (`confidence.has_clinical_anchor`). With no
symptom, lab value or report finding, confidence is capped at 0.50, review is
required, the optimizer is skipped (retrieval cannot create missing input, and
it saved ~5 minutes per such case), and the report's `conflicts[0]` says why.
Diagnoses are still shown. Covered by unit tests (confidence, optimizer,
orchestrator; the gate was mutation-checked). By construction it applies to
`CASE003`, `CASE013`, `CASE017`, `CASE021`, `REAL002` - and to `REAL007`
(image-only pneumothorax, which the key treats as a correct diagnosis: it keeps
its diagnosis but now goes to review, which is the intended trade-off).

**Not verified live and not solved:** (1) the gate was not re-run on the real
pipeline because Gemini quota was still exhausted (4 attempted cases all failed
safe: confidence 0, review required); the change is unit-tested only. (2)
`CASE006` (report only) should count as anchored because the extractor returns
findings from the report, but that depends on the LLM and is unconfirmed. (3)
`CASE008` (vague symptoms: "feeling generally unwell") still gets 0.68 and no
review - the gate checks that input exists, not that it is specific. (4) The
0.65 threshold, the 0.4/0.3/0.3 weights, and the constant 1.0 agreement signal
remain uncalibrated; re-run the 31 cases when quota returns.

**LLM-judge run: did not produce usable scores.** `run_llm_judge` was run over
all 31 reports with Gemini as judge, but 70 of its 72 failed metric calls were
Gemini `429 RESOURCE_EXHAUSTED` (quota; the paid-billing balance had not yet
reflected). The numbers it printed (for example "diagnosis correct 0.4") are
computed from the few calls that succeeded and are **not** results; the output
was discarded, not committed. Re-run when quota is available:
`python -m backend.tests.run_llm_judge --reports eval_run/reports` (it is slow -
several LLM calls per case - so run it detached). Even then it is self-grading (Gemini
judging Gemini output), the answer key is developer-drafted and not
## Update 2026-11-13 (later): retrieval regressions, and a bug that hid case history and labs

**Correction.** The "chest-pain-after-sport 1st -> 5th" regression listed above was
measured on a query I wrote by hand ("...after a collision during sport, no
breathlessness"). The query the pipeline really built for that case (REAL005) was
`left-sided chest pain; imaging findings: normal chest X-ray, no acute findings`:
the clinical agent had dropped the collision/sport detail. On that real query the
relevant passage is rank 1, so it was never a retrieval regression. The real
problem is that extraction lost the mechanism of injury (not fixed: it needs a
clinical-prompt change that can only be verified with a working LLM).

**New harness, on real queries.** `python -m scripts.eval_retrieval` runs the real
BioBERT -> cross-encoder pipeline on an in-memory index of the committed corpus (no
API key, no network, real store untouched) over 14 queries recovered from the live
31-case run's log, relevant passages chosen by the developer (not clinician-verified).

**Bug found: the case's `history` and `labs` never reached the pipeline.** The API
accepts and stores them, but the clinical step only read `clinical_text` and
`medical_report`, so they were absent from the findings, the retrieval query, the
fusion prompt and the confidence gate. 17 of the 31 test cases carry them (e.g.
CASE018's atrial fibrillation + warfarin, CASE011's anemia + hemoglobin, CASE012's
five-item history). Fixed by `merge_structured_fields` (structured labs win;
history appended and de-duplicated); it also keeps them if extraction fails, and a
case whose only input is a lab value is no longer wrongly gated as "no clinical
input". **Consequence: the 31-case live run above, and any judged score from it,
predates this fix for those 17 cases and must be re-run.** (The web form has no
history/labs fields, so this affects API and scripted use.)

**Measured (14 queries, hit@5 / hit@1 / MRR@5):**

| pipeline | hit@5 | hit@1 | MRR@5 |
|---|---|---|---|
| before (history/labs missing from queries) | 13/14 | 7 | 0.685 |
| + history/labs merged (CASE011 miss -> rank 1) | 14/14 | 8 | 0.768 |
| + RRF ordering (`RERANK_ORDER=rrf`, now default) | 14/14 | 10 | 0.827 |

RRF fuses the BioBERT rank with the cross-encoder rank to order the top 5 (k=60,
untuned). It never changes WHICH 5 passages are chosen or their scores, so S_rag and
confidence are unchanged; I first tried full RRF over the whole pool and rejected it
because it changed the top-5 membership on 7 of 14 queries and lowered mean S_rag
0.659 -> 0.630. `RERANK_ORDER=ce` restores the old order.

**Costs, not hidden.** RRF helped 3 queries (CASE001, CASE010, REAL004) and hurt 1:
the CASE009 subdural passage (the original complaint) moved from 2nd to 4th - still
in the top 5 and still shown to the fusion model, but lower. n=14 and developer-judged,
so this is a regression check, not evidence the ordering is better in general. The
remaining weakness is the general-domain cross-encoder: for a child's headache +
URTI (CASE010) pulmonary tuberculosis is still ranked first and the relevant
headache passage third. A biomedical cross-encoder (e.g. NCBI's MedCPT) is the
obvious next step; it needs a ~440 MB model download, so it has not been tried.

## Known gaps (carried over from the team's HANDOFF.md, still true)

- `/diagnose` is fully synchronous — no async polling yet.
- The optimizer's later-iteration RAG/live evidence isn't exposed back
  into `FinalReport.evidence` (only first-pass retrieval is shown,
  even if a later iteration is what actually cleared the threshold).
- ~~`CASE001.json` and `CASE001_with_image.json` share a `case_id`~~ — fixed
  2026-11-12 (the image variant is now `CASE001_with_image`).
- A fresh clone/environment needs `python -m backend.rag.ingest` run
  once (or, in Docker, `docker compose exec backend python -m
  backend.rag.ingest`) before RAG retrieval finds anything.

## Demo script (3-5 minutes)

1. **Open the app** (`http://localhost:5173` via `docker compose up`,
   or `npm run dev` locally). Point out the header framing: "clinical
   decision-support prototype — for clinician review only" — this is
   deliberate, not filler text.
2. **Submit a strong case**: age/sex + clinical text ("headache and
   dizziness, history of hypertension") + the synthetic demo MRI image.
   Narrate while it's in flight: "this runs through a clinical-text
   agent, a BiomedCLIP imaging agent, a BioBERT/Qdrant retrieval agent
   against an offline biomedical knowledge base, and a live
   PubMed/MedlinePlus evidence agent, each stage feeding the next."
   (Corrected 2026-09-01: `orchestrator.py`'s `run_case()` calls each
   agent via `_run_sync()` in a flat sequence — clinical, then imaging,
   then RAG, then live evidence, then fusion — not concurrently. Don't
   say "in parallel" if asked to point at where that happens in code.)
3. **Show the report**: ranked differential diagnoses, confidence
   score and bar, supporting evidence list (point out the `type: rag`
   vs `type: live` tag distinguishing offline vs. live-fetched
   citations), and any flagged conflicts.
4. **Trigger the optimizer**: submit a deliberately thin case — age 40,
   sex other, clinical text **"occasional mild headaches, nothing else
   notable"** (verified live 2026-08-31: confidence 0.40, correctly
   triggers all 3 optimizer iterations, correctly lands below
   threshold). Narrate the confidence-driven refinement loop —
   "confidence came back below 0.65, so the system is now refining its
   own query and re-running retrieval, capped at 3 iterations."
   **Do not use `CASE013_minimal_no_content.json` as literally
   written** — its fields are fully empty and the frontend's own client
   -side validation blocks submission ("Provide at least clinical
   text, a medical report, or an image."), confirmed live. Also avoid
   fully generic filler like "not feeling well" — verified live that it
   pulls almost entirely irrelevant PubMed hits (dementia, romantic
   love, hypnosis) into the live-evidence panel, since there's no
   clinical anchor for the retrieval query; a vague-but-topical phrase
   keeps the evidence panel coherent while still landing low-confidence.
5. **Show HITL**: once a case lands below threshold after 3 iterations,
   walk into the Review screen — Accept / Override / Annotate — and
   explain this is mandatory, not optional, for low-confidence cases.
6. **Close on architecture**: `docker compose ps` showing the three
   services, and the one-sentence framing again: evidence-grounded
   decision *support*, confidence-scored and citation-traceable,
   never an autonomous diagnostic authority.
