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

### Follow-up: MedCPT biomedical cross-encoder (now the default reranker)

Downloaded `ncbi/MedCPT-Cross-Encoder` from Hugging Face (439 MB, into the HF cache)
and compared it to the MS MARCO model on the same 14 real queries (hit@5 / hit@1 /
MRR@5):

| reranker | order | hit@5 | hit@1 | MRR@5 |
|---|---|---|---|---|
| MS MARCO | cross-encoder | 14/14 | 8 | 0.768 |
| MS MARCO | RRF (the previous default) | 14/14 | 10 | 0.827 |
| **MedCPT (new default)** | cross-encoder | 14/14 | 11 | 0.863 |
| MedCPT | RRF (`RERANK_ORDER=rrf`) | 14/14 | 12 | 0.895 |

MedCPT fixes the off-topic-first failure (CASE010: the headache passage is now
1st; pulmonary tuberculosis no longer tops a child's headache). RRF is now OFF by
default: with MedCPT it adds only +0.032 MRR and pushes the CASE009 subdural passage
from 3rd to 5th, the edge of the top 5.

**Costs, not hidden.** (1) ~440 MB more to download on first run (Docker has no
model cache, so every fresh container downloads it; the Docker build has not been
re-run). (2) CASE009's subdural passage is 3rd, not 2nd as with MS MARCO; CASE020
(cough, smoker) also dropped 1st -> 4th with MedCPT alone. (3) n=14, relevance
judged by the developer: a regression check, not a benchmark.

**Calibration changed with the model, and S_rag feeds confidence.** MedCPT's raw
logits span -16..+13 (off-topic top-5 around -15.7, on-topic around -8), nothing like
MS MARCO's range, so the old center/scale (-7 / 2) would have been wrong. MedCPT now
uses (-12.5, 3.5), fitted by hand so the MEAN S_rag over the 14 real queries (0.659)
and 6 off-topic queries (0.284) matches the old pipeline (0.659 / 0.288). Only the
average is matched: per query S_rag moves by up to ~0.3 (CASE009 0.26 -> 0.55, which
is a better reflection of the subdural passage now being retrieved well; REAL006
0.82 -> 0.69), i.e. confidence by up to ~0.12 (mean 0). **Cases near the 0.65
threshold can flip either way, so the 31-case run must be repeated with this model
before any threshold or accuracy number is quoted.** The 0.65 threshold itself is
still unvalidated. sentence-transformers applies a sigmoid to MedCPT by default, which
would have turned every logit into ~1e-5; the model is now loaded with an identity
activation (tested). If your `.env` was copied from the OLD `.env.example`, delete
its `RERANK_SCORE_CENTER` / `RERANK_SCORE_SCALE` lines or they will override the
MedCPT calibration with MS MARCO's (the real `.env` here has none).
`RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` restores the old behaviour
(with its own calibration).

## Update 2026-11-13 (later): real-image imaging benchmark

**What was wrong.** The Mode 3 imaging benchmark scored BiomedCLIP on programmatically
generated gradient-and-ellipse pictures whose "intended label" was arbitrary, so
"near chance" was the expected result and it measured nothing. It is deleted (code, tests
and the committed images) and replaced.

**What it is now.** 62 real images (32 chest X-ray, 18 head CT, 12 brain MRI) in 60
groups, each with a finding label: 56 downloaded from Wikimedia Commons (CC0 / CC BY /
CC BY-SA / public domain; each file's licence read from its own page; downscaled to
<= 768 px and credited in `data/benchmarks/imaging_real/ATTRIBUTION.md`) plus the 6
already in `data/images/real/`. `python -m scripts.build_imaging_benchmark` rebuilds it
from an explicit list of titles; `python -m backend.tests.imaging_benchmark` scores the
**production** classifier (`imaging_agent.rank_image`, not a reimplementation) and writes
`RESULTS.md` / `results.json`. I looked at every image on a contact sheet and replaced or
dropped about two dozen: arrows, circles and boxes drawn on the film (these would hand over the
answer), composite multi-panel figures, viewer screenshots, and films showing what look like
patient-identification strings. One annotated film remains (the existing pneumothorax
figure) and is flagged, not hidden.

**Results (Wilson 95% intervals; chance in brackets).**

| | images | modality detected | top-1 finding | top-3 | normal-vs-abnormal sens. / spec. |
|---|---|---|---|---|---|
| all | 62 | 94% (85-97) | **42%** (30-54) [chance 10%, always-commonest 21%] | 65% (52-75) [28%] | 91% / 78% |
| chest X-ray | 32 | 100% | 53% (36-69) | 72% | 96% / 75% (3/4) |
| head CT | 18 | 78% (55-91) | 33% (16-56) | 56% | 73% / 100% (3/3, one subject) |
| brain MRI | 12 | 100% | 25% (9-53) | 58% | 100% / 50% (1/2) |

Top-1/top-3 give the model the TRUE modality (isolating the findings model); end-to-end
(modality must also be right) is 40% overall. The zero-shot model is clearly better than
chance and not good: roughly one in two chest films and one in three CTs get the right top
label. The pipeline passes the truth downstream in 45% of images and passes at least one
WRONG finding in 63%.

**The failure that matters.** Hemorrhage on head CT is called "acute infarct" in 4 of 5
images, with scores 0.74-0.99 (three of them 0.95 or higher), and "normal" in the fifth (0/5 top-1).
Hemorrhage versus infarct is the most consequential split in head CT, and the zero-shot
labels do not make it. Related: 4 of 18 CTs are detected as MRI (including 2 infarcts, 1
hemorrhage), consolidation is called pulmonary edema 3/3, brain tumours are called infarct
in all 4 MRIs (3 acute, 1 chronic), and 5 abnormal images (a CT hemorrhage, a mass, a rib fracture, hydrocephalus,
sinusitis) are called normal. This supports what the agent's docstring already says: the
imaging read is a coarse pre-filter, not a diagnostic result, and it should not be shown to
anyone as one. The fusion step's raw-image path is the real read and is NOT measured here.

**Limits, all recorded at the top of `RESULTS.md`:** small (1-5 images per finding; CT
midline shift and MRI atrophy had no usable image, MRI white-matter hyperintensity has
one); labels are the uploader's, not a radiologist's, and the selector is not a radiologist;
Commons holds clear teaching examples, so these numbers are optimistic; possible overlap
with BiomedCLIP's PubMed Central training data (a keyword heuristic flagged only 3 images,
so it probably misses some; excluding them changes top-1 by under a point); one normal
head-CT subject, so CT specificity is close to meaningless. It characterises one model on
a hand-picked set; it is not a clinical validation and says nothing about the whole
system's diagnostic accuracy.

### Follow-up: finding-prompt wording, tested on a held-out half

The labels were bare phrases ("hemorrhage"). I tested whether captions that read like the
figure captions BiomedCLIP was trained on do better. Because 62 images is too few to try
variants and keep the winner, the design was fixed and committed BEFORE any result
(`9049e41`): a group-level dev/test split (37 / 25 images), five finding-prompt and four
modality-prompt candidates, "choose by dev MRR", and "adopt only if TEST top-1 gains >= 2
images and MRR does not drop". The test half was used once (the script refuses a second run).

| | dev (37) | **test, held out (25)** |
|---|---|---|
| current wording (bare label) | 19/37 top-1 | **7/25** top-1, MRR 0.49 |
| chosen: "head CT scan showing intracranial hemorrhage, hyperdense blood" | 23/37 | **12/25** top-1, MRR 0.61 |

Paired on test: 7 images gained, 2 lost, exact McNemar p = 0.18 (**not significant**: with 25
images only a very large effect would be). The rule said adopt, so the new wording is the
default (`IMAGING_FINDING_PROMPTS=label` restores the old one). Modality-detection wording was
also tried and **not** adopted (24/25 -> 25/25 on test, a gain of 1, bar 2). The dev-to-test
drop for the chosen wording (62% -> 48%) is what selecting on dev looks like.

Whole benchmark, before -> after (the all-image figures are now partly in-sample, so quote the
test-half row):

| | before | after |
|---|---|---|
| top-1 (all 62) | 42% | 56% (44-68) |
| top-3 | 65% | 79% |
| test half only, top-1 | 28% (7/25) | **48% (12/25)** |
| reported findings contain truth | 45% | 60% |
| pipeline also passes a wrong finding | 63% | 48% |
| sensitivity (abnormal called abnormal) | 91% | 96% |
| **specificity (normal called normal)** | **78% (7/9)** | **56% (5/9)** |

**What improved:** head-CT hemorrhage 0/5 -> 3/5 top-1 and none is called "acute infarct"
any more (the failure flagged above); brain-MRI tumours 0/4 -> 4/4; CT sinus 0/2 -> 2/2;
emphysema 0/2 -> 2/2; lung nodule/mass 0/3 -> 2/3.

**What got worse, and the rule did not see it.** On the whole set, specificity (normal called
normal) fell from 7/9 to 5/9. Taken apart:

- It is entirely a DEV-half effect (6/6 -> 3/6). On the held-out TEST half specificity went
  1/3 -> 2/3 and sensitivity was unchanged (21/22 both): no harm showed up there, on 3 normals.
- It is one failure mode: the "sinus abnormality" caption attracts normal brain images. Two of
  the three normal head-CT slices (one subject) and one normal brain MRI are now called sinus
  abnormality (scores 0.72-0.89). Those two CT slices are the ones the real-image demo cases use.
- Part of the "normal got worse" signal is a scoring inconsistency of mine, not the model:
  3 of 4 normal chest films now get "no significant abnormality", which is semantically normal
  and counts as normal in sensitivity/specificity, but my top-1 scoring accepted that synonym for
  MRI normals only (not chest or CT), so it counted them wrong. Counted as normal, chest is 4/4
  (old wording 3/4). I left the benchmark's labels alone so the held-out result stays exactly what
  was pre-registered; changing the accepted labels after seeing it would favour the new wording.
- Atelectasis did fall 2/3 -> 0/3 (top-3 3 -> 1), and consolidation is still 0/3 top-1 (3/3 top-3).

**Decision (2026-11-13): keep the new captions.** The held-out half is the only unbiased evidence
and favours them on every downstream-relevant measure (top-1 7 -> 12 of 25, truth reported 8 -> 12,
a wrong finding passed on 19 -> 14, sensitivity equal, specificity not worse). Reverting would
overrule the pre-registered rule because of a concern that did not appear on the held-out half.
Two honest caveats. (1) Not statistically significant (p = 0.18), and the normals are only 9
images from about 6 subjects, so a real specificity cost is not excluded; head CT in practice is
mostly normal, which makes false findings matter more than in this abnormal-heavy set. (2) The
real normal-CT demo images (`real_ct_scan_06/10`, used by REAL001-REAL003) now report "sinus
abnormality" at the imaging step, which also reaches their retrieval query. If that is unacceptable
for the demo, set `IMAGING_FINDING_PROMPTS=label` in `.env` (all three read "normal head CT" at
~1.00 under the old wording) and restart; do not edit the descriptions to chase it.

**Limits.** I had seen the full-set failure patterns before writing the descriptions, so they
are informed by them (never tuned against a run); the descriptions are a non-radiologist's; the
labels are the uploader's; ~25 test images; the descriptions must not be edited to chase this
benchmark now that the test half has been used.

## Update 2026-11-13 (later): async /diagnose with real progress

A case can take minutes (REAL004 took ~500 s with three optimizer rounds), and the old processing screen was a
timer-driven animation that could not tell you what was actually happening. Now:

- `POST /diagnose?async=true` returns 202 at once and runs the case in a worker thread
  (`backend/services/jobs.py`); `GET /diagnose/{id}/status` returns live progress. The plain `POST /diagnose` is
  unchanged, so `scripts/run_cases_live.py` and every existing test keep working.
- The progress is real: the orchestrator reports each step from `_run_step` (the one place every step passes
  through) and the optimizer reports each refinement round. Each step carries its actual result ("5 offline
  passage(s)", "3 diagnosis(es), confidence 52%"), its measured duration, and, if it failed, a generic "failed, the
  analysis continued with less" with no raw provider error. Imaging shows as skipped when no image was attached; the
  optimizer shows as skipped with its reason when confidence already cleared the threshold or there was no clinical
  input to refine from.
- The processing screen shows exactly that, plus a queue message when workers are busy, and polls every 1.2 s. It
  tolerates up to 5 consecutive failed polls (a blip does not discard a run that is still going), and starting a new
  case stops the old polling loop (checked: 0 requests in the 5 s after abandoning).

**Verified:** 563 backend tests (registry, the async routes with the pipeline faked, the orchestrator's progress
events, the optimizer callback; mutation-checked) and 23 frontend tests (`npm test`: the polling logic with injected
fakes). **Run for real** against the real backend (isolated store): 202 in 0 s, then the steps moved in real time
with genuine timings (clinical 4.2 s, retrieval 18.5 s, ...) and the UI followed them to the review screen. Because
Gemini was over quota, every LLM-dependent step ended `failed`, so that run exercised failure reporting, not a
healthy end-to-end analysis. The healthy path (green steps with details, optimizer rounds, queue message, hand-over to
the report) was checked in the browser against a scripted mock that replays a realistic run, not against the live
pipeline. **A healthy live run is still to do once Gemini quota returns.**

**Limits:** state is in this process's memory (one server process; a restart or `--reload` drops running jobs and
the status endpoint reports them as interrupted); concurrency is capped (`DIAGNOSE_WORKERS`, default 2) and extra
cases queue; there is no cancel button (leaving the page stops following the run, the server finishes it).

**Noticed, not changed:** when fusion fails, the report's "Conflicting evidence" still shows the provider's raw error
text (e.g. "429 RESOURCE_EXHAUSTED ... check your plan and billing details"). That is deliberate and tested today
(`test_orchestrator.py` asserts it) so quota failures are diagnosable, but it is provider detail on a clinician-facing
screen; worth replacing with a generic message before a demo if the quota problem could recur.

## Known gaps (carried over from the team's HANDOFF.md, still true)

- ~~`/diagnose` is fully synchronous~~ — done 2026-11-13: `POST /diagnose?async=true` + `GET /diagnose/{id}/status`
  give real step-by-step progress (see the update above); the plain `POST /diagnose` is still synchronous.
- ~~The optimizer's later-iteration RAG/live evidence isn't exposed back into `FinalReport.evidence`~~ - done
  2026-11-13: `optimize()` now returns `final_rag` / `final_live` and the orchestrator puts them in the report (see below).
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

---

## Vague-input signal (2026-11-13)

`CASE008` ("feeling generally unwell for a few days, mild fatigue, occasional
lightheadedness, nothing specific") scored 0.68 and skipped review in the 31-case
run, because the gate above only checks that *some* symptom exists.

Fix: `confidence.input_quality()` now returns `ok`, `no_anchor` or `nonspecific`.
When every extracted symptom is a general one (whole-phrase match against a small
hand-written list, after dropping qualifiers such as "mild", "occasional" and
duration words) and there are no labs or report findings, confidence is capped at
`NONSPECIFIC_INPUT_CAP = 0.60`, review is required, the optimizer is skipped, and
`conflicts[0]` carries `NONSPECIFIC_INPUT_NOTE`. The breakdown gains
`input_quality`; `input_sufficient` keeps its old meaning. The frontend shows the
note beside the review notice, as for the first gate.

**Verified:** 581 backend tests (18 new: lexicon cases including "arm weakness",
"dizziness" and "fatigue + headache" staying specific, labs and report findings
overriding the cap, the cap never raising a lower score, the optimizer skip and
the orchestrator note) and the frontend build. A rough scan of the 24 text fixtures
finds `CASE008` the only one it would cap. One integration test used "fatigue" as
its low-confidence symptom to exercise the optimizer loop; it now uses
"unexplained weight loss", since "fatigue" alone is now correctly non-specific.

**Not verified:** not run live (Gemini quota). Whether the extractor phrases
`CASE008`'s symptoms in a way the lexicon matches is unconfirmed: the list matches
phrases such as "feeling unwell" or "mild fatigue", not every paraphrase, so a
vague case worded differently can slip through. Cases a clinician would call
non-specific but that name one specific symptom (for example "dizziness") are not
caught on purpose. The 0.60 cap is chosen, not fitted. Demo note: the "avoid
generic filler" advice above still holds for the evidence panel, but such a case
now goes straight to review instead of three optimizer rounds.

---

## Imaging benchmark widened to 79 images (2026-11-13)

17 images were added (4 normal chest films; CT midline shift x2, infarct, 2 normal slices; MRI vascular malformation
x3, chronic infarct, edema, WMH, 2 normals), chosen to fill gaps and fixed in `data/benchmarks/imaging_real/EXTENSION.md`
*before* they were scored. They are marked `set: extension`, are not in `split.json`, and the prompt experiment is frozen
on the original 62 (`load_manifest(only="original")`).

**Result on the 17 fresh images:** current captions 10/17 top-1 (Wilson 36-78%) against 11/17 for bare labels; paired, 3
gained and 4 lost (p = 1.0). So the captions' advantage on the original held-out half was **not** reproduced. The
trade-off held: sensitivity 59/62 vs 54/62, specificity 11/17 vs 15/17 over all 79. Nothing was re-tuned; the captions stay
with the weaker claim ("more sensitive, less specific, net gain unconfirmed") and `IMAGING_FINDING_PROMPTS=label` as the
way back. Still uncovered: cerebral atrophy and CT calcification (only composites or annotated images were available).
There is still one normal head-CT subject; two extension slices share it with the original set.

`results.json` / `RESULTS.md` now cover all 79; the 62-image figures quoted in the project report are kept in
`results_original62.json` / `RESULTS_original62.md`; `results_bare_labels.json` is the bare-label comparison run.


---

## Provider error text no longer reaches the report (2026-11-13)

When the fusion step failed, the raw exception text was pasted into `FinalReport.conflicts` ("Fusion reasoning failed:
429 RESOURCE_EXHAUSTED ... quota ..."), so it reached the API response and the UI. Provider errors can carry quota and
billing figures, project or key identifiers and request URLs.

Fix: `backend/services/errors.py` maps an exception to a fixed one-clause reason (request limit reached, credentials
rejected, provider unreachable or unavailable, answer the system could not use, or a generic "unexpected error") and the
report now says, e.g., "The reasoning step could not be completed (the language-model provider's request limit was reached),
so no diagnosis is shown. A clinician must review this case; retrying later may help." No part of the exception's own text is
included. The full text still goes to the structured log, where an operator needs it. Both places that wrote the old
message (the fusion tool and the CrewAI-layer fallback) use it.

**Verified:** 601 backend tests (new `test_errors.py`: each category, status codes matched as whole numbers, secrets never
present in the message; two orchestrator tests, one per failure path, assert the report is clean and the log is not). The
old test that asserted the raw `RESOURCE_EXHAUSTED` text was changed to assert its absence, and a mutation check (restoring the
raw message in the fusion tool) fails the new test.

**Limits:** the category is a keyword match on the exception, so an error that matches nothing is reported as "an unexpected
error occurred" (never its own text); an unusual provider wording can land in the wrong category. Other steps (clinical,
imaging, RAG, live evidence) only send fixed text to the report and progress stream; their exception text goes to the log
only. Not exercised against the real provider: quota is still exhausted, which is itself the case this handles.

---

## Frontend component tests (2026-11-13)

The frontend had only plain-JS helper tests (23, Node's runner); no React component was exercised. Added Vitest + React
Testing Library + jsdom (dev dependencies), `npm run test:ui` (and `npm test` now runs both suites): **92 tests in 9 files**
beside the 23 helper tests.

Covered: the case form (age/sex required, nothing-to-analyse and history-only messages, half-filled lab row rejected, exactly
what is put in the FormData, an unconfirmed history draft still sent, the image preview, the sparse-input hint, the example
chips, restore after a failed submit), history chips and lab rows, the report (rank order, citation chips that link or stay
inert, review notice, both input-note kinds kept out of "Conflicting evidence", evidence filters and precedents, live-evidence
unavailable, imaging assessed / not assessed / no findings), the review screen (each validation, the exact payload for accept,
override, annotate and rating, the confirmation and failure paths), the progress screen (step statuses, durations, skipped
reasons, optimizer rounds, the elapsed clock), the error banner (kind-specific text, retry only for network and server) and
the confidence gauge (tiers, clamping, the report's own review flag), plus the App flow with the API mocked (form to report,
straight to review when required, queued polling, failure and retry with the form kept, route guards, new case).

Mutation-checked: letting Enter submit the form (8 failures), dropping the unconfirmed history draft (1; found that an
earlier test passed for the wrong reason because clicking the button blurs the box, so a second test now submits without a
blur) and no longer recognising the non-specific note (2).

**Not covered:** layout and styling (jsdom has none; the visual check was done in the browser earlier), the real backend
(the API is mocked at `services/api.js`), drag-and-drop of an image, and the `Report` page's own wrapper beyond what `App`
exercises. The polling test waits one real interval (1.2 s), so the UI suite takes about 25 s.

---

## Three gaps closed (2026-11-13)

### 1. The report now carries the evidence of the optimizer round that won

A later optimizer round re-retrieves and re-fuses, and its fusion cites `[RAG1]`, `[LIVE2]` numbered over *that round's*
lists. The report used to carry only the first pass's evidence, so those citations pointed at the wrong item or at nothing.
`optimize()` now returns `final_rag` / `final_live` (the evidence the returned fusion was written from; `None` when the
first pass is still the best) and `run_case` uses them. The confidence breakdown already followed the best round.
Tests: 4 in `test_optimizer_agent.py` (the winning round's evidence is returned, the best of several rounds supplies it and not
the last, none when the first pass wins, none from the short-circuits) and 2 in `test_orchestrator.py` (the report holds the
round's evidence and not the first pass's; the first pass's stays when the optimizer returns none); a mutation check (not
applying `final_rag`) fails the first. **Not verified live** (Gemini quota): the optimizer path has only been run with faked
providers since this change.

### 2. Knowledge base: 42 to 67 passages (details: `data/knowledge_base/README.md`)

25 passages for conditions the answer key and the live reports reached for, and for findings the imaging models can name
(pulmonary edema, atelectasis, mediastinal widening, mass lesion, vascular malformation, midline shift, calcification),
fetched and checked by the existing build tool. Two PubMed candidates were swapped before the commit because their abstracts
were under the 100-word band the ingest test enforces.

**Measured, including the cost.** On the 14 real agent queries hit@5 stays 14/14 but MRR@5 falls from 0.863 to 0.764
(hit@1 11 to 9): new, plausibly related passages now sit above the developer's marked-relevant one (those sets were not
changed). At the old top-10 candidate pool the anticoagulated-fall case lost its subdural-hematoma passage entirely, so the
pool is now 20 (`RETRIEVE_TOP_K`), and the reranker's score centre was refit from -12.5 to -10.4 so the mean S_rag of those 14
queries is 0.659 again (off-topic queries 0.209; it was 0.288). Confidence values of earlier runs therefore are not directly
comparable with new ones beyond that average. For the new conditions, 21 queries I wrote for this (so a coverage check, not an
independent test) reach a relevant passage in the top 5 for 18, against 3 on the original 42; the three misses are
mediastinal widening, vertebrobasilar insufficiency and musculoskeletal chest pain. **To use it you must re-ingest**
(`python -m backend.rag.ingest`); the Docker image needs a rebuild too.

### 3. Imaging benchmark: 88 images (EXTENSION2.md)

9 more images (CT calcification x4, one atrophy MRI, a second normal head CT, two normal brain MRIs, one lateral normal chest
film), plan committed before scoring. Captions 5/9 top-1 against 3/9 for bare labels (3 gained, 1 lost, p = 0.63); calcification
3/4 with captions against 0/4 (my stated expectation was the opposite). The normals again went the wrong way (1/4 against
2/4). Over both fresh sets (26 images): 15/26 against 14/26 top-1 (6 gained, 5 lost, p = 1.0), sensitivity 12/14 against 9/14,
specificity 7/12 against 10/12. Still thin: one atrophy image and no CT atrophy (no clean images; the CT label set has no
atrophy label), and the second normal head CT is a single slice through the orbits (read as sinus abnormality at 0.98).


---

## Live check of the gate and the general-symptom cap (2026-11-13, 4 cases)

Gemini answered again (a 9-token test call, then 4 real cases, about 140 s in total, no 429 anywhere, no tracebacks), on an
isolated store (`eval_run/qdrant_live4`: KB = 67 passages, case memory emptied; `eval_run/live4.db`).

| Case | Before (31-case run) | Now, live |
|---|---|---|
| CASE013, empty input | 0.79, no diagnosis, **skipped review** | 0.00, no diagnosis, review required, no optimizer, input note shown |
| CASE008, "generally unwell, mild fatigue, occasional lightheadedness" | 0.68, no review | **0.60**, review required, optimizer skipped, non-specific-input note shown; two diagnoses still listed |
| CASE003, image only (MRI) | 0.67 to 0.79 band, skipped review | **0.50**, review required, input note shown; fusion even flagged the picture as a graphic and not a scan |
| CASE002, ordinary text case | 0.656 | 0.679, no review, 0 optimizer rounds, 5 offline + 11 live evidence items |

So the input-sufficiency gate and the general-symptom cap work on the real pipeline for these four, and the extractor's wording for
CASE008 did match the lexicon. **Not shown by this run:** no case entered the optimizer, so the new "return the winning round's
evidence" path is still only tested with fakes; four cases are not the 31 and say nothing about calibration or the threshold; one small
wording flaw, fixed afterwards: the empty-case note ended "Any diagnosis shown rests on the image and retrieved literature alone"
although there was no diagnosis and no image; it now says "(there may be none) rests only on the image, if one was given, and on
retrieved literature".

---

## Full 31-case live re-run on the final system (2026-11-13/14)

All 31 cases through the real pipeline (Gemini `gemini-3.5-flash-lite`), isolated store (KB = 67, case memory emptied), in batches of
4, 8, 7, 5 and 9 (the last two are one set of cases that was interrupted: the **backend process died once**, exit code 139 with no
traceback, during CASE022, the second time this has happened in a long run; it was restarted and the nine unfinished cases re-run).
**No quota errors at all**; every case returned a report. Reports are in `eval_run/reports_final/` (gitignored); the previous
(pre-gate) run is `eval_run/reports/`.

| | previous run (pre-gate, 42-passage KB) | this run |
|---|---|---|
| mean confidence | 0.69 | 0.632 |
| sent for review | 3 / 31 | **13 / 31** |
| the 6 "should abstain" cases sent for review | 1 / 6 | **6 / 6** (CASE003, 008, 013, 017, 021, REAL002) |
| empty case (CASE013) | 0.79, no diagnosis, skipped review | 0.00, review |
| optimizer ran | 14 cases | 10 cases (rounds: 0 x21, 1 x2, 2 x1, 3 x7) |
| live evidence present | 28 / 31 | 30 / 31 (the empty case has none) |
| CASE007 | invalid (429) | valid, 0.82 |
| reports citing an evidence id that is not in the list | 1 of the 14 optimizer cases | **0 of the 10** |

What this shows: the input-sufficiency gate and the general-symptom cap do what they were built for on the whole set, and the
optimizer path now runs live. The citation row is a consistency check on the "return the winning round's evidence" change: no
unresolved citation in any report, including the 10 that entered the optimizer; it cannot show that a later round's evidence was
what got returned, only that nothing dangles.

What it does not show: correctness. 7 of the 31 went through three optimizer rounds and still ended below the threshold. The real-image
pneumothorax case with no text (REAL007) is now sent for review at 0.50 (the intended trade-off of the gate). The answer key is
AI-drafted, so no accuracy figure is quoted here; the LLM-judge scoring is the next step and is not done. 13/31 reviewed is a
behaviour of the current threshold and weights, which are still unvalidated.

---

## LLM-judge run over the 31 final reports (2026-11-14)

Judge `gemini-3.5-flash-lite` (the same family that wrote the reports, so a consistency check only), 0 judge errors, ~150 calls.
Diagnosis correct 0.903 (28/31; the 6 abstention cases scored by rule, all correct because flagged for review), faithfulness 0.453
(30 scored), context precision 0.616 (30), unsupported claims in 1 case (3 claims). Output: `eval_run/judge_final.{md,csv,json}`.

The 3 diagnoses marked wrong, inspected: **CASE004 is a real failure** (a generated picture; BiomedCLIP returned "edema, cerebral
atrophy, chronic infarct", the differential followed it - TIA / vertebrobasilar insufficiency / PRES - at 0.73 with no review,
although the key says the text decides); REAL005 is defensible (normal film, no lung disease named, but not the musculoskeletal
cause the key expects); CASE007 looks like a judge error (the second diagnosis matches the key). Low faithfulness, most of all on
image-only cases, was not checked against a clinician. An independent (non-Gemini) judge and a clinician spot-check are still open.

Possible follow-up from CASE004: an image that carries no clinical signal still steers the differential; consider a plausibility
check on very low-confidence image findings, or lowering the weight of the image when the text conflicts.

---

## Three optional items done (2026-11-14)

**1. Browser end-to-end tests (`npm run test:e2e`, 7 tests, about 1 min).** Playwright drives the installed Chrome (no browser
download) against the built frontend and a real FastAPI backend. `scripts/e2e_backend.py` keeps the real app, async job registry and
progress events, orchestrator, confidence formula with both input-quality rules, optimizer loop, review endpoints and SQLite, and fakes
the extractor, BiomedCLIP, retrieval, live evidence, the fusion model, the optimizer's strategy choice and the case-memory write. They
cover: validation in the browser with no request sent; an ordinary case through progress, report, a citation chip that scrolls to its
evidence, and the patient summary; a vague complaint going straight to Review with the non-specific note at 60%, closed with an
annotation and a rating; an image with no text held at 50%; weak evidence running the three optimizer rounds on screen and ending in
review; an override recorded and the report reopened; and a reload falling back to the form. Mutation-checked (removing the 0.60 cap
makes the vague-complaint test fail: 94% shown). **Not covered:** any real model (the backend is canned by design), layout and
styling beyond what the assertions touch, drag-and-drop, other browsers. It uses port 3000 because 5173 was taken by a dev server that
is not part of the test.

**2. Imaging benchmark gaps: nothing usable found.** A second search added no images (see EXTENSION2.md): the remaining
candidates are composites, annotated, or carry a patient's name and birth date. The set stays at 88; atrophy has one film, CT atrophy none.

**3. Retrieval relevance re-judged.** About 40% of the MRR drop was a stale key: re-judged MRR@5 is 0.881 (42 passages) against
0.821 (67 passages), versus 0.863 against 0.764 under the original sets (knowledge-base README has the table and the caveats; the
additions were committed before re-scoring, by a non-clinician).

---

## Six items worked through (2026-11-14)

### 1. CASE004 image steering: fixed (indecisive reads are not used)

**The failure.** CASE004 pairs "headache and dizziness, history of hypertension" with a generated picture. BiomedCLIP's labels for it
were edema 0.34 / cerebral atrophy 0.23 / chronic infarct 0.22 (top score 0.34: a flat softmax, i.e. "could not tell"). They were listed
as imaging findings, went into the retrieval query, and steered the differential to TIA / vertebrobasilar insufficiency / PRES at
confidence 0.73 with no review, although the key says the text decides.

**The fix.** `imaging_agent.INDECISIVE_SCORE_THRESHOLD = 0.40` (env `IMAGING_MIN_TOP_SCORE`): a read whose top label scores below it is
returned with `inconclusive=True`. `ImagingFindings.drives_retrieval` is false for it, so it is kept out of the RAG query, case memory,
the optimizer prompt and the fusion prompt (which is told the read was inconclusive and not to use its labels). The report keeps the
labels for transparency and the UI says "Inconclusive read (best label 34%) ... not used".

**Live result.** CASE004 now returns vestibular migraine / hypertension-related headache and dizziness / giant cell arteritis at 0.75, no
review (text-driven, as the key expects). CASE001_with_image, CASE016, REAL004 and REAL006 were re-run as controls and their images were
still read as decisive (0.65 to 0.82), so nothing changed in how they are used.

**What it does not fix, and what it costs.** (a) A *confident* wrong read is not caught: the generated brain MRI in CASE001_with_image
scores "cerebral atrophy" 0.65 and a generated chest X-ray "pulmonary nodule" 0.9, because a softmax over a closed label set always
looks decisive on nonsense. (b) The threshold is not validated. On the 88 real benchmark images, reads with a top score under 0.4 were
right 4 times in 9 and reads at 0.8 or above 24 times in 35, so the rule drops about 4 correct reads to avoid 5 wrong ones. It is a
safety measure for indecisive reads, not an accuracy gain.

### 2. Backend crash: reproduced without any LLM call, worked around, not root-caused

The backend process died with exit code 139 and no Python traceback in three long runs. With `faulthandler` on, the dump showed a
Windows access violation inside a torch operation (an embedding lookup the first time, a GELU the next) in a ThreadPoolExecutor worker,
in the middle of the MedCPT cross-encoder call from `rag_agent._run_pipeline`. The orchestrator runs each step with `asyncio.run()`,
which creates a new event loop and a new default executor, so retrieval ran on a fresh thread on almost every call.
`scripts/stress_model_threads.py` repeats the real retrieval + rerank + BiomedCLIP read in that pattern with no model API calls: it
**crashed after about 35 iterations (about 4 minutes)**, in a different torch op than the server crash. `backend/services/inference.py`
now routes every neural-network call (BioBERT embed, MedCPT rerank, BiomedCLIP) through one persistent thread; the same stress test then
**completed 300 iterations (33 minutes) without a crash**. That points at torch's per-thread native state, not at a particular model, but
the exact native cause was not found, so treat it as a measured workaround. Side effect: model calls are serialised across concurrent
cases (they were CPU-bound anyway). Tests: `test_inference.py`.

### 3. Clinical extraction dropping detail (REAL005): fixed in the prompt

For "left-sided chest pain after a collision during sport, no breathlessness" the extractor returned only "left-sided chest pain"
(and on a later call kept the mechanism but not the negative), so the trauma context never reached retrieval or fusion. The prompt now
asks for stated negatives as "absence of X" and for the recent event and its timing in `history`. Three live calls on the REAL005 text
now all return `symptoms: [left-sided chest pain, absence of breathlessness]`, `history: [collision during sport]`; two other texts
(a URTI headache, an anticoagulated fall) were also checked. Because "absence of ..." entries are now symptoms, the confidence gate
ignores stated negatives (`absence of`, `no `, `denies`, `without`), so a case made only of negatives is still "no clinical anchor";
3 new tests. Not shown: that retrieval or the differential improved for REAL005 (it was not re-run end to end).

### 4. Demo CT cases: two of the three real CT demo images are read as "sinus abnormality"

REAL001 and REAL003 use `real_ct_scan_10` and `real_ct_scan_06`: normal head CT slices at orbit and skull-base level (one subject).
With the current descriptive captions BiomedCLIP reads them as sinus abnormality (0.72, 0.89) and REAL001's differential lists sinusitis
second. `real_ct_scan_14` (REAL002) reads as normal (0.65). With the bare labels (`IMAGING_FINDING_PROMPTS=label`) all three read as
normal at 0.99 to 1.0. This is the captions' known specificity cost (see EXTENSION.md), now visible on the demo images. **Decision left
to the team:** for the demo either set `IMAGING_FINDING_PROMPTS=label`, or use the REAL002 image for the CT walk-through and explain the
others as a known limitation; the default was not changed because the captions' adoption was a pre-registered decision.

### 5. Fusion step's reading of the raw image: measured on 88 images (`scripts/measure_fusion_image_read.py`)

Asking the multimodal LLM (Gemini `gemini-3.5-flash-lite`) the same closed-set question BiomedCLIP is asked, with the true modality given:
**68 of 88 right (77%, Wilson 67 to 85%) against BiomedCLIP's 50 of 88 (57%) on the same images; paired, the LLM alone was right 26
times and BiomedCLIP alone 8 times (exact McNemar p = 0.003).** Abnormal films flagged 66/67, normal films called normal 20/21 (BiomedCLIP
with captions, all 88: 63/67 and 12/21). By modality LLM / BiomedCLIP: MRI 17/23 vs 14, CT 23/28 vs 14, chest 28/37 vs 22; on the
fresh extension sets 19/26 vs 15/26. **Caveats that matter:** the labels are the uploaders'; the images are public teaching examples a
large model may have seen in training, so this is probably optimistic; the list is closed, which flatters the LLM; one model, one
prompt, one pass. What it does show: the raw image attached to fusion carries more reliable information than the zero-shot labels, which
supports sending it, and which is a reason to be careful about the labels-driven query. Raw output: `eval_run/fusion_image_read.json`.

### 6. The 31 cases a second time, and the interface rehearsal

**Repeat run (final code, no crash).** All 31 cases re-run through the real pipeline with the CASE004 rule, the extractor prompt fix and the
single model thread, with `faulthandler` on: no crash, no quota error (the earlier long runs crashed after 8 to 23 cases). Against the first
final run: mean |confidence difference| 0.023 (median 0.01, max 0.17; 27 of 31 within 0.05, 30 within 0.10); **review decision identical for 27 of
31**, the 4 that flipped all sat at 0.58 to 0.69 (CASE001_with_image 0.65 to 0.65, CASE006 0.69 to 0.63, REAL001 0.65 to 0.58, REAL003 0.61 to
0.66), so reviewed cases went 13 to 15; top diagnosis same (loose name match) 22 of 31; mean top-3 overlap 83%; optimizer rounds the same in 26
of 31. The six gated cases were identical (0.50 / 0.60 / 0.00, review). CASE004 was the only inconclusive image read. Two passes give the
size of the variation, not an interval. Details: `eval_run/variation.json`.

**Rehearsal through the browser UI against the real backend and Gemini (the healthy async path).** (a) "Hypertension, headache" example:
live steps with real timings (extraction 2.6 s, retrieval 12 s, live evidence 1.8 s), report at 76% above the threshold with cited evidence.
(b) "Elderly, on warfarin" example: the optimizer's rounds appeared live ("Round 1 of 3: confidence 53% (paraphrase)", then round 2 refining),
ending on the Review screen at 62% with chronic subdural hematoma first (model-stated 90%), supporting passages cited, and the
"clinician review required" notice. Two cases, watched by the developer; not a usability study. The review submission itself was not
repeated here (it is covered by the browser end-to-end tests).

---

## Demo CT wording: decision taken (2026-11-14) - use the bare labels for the demonstration

**Evidence.** All seven real demo images, read both ways (no LLM involved): with the descriptive captions (`showing`) the two normal head-CT
slices of REAL001 and REAL003 are called "sinus abnormality" (0.72, 0.89) and the REAL001 differential lists sinusitis; with the bare labels
(`label`) all three real CTs read as normal (0.99 to 1.0) and the MRI, normal chest film and both pneumothorax films read the same or better
(chest X-ray normal 0.55 as "normal chest X-ray", pneumothorax 0.68). Beyond the demo images, the later fresh-image sets did not favour the captions
overall (15/26 vs 14/26 top-1) and again cost specificity (7/12 vs 10/12), while the raw image the fusion step also sees is read with sensitivity 66/67,
so the zero-shot labels are not the only safety net for missed abnormalities, whereas a false "abnormal" label does steer retrieval and the
differential. For a demonstration that must not show a normal scan as abnormal, the bare labels are the better choice.

**What was changed.** The demonstration profile sets `IMAGING_FINDING_PROMPTS=label`: appended to the local `.env`, and documented in `.env.example` and the
README. **The code default stays `showing`** on purpose: the adoption of the captions was a pre-registered decision, the whole evaluation and the report
describe the captions as the default, and flipping a scientific default because of seven demo images would not be defensible; the team can decide that
after clinician input. (Anyone running the demo must have `IMAGING_FINDING_PROMPTS=label` in their own `.env`.)

**Live check with the demo profile** (real backend and Gemini, isolated store, no crash): REAL001 imaging "normal head CT, no acute findings" (1.0), sinusitis
gone from the differential, 0.67 and no review (was 0.58, review); REAL002 normal CT, image-only, 0.50 with review as the gate requires; REAL003 normal CT
read 1.0, 0.79; REAL005 normal chest film read "normal chest X-ray" and, with the extractor fix, the top diagnosis is now rib contusion / chest-wall
musculoskeletal pain (the key's answer) at 0.62 with review; REAL006 pneumothorax first at 0.72. Differentials vary a little from run to run (see the repeat run).
