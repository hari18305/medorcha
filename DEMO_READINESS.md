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
| 19 | At least 10-20 test cases in `data/test_cases/` | ✅ | 17 present |
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

## Known gaps (carried over from the team's HANDOFF.md, still true)

- `/diagnose` is fully synchronous — no async polling yet.
- The optimizer's later-iteration RAG/live evidence isn't exposed back
  into `FinalReport.evidence` (only first-pass retrieval is shown,
  even if a later iteration is what actually cleared the threshold).
- `CASE001.json` and `CASE001_with_image.json` share a `case_id` —
  harmless unless both are submitted through the real API.
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
