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
| 6 | Qdrant contains biomedical vectors + metadata | ✅ | Verified via the real Qdrant HTTP API: `medorchestrate_kb`, 18 points, 768-dim, both in the local embedded store and in the dockerized `qdrant` service |
| 7 | RAG returns cited source passages | ✅ | Real eval run: RAG relevance true for 14/17 cases (keyword-overlap heuristic) |
| 8 | Evidence Agent: live when available, degrades gracefully when not | ✅ | Real PubMed/MedlinePlus calls made during the eval run; retry+timeout confirmed in `evidence_agent.py` |
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
| 20 | 3-5 minute demo script rehearsed end-to-end, once, out loud | ⚠️ Script below is ready; **you still need to read it aloud once** before the real thing |

## The one real gap

**No real Claude API call has ever completed in this project or this
repo.** Every "works" claim above for Claude-dependent behavior
(clinical extraction quality, fusion reasoning quality, optimizer query
refinement quality, CrewAI's own orchestration-decision layer) is
verified through:

- 154 passing unit/integration tests with the Anthropic client mocked
  at the `_get_client()` boundary, and
- a real, non-mocked run of everything **except** the LLM calls
  themselves (`backend/tests/run_evaluation.py`) — real Qdrant
  retrieval, real PubMed/MedlinePlus network calls, real BiomedCLIP
  inference, real confidence-formula math, real optimizer control flow.

This means the *pipeline mechanics* are solid and demo-safe (routing,
degradation, thresholds, iteration caps, persistence, logging), but
nobody has seen what an actual Claude-generated differential diagnosis
looks like end-to-end in this repo yet.

**Before the real demo:** get access to Claude through *any* of the
three supported paths (see `.env.example` /
`backend/services/llm_client.py`) — `ANTHROPIC_API_KEY` from
console.anthropic.com, or `LLM_PROVIDER=bedrock`/`vertex` if
console billing is blocked and an AWS/GCP account (with its usual free
trial credit) is faster to stand up — put it in `.env` (or pass it via
`docker compose`'s env file), and run at least one real case through
`POST /diagnose` (or the UI) to see genuine output once. Budget API
cost for this: CrewAI's orchestration layer alone makes ~5-6 Claude
calls per case (one per agent step, to "decide" to invoke its tool),
plus the direct calls inside clinical/fusion/optimizer — so each demo
case is roughly 8-10 Claude calls, not 1.

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
   Narrate while it's in flight: "this fans out to a clinical-text
   agent, a BiomedCLIP imaging agent, a BioBERT/Qdrant retrieval agent
   against an offline biomedical knowledge base, and a live
   PubMed/MedlinePlus evidence agent — running in parallel."
3. **Show the report**: ranked differential diagnoses, confidence
   score and bar, supporting evidence list (point out the `type: rag`
   vs `type: live` tag distinguishing offline vs. live-fetched
   citations), and any flagged conflicts.
4. **Trigger the optimizer**: submit a deliberately thin case (e.g.
   `CASE013_minimal_no_content.json`'s fields) and narrate the
   confidence-driven refinement loop — "confidence came back below
   0.65, so the system is now refining its own query and re-running
   retrieval, capped at 3 iterations."
5. **Show HITL**: once a case lands below threshold after 3 iterations,
   walk into the Review screen — Accept / Override / Annotate — and
   explain this is mandatory, not optional, for low-confidence cases.
6. **Close on architecture**: `docker compose ps` showing the three
   services, and the one-sentence framing again: evidence-grounded
   decision *support*, confidence-scored and citation-traceable,
   never an autonomous diagnostic authority.
