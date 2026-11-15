# Architecture and design notes

Deep dives behind the main [README](../README.md): how the pieces fit, why the confidence score behaves as it does,
and how clinician feedback is remembered. Everything here describes the code as it is, including what is not validated.

## LLM provider choice

The Milestone II design doc (Section 8) lists a multi-provider setup
per agent — GPT-4o-mini, Gemini-2.0-Flash, DeepSeek-R1, Qwen2-72B —
under the stated principle "off-the-shelf LLMs selected per task, no
fine-tuning." This implementation runs every agent (clinical
extraction, RAG reranking, fusion reasoning, the optimizer, and
CrewAI's own orchestration layer) on a single model by default instead
(the code default is `LLM_PROVIDER=anthropic` — Claude). **Every live run and
evaluation recorded in this repository was made with `LLM_PROVIDER=gemini`
(`gemini-3.5-flash-lite`)**; the Claude path is implemented and unit-tested
but has not been exercised end to end here.

This is a deliberate, consistent substitution, not an oversight or a
partial implementation: the same underlying principle holds (an
off-the-shelf, pretrained, non-fine-tuned LLM chosen per task) with a
single vendor rather than four, which keeps auth/config/error-handling
uniform across every agent and avoids juggling four separate provider
SDKs, rate limits, and pricing models in a course-timeline prototype.
Every agent's system prompt and role remains task-specific per the
original design (e.g. the RAG reranker's relevance-scoring prompt
differs entirely from the fusion agent's differential-diagnosis
prompt) — only the model vendor is unified.

`LLM_PROVIDER` also accepts `gemini` and `mistral` as a functional
fallback if no Claude-serving path (Anthropic direct, Bedrock, or
Vertex — see [CONFIGURATION.md](CONFIGURATION.md#switching-llm-provider)) is reachable in time.
Gemini specifically is one of the four models this project's own
Milestone II doc originally named, so using it isn't off-design if it
comes to that — but be clear about which provider actually produced a
given run's output rather than presenting Gemini/Mistral output as
Claude's.

## Joint multimodal fusion

`fuse()` (`backend/agents/fusion_agent.py`) attaches the case's actual
image bytes to its LLM call — via `call_multimodal()` in
`backend/services/llm_client.py` — whenever all of these hold: an image
is present, its modality has a real BiomedCLIP findings model
(`unsupported_modality=False`), and the active `LLM_PROVIDER` has
confirmed vision support (`supports_vision()`: anthropic/bedrock/vertex/
gemini today; mistral defaults to `False`, unconfirmed). In every other
case it falls back to the original text-only path (BiomedCLIP's
candidate labels, or the "NOT ASSESSED" wording for an unsupported
modality) — always logged, so a report reviewer can tell "no image" from
"the provider can't use one."

## Case memory (text + visual)

When a clinician overrides a diagnosis (`POST /review/{case_id}` with
`decision: "override"`), the correction is embedded and stored as
retrievable evidence for future similar cases:

- **Text** (`backend/rag/case_memory.py`): always written, in the
  `medorchestrate_case_memory` Qdrant collection (BioBERT/768-dim,
  separate from the curated KB), tagged `clinician_verified:<reviewer>`.
- **Visual** (`backend/rag/visual_memory.py`, new): additionally written
  when the case had an image in a supported modality, in the
  `medorchestrate_visual_memory` collection — a **separate embedding
  space** (BiomedCLIP's image encoder, whatever dimension that actually
  is, determined at runtime — never assumed to match BioBERT's 768),
  tagged `visual_precedent:<reviewer>`.

Both get folded into `RAGEvidence` on later similar cases (`run_rag()`)
and reach the fusion prompt with an explicit "this is a past-case
precedent, weigh it accordingly, don't treat it as infallible" framing.
Each precedent's text states **the clinician's verified diagnosis**, not
just the past case ("Similar past case: … Clinician-verified diagnosis: …").

A precedent is only retrieved if it is genuinely similar:

- **Text:** BioBERT cosine ≥ `CASE_MEMORY_MIN_SCORE` (default 0.6 —
  measured: similar cases ≈ 0.75–0.81, unrelated ≈ 0.2–0.3).
- **Visual:** only past images of the **same imaging modality**, and
  BiomedCLIP cosine ≥ `VISUAL_MEMORY_MIN_SCORE` (default 0.85 —
  an *unvalidated* prototype default; the synthetic demo images can't
  calibrate it).

## Confidence and the input-sufficiency gate

`C = w1*S_rag + w2*S_web + w3*A` measures how much *evidence* was retrieved,
not whether the case gave the system anything to reason from, and retrieval
always returns something. So two rules sit on top of the formula:

- **No diagnosis -> confidence 0.0.**
- **No clinical anchor -> confidence capped at 0.50 (< 0.65), review required,
  optimizer skipped.** An anchor is at least one symptom, lab value or finding
  extracted from a report; history and demographics alone do not count, nor do
  findings the imaging model produced. The report carries a note saying why.
  This makes an image-only or empty case go to a clinician; any diagnosis is
  still shown.
- **Only general symptoms -> confidence capped at 0.60 (< 0.65), review
  required, optimizer skipped.** If every symptom is a general one ("feeling
  unwell", "mild fatigue", "lightheadedness") and there are no lab values or
  report findings, the differential is too broad to be sure of, and more
  retrieval cannot narrow it. The match is on the whole symptom phrase against
  a small hand-written list (`confidence.py`), so "arm weakness" or "fatigue
  with chest pain" are not caught. It is a transparent heuristic, not a
  validated classifier, and the 0.60 cap is a chosen value, not a fitted one.

The 0.65 threshold and the weights remain unvalidated.

## Confidence weight recalibration

`backend/services/weight_tuner.py` periodically refits the confidence
formula's `w1`/`w2`/`w3` (never the fixed `THRESHOLD=0.65`) from real
clinician accept/override outcomes — gated against small samples
(`min_reviews`, default 10, counted as **distinct reviewed cases** — a
case's latest accept/override counts once), fully audited
(`confidence_weight_history`), and never run automatically:

```bash
python -m backend.services.weight_tuner
```

The command first prints how many qualifying outcomes exist (e.g. "3 of
10 needed") and leaves the weights unchanged until there are enough real
reviews. Don't recalibrate from the Mode 2 benchmark: its fusion step is a
placeholder, so the agreement signal is constant and the "after" score only
matches an always-trust baseline (the harness now prints that caveat).

## Async analysis API (live progress)

A case takes from about 10 seconds to several minutes, so the web app no longer holds one request open:

```text
POST /diagnose?async=true        -> 202 {"case_id", "status": "queued", "status_url"}
GET  /diagnose/{case_id}/status  -> {"status": queued|running|completed|failed, "queue_position",
                                     "elapsed_s", "error", "steps": [...], "report": {...} (when completed)}
GET  /diagnose/{case_id}         -> the stored report (unchanged)
```

`steps` is what the pipeline actually reported: `clinical`, `imaging`, `rag`, `evidence`, `fusion`, and an
`optimizer` step (with per-round `iterations`) only when the first answer was below the confidence threshold. Each
has a status (`pending|running|done|failed|skipped`), a short result such as "9 live item(s)", and a duration. A
failed step never carries the raw provider error. The plain `POST /diagnose` is unchanged (synchronous, returns the
report), and the scripts under `scripts/` use it.

Limits: job state is in this process's memory, so run **one** server process (the Docker image does; `--reload` or a
restart drops running jobs, and the status endpoint then says the run was interrupted). Concurrency is capped by
`DIAGNOSE_WORKERS` (default 2); further cases wait in a queue and report their position. Finished jobs are pruned
after an hour or beyond 200; the report itself stays in the database.
