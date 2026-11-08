# MedOrchestrate

Clinician-facing, multimodal (text + medical report + MRI image) clinical
decision-support prototype.

- **Backend**: Python 3.11+ (developed against 3.12) + FastAPI
- **Agent orchestration**: CrewAI (Claude as the CrewAI LLM)
- **RAG**: BioBERT sentence embeddings + Qdrant (local embedded fallback if no Qdrant server configured)
- **Imaging**: BiomedCLIP zero-shot classification
- **Live evidence**: PubMed + MedlinePlus (no API key required)
- **Frontend**: React + Vite + Tailwind
- **DB**: SQLite
- **LLM provider**: Anthropic Claude API

### Note on LLM provider choice

The Milestone II design doc (Section 8) lists a multi-provider setup
per agent — GPT-4o-mini, Gemini-2.0-Flash, DeepSeek-R1, Qwen2-72B —
under the stated principle "off-the-shelf LLMs selected per task, no
fine-tuning." This implementation runs every agent (clinical
extraction, RAG reranking, fusion reasoning, the optimizer, and
CrewAI's own orchestration layer) on a single model by default instead
(`LLM_PROVIDER=anthropic` — Claude).

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
Vertex — see "Switching LLM provider" below) is reachable in time.
Gemini specifically is one of the four models this project's own
Milestone II doc originally named, so using it isn't off-design if it
comes to that — but be clear about which provider actually produced a
given run's output rather than presenting Gemini/Mistral output as
Claude's.

## Status

The full pipeline is implemented end to end: clinical extraction →
imaging classification → offline RAG retrieval → live evidence →
**joint image+text evidence fusion** → confidence scoring → iterative
optimization → CrewAI orchestration → FastAPI HTTP layer → SQLite
persistence → structured/redacted logging → a 4-screen React frontend
(submission, processing, report, human-in-the-loop review) → clinician
feedback captured as retrievable case memory (text **and** visual) →
periodic confidence-weight recalibration from real accept/override
outcomes.

**Running the pipeline for real requires a valid `ANTHROPIC_API_KEY`**
(see below) — several steps (clinical extraction, fusion reasoning, the
optimizer, and CrewAI's own agent-decision layer) call Claude directly.
Without a key, `/diagnose` will return a clean `500` rather than hang
or crash (the backend degrades gracefully), but it won't produce a real
diagnosis.

### Joint multimodal fusion

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

### Case memory (text + visual)

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

### Confidence weight recalibration

`backend/services/weight_tuner.py` periodically refits the confidence
formula's `w1`/`w2`/`w3` (never the fixed `THRESHOLD=0.65`) from real
clinician accept/override outcomes — gated against small samples
(`min_reviews`, default 10), fully audited
(`confidence_weight_history`), and never run automatically:

```bash
python -m backend.services.weight_tuner
```

### Evaluation (`backend/tests/run_evaluation.py`)

```bash
python -m backend.tests.run_evaluation              # Mode 1: synthetic in-house cases
python -m backend.tests.run_evaluation --mode2       # Mode 2: real MedQA/PubMedQA subset (needs `pip install datasets` + network)
python -m backend.tests.run_evaluation --mode3       # Mode 3: imaging mini-benchmark + with/without-raw-image fusion comparison
python -m backend.tests.run_evaluation --case-memory-ablation  # required ablation: with vs. without text case memory
python -m backend.tests.run_evaluation --full        # everything above in one run
```

---

## Setup — step by step (VS Code)

### 1. Prerequisites

- [VS Code](https://code.visualstudio.com/)
- Python 3.11+ ([python.org](https://www.python.org/downloads/))
- Node.js 18+ ([nodejs.org](https://nodejs.org/)) — includes npm
- Git
- Recommended VS Code extensions: **Python** (ms-python.python), **Pylance**, and (for the frontend) **ES7+ React/Redux/JS snippets** or just the built-in JS/TS support — none are required, just convenient.

### 2. Open the project

Open the `MedOrchestrate` folder in VS Code (`File > Open Folder…`), or
from a terminal:

```bash
code D:\MedOrchestra
```

### 3. Create and activate the backend virtual environment

Open a terminal in VS Code (`` Ctrl+` ``) at the project root:

```bash
python -m venv venv
```

Activate it:

```bash
# Windows (PowerShell / VS Code integrated terminal default)
venv\Scripts\Activate.ps1

# Windows (Git Bash)
source venv/Scripts/activate

# macOS / Linux
source venv/bin/activate
```

You should see `(venv)` in the terminal prompt. In VS Code, also select
this interpreter so Pylance/IntelliSense and the built-in test runner
use it: `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the one
under `.\venv\Scripts\python.exe` (Windows) or `./venv/bin/python`.

### 4. Install backend dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, CrewAI, the Anthropic SDK, sentence-transformers
(BioBERT), open_clip (BiomedCLIP), torch, Qdrant client, etc. The first
install takes a few minutes (torch and the model libraries are large).

### 5. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Variable | Required? | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | **Yes**, for real runs | Get one from console.anthropic.com. Without it, Claude-dependent steps fail gracefully but the pipeline won't produce real output. |
| `QDRANT_URL` / `QDRANT_API_KEY` | No | Leave blank to use an embedded local Qdrant store at `data/qdrant_local/` — no server needed. Set these only if you have a real Qdrant Cloud/self-hosted instance. |
| `PUBMED_API_KEY` | No | PubMed/MedlinePlus work without a key at this call volume. |
| `DATABASE_URL` | No | Defaults to `sqlite:///./medorchestrate.db`; created automatically on first run. |
| `LANGFUSE_ENABLED` | No | Defaults to `false`. Leave off unless you have Langfuse credentials. |

**`.env` is git-ignored — never commit it.**

#### Switching LLM provider

By default the app calls Claude directly through Anthropic's own API
(`LLM_PROVIDER=anthropic`, needs `ANTHROPIC_API_KEY`). If getting an
Anthropic API key is blocked (billing setup, region, etc.), the same
Claude models are also reachable through **AWS Bedrock** or **Google
Cloud Vertex AI** — set `LLM_PROVIDER=bedrock` or `LLM_PROVIDER=vertex`
in `.env` and fill in that provider's credentials instead (see the
comments in `.env.example`). This is still genuinely Claude, just a
different cloud's auth/billing path — nothing else about the app
changes. See `backend/services/llm_client.py` for exactly what each
provider needs installed (`pip install boto3` for Bedrock,
`pip install google-auth google-cloud-aiplatform` for Vertex).

As a further fallback, `LLM_PROVIDER=gemini` or `LLM_PROVIDER=mistral`
route through Google's Gemini API or Mistral's API instead — both have
low-friction free-tier keys (no cloud account/billing needed:
aistudio.google.com and console.mistral.ai respectively). Unlike
Bedrock/Vertex, **this is a genuinely different model, not Claude
served differently** — see "Note on LLM provider choice" above before
using this for anything you'll present as Claude-based work.

`google-genai` is already in `requirements.txt` — Gemini is this
project's actual confirmed-working provider as of 2026-08 (real
end-to-end runs, not just a documented fallback), so it isn't left
optional the way the others are. Mistral (`pip install mistralai
litellm` — litellm specifically because Mistral isn't one of CrewAI's
natively-bundled providers the way the others are) hasn't been tried
live and stays optional.

**Free-tier rate limits are real and this pipeline is call-heavy**:
CrewAI's orchestration layer makes ~5-6 real calls per case on top of
the direct per-agent calls, so a single case can need 8-20+ real
requests (more with optimizer iterations). Gemini's free tier caps
`gemini-3.5-flash-lite` (this provider's default model) at 15
requests/minute — confirmed live: an un-throttled bulk run across all
17 synthetic test cases failed 82% of them with 429s. This app now
self-throttles to a configurable budget (`GEMINI_RPM_BUDGET` in
`.env`, default 12, see `backend/services/llm_client.py`) *within one
process* — running the evaluation harness and a live demo
simultaneously against the same key will still contend for the same
real quota, since each process's throttling doesn't know about the
other.

### 6. Ingest the offline knowledge base (one-time, required for RAG)

The RAG agent queries a local Qdrant collection that starts empty. Populate it once:

```bash
python -m backend.rag.ingest
```

This downloads the BioBERT embedding model (first run only, a few
minutes) and embeds the 30 knowledge-base passages from
`data/knowledge_base/` into the local Qdrant store. The corpus is
**100% real, citable content** — verbatim MedlinePlus text (U.S. NLM,
public domain) and short PubMed abstract excerpts, each with its real
citation and URL (see `data/knowledge_base/README.md`). Ingest rebuilds
the `medorchestrate_kb` collection from scratch each time (case memory
is separate and untouched), so re-run it any time the corpus changes.
You only need to do this once otherwise (the local store persists on
disk).

### 7. Run the backend

```bash
uvicorn backend.main:app --reload
```

- API base: `http://localhost:8000`
- Health check: `http://localhost:8000/health` → `{"status": "ok"}`
- Interactive API docs: `http://localhost:8000/docs`

Keep this terminal running. Open a **second** VS Code terminal
(`` Ctrl+Shift+` `` or the `+` in the terminal panel) for the frontend.

### 8. Run the frontend

In the new terminal:

```bash
cd frontend
npm install
npm run dev
```

- App: `http://localhost:5173`
- The frontend is pre-configured to call the backend at
  `http://localhost:8000` (see `frontend/src/services/api.js`) — no
  extra config needed as long as both are running locally on their
  default ports.

Open `http://localhost:5173` in a browser to use the app: submit a
case (age/sex + clinical text and/or an image), watch it process, and
view the report. If confidence is below threshold, you'll be routed
automatically to the review screen (also reachable manually via the
"Review Case" button on any report).

### 9. Run the tests (optional but recommended)

Backend, from the project root with the venv active:

```bash
pytest
```

All ~140 tests run offline (no API key or network access required —
external calls are mocked). Should finish in under 20 seconds.

Frontend build check:

```bash
cd frontend
npm run build
```

---

## Docker Quickstart

Runs the whole system (backend + frontend + a real Qdrant server) in
containers — no local Python/Node install required, and no local
`data/qdrant_local/` embedded store (the compose `qdrant` service is a
real server instead; see `docker-compose.yml`'s `QDRANT_URL` override).

```bash
cp .env.example .env
# edit .env and fill in ANTHROPIC_API_KEY (required for real diagnoses;
# see the "Status" section above for what happens without one)

docker compose up --build
```

This starts three services:

- `qdrant` — vector DB, `http://localhost:6333`
- `backend` — FastAPI, `http://localhost:8000` (health check: `/health`)
- `frontend` — static build served on `http://localhost:5173`

**One-time step after the first `up`:** the `qdrant` service starts
empty — populate it exactly like the non-Docker setup, just run inside
the backend container:

```bash
docker compose exec backend python -m backend.rag.ingest
```

Then open `http://localhost:5173` and submit a case. `docker compose
down` stops everything; add `-v` to also drop the `qdrant_data` volume
(re-ingest required after that).

```bash
docker compose down
```

To confirm the backend is actually reachable through its container
(rather than assuming the healthcheck passing means the same thing):

```bash
curl http://localhost:8000/health
```

## Quick reference: what needs to run, and in what order

1. `venv` activated, `pip install -r requirements.txt` done — once.
2. `python -m backend.rag.ingest` — once (populates the local Qdrant store).
3. `uvicorn backend.main:app --reload` — every session, terminal 1.
4. `npm run dev` (from `frontend/`) — every session, terminal 2.
5. Browse to `http://localhost:5173`.

## Project structure

```
MedOrchestrate/
├── backend/
│   ├── main.py                # FastAPI app entry point
│   ├── api/                   # Routes (/diagnose, /review) + API-layer schemas
│   ├── agents/                # clinical, imaging, rag, evidence, fusion, optimizer
│   ├── rag/                   # BioBERT embeddings + Qdrant ingest/retrieve/rerank
│   ├── services/              # orchestrator (CrewAI), confidence, storage (SQLite), logging
│   ├── models/schemas.py      # Shared Pydantic schemas -- single source of truth
│   ├── tests/                 # ~140 offline tests
│   └── Dockerfile
├── frontend/
│   ├── Dockerfile              # multi-stage: npm build -> static `serve`
│   └── src/
│       ├── pages/             # CaseSubmission, Processing, Report, Review
│       ├── components/        # ConfidenceBar, ReportSummary
│       └── services/api.js    # fetch wrapper for the backend
├── data/
│   ├── knowledge_base/        # real, cited corpus (guidelines/, symptom_disease/ JSON) read by backend/rag/ingest.py
│   ├── images/                # synthetic demo images + user uploads
│   ├── test_cases/            # CASE001.json etc.
│   └── qdrant_local/          # local embedded Qdrant store (git-ignored, created by ingest.py)
├── logs/                      # rotating structured log file (git-ignored)
├── medorchestrate.db          # SQLite dev database (git-ignored, auto-created)
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

## Shared contracts

All agents and API routes import their data shapes from
[`backend/models/schemas.py`](backend/models/schemas.py) rather than
redefining them, so `CaseInput`, `ClinicalFindings`, `ImagingFindings`,
`RAGEvidence`, `LiveEvidence`, `FusionResult`, and `FinalReport` stay
consistent across the whole pipeline.
