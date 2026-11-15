# Configuration

All settings are environment variables read from `.env` (copy `.env.example`). `.env` is git-ignored: never commit it.
Only the LLM provider and its key are needed for a first run; everything else has a default.

## Variables

| Variable | Required? | Notes |
|---|---|---|
| `LLM_PROVIDER` | **Yes** in practice | `anthropic` (the code default if unset), `bedrock`, `vertex`, `gemini` or `mistral`. `.env.example` sets `gemini`, the provider every run in this repo used. |
| `GEMINI_API_KEY` | **Yes** if `LLM_PROVIDER=gemini` | From aistudio.google.com. A free key works for single cases but its daily quota is too small for batch runs or evaluation (one model allowed 20 requests/day); use a paid key for those. |
| `ANTHROPIC_API_KEY` | **Yes** if `LLM_PROVIDER=anthropic` | Get one from console.anthropic.com. Without a key for the chosen provider the LLM steps fail gracefully but the pipeline won't produce real output. |
| `QDRANT_URL` / `QDRANT_API_KEY` | No | Leave blank to use an embedded local Qdrant store at `data/qdrant_local/` — no server needed. Set these only if you have a real Qdrant Cloud/self-hosted instance. |
| `PUBMED_API_KEY` | No | PubMed, MedlinePlus, Europe PMC and ClinicalTrials.gov all work without a key at this call volume. |
| `CASE_MEMORY_MIN_SCORE` / `VISUAL_MEMORY_MIN_SCORE` | No | Minimum similarity for a past clinician override to be shown as a precedent (defaults 0.6 text / 0.85 visual — see [Case memory](ARCHITECTURE.md#case-memory-text--visual)). |
| `FINDING_RELATIVE_THRESHOLD` | No | Imaging: a finding label beyond the top one is reported only if it is at least this fraction (default 0.6) as strong as the top label, and above a 0.15 floor. Drops weak secondary labels that were leaking spurious diagnoses into the differential. A prototype value, not validated. |
| `EVIDENCE_LLM_SUMMARIES` | No | `false` (default): PubMed abstracts are truncated, with no LLM call. `true`: one LLM call per abstract, concurrent, 12s budget. Keep it off on a free tier — it uses most of the daily quota. |
| `RERANK_MODEL` | No | The cross-encoder reranker (default `ncbi/MedCPT-Cross-Encoder`, biomedical, **~440 MB downloaded on first use** on top of the other models; set `cross-encoder/ms-marco-MiniLM-L-6-v2` for the original ~90 MB general-domain model). On 14 real agent queries (`python -m scripts.eval_retrieval`) MedCPT beats MS MARCO: MRR@5 0.827 → 0.863 and hit@1 10 → 11/14 against MS MARCO with RRF ordering (0.768 / 8 with its plain order), but the subdural-hematoma passage for the anticoagulated-fall case drops from 2nd to 3rd. n=14, developer-judged: a regression check, not a benchmark. |
| `RERANK_SCORE_CENTER` / `RERANK_SCORE_SCALE` | No | Calibration of the reranker's 0–1 relevance score (the S_rag confidence signal): `sigmoid((logit − center) / scale)`. Per-model defaults: MedCPT −12.5 / 3.5, MS MARCO −7 / 2. Hand-fitted; the MedCPT pair reproduces the MS MARCO pipeline's *average* S_rag on 14 real queries (0.659) and 6 off-topic ones (0.284), but per-query S_rag differs by up to ~0.3. Re-check with `python -m backend.rag.calibrate_reranker` if the corpus or model changes. |
| `RERANK_ORDER` | No | How the reranked top-5 is *ordered* (default `ce`: cross-encoder order). `rrf` fuses the BioBERT rank with the cross-encoder rank (reciprocal rank fusion, k=60). Which 5 passages are chosen, and so S_rag, is the same either way. With MedCPT, `rrf` adds a little (MRR@5 0.863 → 0.895, hit@1 11 → 12) but moves the CASE009 subdural passage from 3rd to 5th, the edge of the top 5, so it is off by default; with MS MARCO it helped more. |
| `IMAGING_CROSS_CHECK` | No | `off` (default) or `on`. When on, the multimodal LLM reads the raw image with the same closed label list BiomedCLIP uses; if its label differs from BiomedCLIP's, the read is marked **contested** and kept out of retrieval, case memory and the differential (the image itself still goes to the fusion step). Targets confident wrong zero-shot reads that the inconclusive-read rule cannot see. One extra LLM call per image case; skipped for providers without vision; fails open on any error. Replayed offline on the 88 saved benchmark reads it flagged 25 of 29 confident wrong zero-shot reads and 5 of 50 confident right ones (optimistic: public images, closed list). Not part of the reported evaluation. |
| `IMAGING_FINDING_PROMPTS` | No | Wording of the finding captions BiomedCLIP scores an image against (default `showing`: e.g. "head CT scan showing intracranial hemorrhage, hyperdense blood"; `label` = the bare label). Chosen on the dev half of the real-image benchmark and judged once on a held-out half: test top-1 7/25 → 12/25, McNemar p = 0.18 (not significant). It raises sensitivity but lowers specificity (normal films 7/9 → 5/9). See `../data/benchmarks/imaging_real/PROMPT_EXPERIMENT.md`. **For the demonstration set `label`**: with `showing`, two of the three real demo CTs (normal head slices) are read as "sinus abnormality"; with `label` all seven real demo images read correctly (../DEMO_READINESS.md). |
| `DATABASE_URL` | No | Defaults to `sqlite:///./medorchestrate.db`; created automatically on first run. |
| `LANGFUSE_ENABLED` | No | Defaults to `false`. Leave off unless you have Langfuse credentials. |

## Switching LLM provider

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
served differently** — see [LLM provider choice](ARCHITECTURE.md#llm-provider-choice) before
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
