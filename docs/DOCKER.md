# Docker

Runs backend + frontend + a real Qdrant server in containers (development-oriented: bind mounts, live reload).
For sharing a demo with others, see [../DEPLOY_DEMO.md](../DEPLOY_DEMO.md) instead, which needs no Docker.

No local Python/Node install is required. The embedded `data/qdrant_local/` store is not used: the compose `qdrant`
service is a real server (see the `QDRANT_URL` override in `docker-compose.yml`).

```bash
cp .env.example .env
# edit .env: set LLM_PROVIDER and the matching key, e.g. GEMINI_API_KEY (required for real diagnoses;
# without a key `/diagnose` returns a clean 500; see docs/ARCHITECTURE.md)

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
down` stops everything; add `-v` to also drop the `qdrant_data` and `hf_cache` volumes
(re-ingest and a model re-download required after that).

**Tested 2026-11-14 (Docker Desktop 4.88, Windows, WSL2):** build 30 s warm / several minutes cold; backend image
1.0 GB (`torch`/`torchvision` come from the CPU-only wheel index in `backend/Dockerfile`, so the image carries no
CUDA libraries); ingest 3m50s including the first model download (67 chunks); first request on a fresh volume is
slow for the same reason. **Models are cached in the `hf_cache` volume** (BioBERT, MedCPT, BiomedCLIP, about 2.4 GB):
the first run on a machine needs the internet, after that all three load with `HF_HUB_OFFLINE=1`. Settings in `.env`
reach the container (`env_file`), so the demonstration profile (`IMAGING_FINDING_PROMPTS=label`,
`IMAGING_CROSS_CHECK=on`) applies there too. Needs about 10 GB free on the Docker disk; a full system drive freezes the
Docker engine (it hangs without an error message). If Docker Desktop refuses to start with "an unexpected error occurred" after an unclean
shutdown, the cause can be stale `*.sock` files in `%LOCALAPPDATA%\Docker
un` and `%LOCALAPPDATA%\docker-secrets-engine`
that Windows cannot delete; remove them from WSL (`wsl rm ...`) and quit Docker from its tray icon in future instead of killing it.

```bash
docker compose down
```

To confirm the backend is actually reachable through its container
(rather than assuming the healthcheck passing means the same thing):

```bash
curl http://localhost:8000/health
