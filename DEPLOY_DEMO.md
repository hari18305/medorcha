# Live demo deployment (for judges)

Purpose: let a judge open one link, enter a shared password, and try MedOrchestrate themselves.
It is a demonstration of a research prototype, not a clinical service.

## What protects it

| Risk | Protection |
|---|---|
| Strangers using the site | One shared password (HTTP basic auth at the front door) |
| Someone looping the app and burning the Gemini quota | `DEMO_MAX_PER_IP_HOUR` (default 6 cases per visitor per hour) and `DEMO_MAX_CASES_TOTAL` (default 150 cases for the whole run) |
| Huge uploads | 8 MB limit with a clear message (backend), 10 MB hard stop (proxy) |
| Cross-site / open API | The browser only ever talks to one address; the backend, Qdrant and the model cache are not published |
| Real patient data | The page says it is a demo; tell judges to use only the supplied or made-up cases |

The limits live in `backend/demo_guard.py` (off unless `DEMO_MODE=on`, so normal use is unchanged) and are covered by
`backend/tests/test_demo_guard.py`. A refused request returns a plain message the app shows as-is.
The case counter resets when the backend restarts.

## What was tested here, and what was not

- Tested: the guard (unit tests, plus against the real backend: 3rd request from one visitor gets the "2 cases per hour"
  message, an oversized upload gets the 413 message, `/health` stays open); the frontend builds with the API under `/api`
  and no `localhost:8000` left in the bundle.
- **Not tested: the Docker/Caddy stack as a whole** (`docker-compose.demo.yml`, `deploy/Caddyfile`). The development
  laptop's C: drive was too full to build the images safely. Do one full dry run on the machine that will host it
  (steps below) and open the site from a phone on mobile data before the review day.

## Set-up (any machine with Docker and ~8 GB RAM, ~10 GB disk)

1. Normal `.env` as for the local demo (Gemini key, `LLM_PROVIDER`, `IMAGING_CROSS_CHECK=on`).
2. Make the shared password hash and the settings file:
   ```bash
   docker run --rm caddy:2-alpine caddy hash-password --plaintext "choose-a-password"
   cp deploy/demo.env.example .env.demo
   ```
   Paste the hash into `.env.demo` as `DEMO_PASSWORD_HASH=...` with every `$` written as `$$`.
3. Start it:
   ```bash
   docker compose -f docker-compose.demo.yml --env-file .env.demo up -d --build
   ```
4. One-time, after the first start (fills the knowledge base; the first case also downloads about 2.4 GB of models, so run
   one warm-up case yourself):
   ```bash
   docker compose -f docker-compose.demo.yml exec backend python -m backend.rag.ingest
   ```
5. Check `http://localhost:8080` asks for the password, then run Case 1 from your demo notes.

## Making it reachable by judges (pick one; each needs an account or a host only you can provide)

- **Your own laptop plus a tunnel (simplest, free).** Install Cloudflare's `cloudflared` and run
  `cloudflared tunnel --url http://localhost:8080`. It prints an `https://....trycloudflare.com` link that works while
  the laptop and Docker are running. Nothing to rent, the link changes each time, and it dies if the laptop sleeps.
- **A small rented VM (steadier).** Any provider with 4 CPU / 8 GB RAM. Point a domain at it, set `SITE_ADDRESS` to that
  domain in `.env.demo` (Caddy then gets an https certificate by itself), open ports 80 and 443.
- A free Hugging Face Space is possible but its 16 GB / 2 CPU free tier makes each case slow; not recommended.

Switch it off after the review: `docker compose -f docker-compose.demo.yml down`. Rotate the Gemini key afterwards.

## Text to give the judges

> Open the link and sign in with user `judge` and the password you were given. Fill in age, sex and a short
> clinical story (or use the sample cases we provide) and optionally attach an X-ray, CT or MRI image. A case takes
> 30 to 60 seconds. The report shows the diagnoses, the evidence, the confidence and whether it was flagged for doctor
> review. This is a research prototype: please do not enter real patient information. Each visitor can run 6 cases an
> hour.
