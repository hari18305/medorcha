# Live demo for judges: your laptop + a free tunnel (no Docker, no cost)

One program serves the website and the API on one address (`backend/demo_app.py`), behind a shared password and the
demo limits. A free tunnel then gives that address a public https link. Nothing is installed on a server and nothing is
rented. The link works only while this laptop is on, awake and online.

## What protects it

| Risk | Protection |
|---|---|
| Strangers using the link | One shared password (user `judge`); the link alone shows nothing |
| Someone looping the app and burning the Gemini quota | 6 cases per visitor per hour and 150 cases in total for the run (`DEMO_MAX_PER_IP_HOUR`, `DEMO_MAX_CASES_TOTAL`) |
| Huge uploads | 8 MB limit with a clear message (`DEMO_MAX_UPLOAD_MB`) |
| Open API / other ports | Only port 8000 on this laptop is tunnelled; the browser only talks to that one address |
| Real patient data | Tell judges to use only the supplied or made-up cases |

The limits are in `backend/demo_guard.py` (inactive unless `DEMO_MODE=on`, which `demo_app.py` sets) and the password
gate is in `backend/demo_app.py`. The case counter resets whenever the server restarts.

## Before the review day (one dry run)

1. Close anything else using port 8000 or the local Qdrant store (a normal backend started with `uvicorn backend.main:app`
   locks the store; two backends cannot run at once).
2. Run `scripts\start_demo.bat`. It asks for the password to share, builds the website the first time, and starts the
   server on `http://localhost:8000`. Leave the window open.
3. Open `http://localhost:8000`, sign in as `judge`, and run one warm-up case (the first request after a start is the
   slowest). The laptop must not sleep: Settings > System > Power > Screen and sleep > "When plugged in, put my
   device to sleep" = Never.
4. Start a tunnel in a second terminal. Either one works, both are free and need no account:
   - Already installed on Windows (OpenSSH): `ssh -R 80:localhost:8000 nokey@localhost.run`. It prints an
     `https://xxxx.lhr.life` link.
   - Or Cloudflare's `cloudflared tunnel --url http://localhost:8000` (needs a one-time download of `cloudflared.exe`);
     it prints an `https://xxxx.trycloudflare.com` link.
5. Open that link from a phone on mobile data, sign in, and run a case. If that works, the judges' experience works.
   The link changes every time the tunnel restarts, so start it right before sharing and keep the window open.

## Presenting from the same machine

The demo server is the same pipeline as the normal one. You can present from `http://localhost:8000` in your own
browser (sign in once), so judges and you are on one server. Your own cases count toward the same limits, so for a
10-minute live demo (3 cases) that is fine, but rehearsing first can use up your own 6 an hour: in that case run
`set DEMO_MAX_PER_IP_HOUR=20` before `scripts\start_demo.bat`, and set it back to 6 before sharing the link.

## After the review

Close the tunnel window and the server window (the link dies immediately), and rotate the Gemini key if you want to be
safe.

## Text to give the judges

> Open the link and sign in with user `judge` and the password you were given. Fill in age, sex and a short
> clinical story (or use the sample cases we provide) and optionally attach an X-ray, CT or MRI image. A case takes
> 30 to 60 seconds. The report shows the diagnoses, the evidence, the confidence and whether it was flagged for doctor
> review. This is a research prototype: please do not enter real patient information. Each visitor can run 6 cases an
> hour.

## What was tested

Guard unit tests (including the `/api` mount); the real server: no password and a wrong password get 401, the right one
gets the site and API, the 3rd request in an hour from one visitor is refused with a clear message, and a case submitted
through `/api` returned its status and report. Not tested: the tunnels themselves (starting one publishes the laptop,
which is your call), and a full clean case through the combined server (my test run collided with a normal backend that
was already holding the Qdrant store, which is why step 1 exists).
