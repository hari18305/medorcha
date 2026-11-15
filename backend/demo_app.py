"""Single-address demo server: the built website at "/", the API under "/api", behind one shared password.

Run with `uvicorn backend.demo_app:app` (see scripts/start_demo.bat). It is only for sharing a demo through a tunnel
from one machine; normal development keeps using backend.main:app and the Vite dev server.

Needs: DEMO_PASSWORD (refuses to start without it), and a frontend built with VITE_API_BASE_URL=/api.
The per-visitor and total-case limits come from backend/demo_guard.py (DEMO_MODE is switched on here).
"""
from __future__ import annotations

import base64
import hmac
import os
from pathlib import Path

os.environ.setdefault("DEMO_MODE", "on")
os.environ.setdefault("DEMO_TRUST_PROXY", "on")  # a tunnel forwards the visitor address in X-Forwarded-For

from fastapi import FastAPI  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402

from backend.main import app as api_app  # noqa: E402

DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
PASSWORD = os.getenv("DEMO_PASSWORD", "")
USER = os.getenv("DEMO_USER", "judge")

if not PASSWORD:
    raise RuntimeError("Set DEMO_PASSWORD before starting the demo server.")
if not (DIST / "index.html").exists():
    raise RuntimeError("frontend/dist is missing: run `VITE_API_BASE_URL=/api npm run build` in frontend/ first.")


class BasicAuth:
    """Shared-password gate in front of everything (site and API)."""

    def __init__(self, app, user: str, password: str):
        self.app = app
        self.expected = f"{user}:{password}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope["headers"]}
        given = b""
        auth = headers.get(b"authorization", b"")
        if auth[:6].lower() == b"basic ":
            try:
                given = base64.b64decode(auth[6:])
            except Exception:
                given = b""
        if given and hmac.compare_digest(given, self.expected):
            await self.app(scope, receive, send)
            return
        body = b"Password required."
        await send({"type": "http.response.start", "status": 401, "headers": [
            (b"www-authenticate", b'Basic realm="MedOrchestrate demo"'),
            (b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


app = FastAPI(title="MedOrchestrate demo")
app.mount("/api", api_app)
app.mount("/", StaticFiles(directory=DIST, html=True), name="site")
app.add_middleware(BasicAuth, user=USER, password=PASSWORD)
