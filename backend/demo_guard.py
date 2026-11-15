"""Optional guard for a public demo deployment (DEMO_MODE=on). Off by default: nothing changes for normal use.

A live demo shares one LLM key and one CPU machine with whoever opens the link, so this caps the damage a stranger
(or a runaway script) could do, without needing user accounts:

* per-visitor limit on new cases per hour (DEMO_MAX_PER_IP_HOUR, default 6)
* a total cap on cases for the whole run (DEMO_MAX_CASES_TOTAL, default 150), which bounds the Gemini spend
* an upload size limit (DEMO_MAX_UPLOAD_MB, default 8)

Only POST /diagnose is limited (that is the call that runs the models and the LLM). Reviews and status polling are
cheap and stay open. Refusals are JSON {"detail": "..."} so the frontend shows the message as-is.

Behind a reverse proxy set DEMO_TRUST_PROXY=on so the visitor address is read from X-Forwarded-For (last entry, which
is the one the proxy appended); without a proxy the header is ignored, since a client could forge it.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque

HOUR = 3600.0


def demo_mode_enabled() -> bool:
    return os.getenv("DEMO_MODE", "off").strip().lower() in {"1", "on", "true", "yes"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class DemoGuard:
    """Pure ASGI middleware (no body buffering, so uploads stream through untouched)."""

    def __init__(self, app, clock=time.monotonic):
        self.app = app
        self.clock = clock
        self.max_per_ip = _int_env("DEMO_MAX_PER_IP_HOUR", 6)
        self.max_total = _int_env("DEMO_MAX_CASES_TOTAL", 150)
        self.max_upload = _int_env("DEMO_MAX_UPLOAD_MB", 8) * 1024 * 1024
        self.trust_proxy = os.getenv("DEMO_TRUST_PROXY", "off").strip().lower() in {"1", "on", "true", "yes"}
        self._lock = threading.Lock()
        self._hits: dict[str, deque] = defaultdict(deque)
        self._total = 0

    def _client(self, scope) -> str:
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        if self.trust_proxy and b"x-forwarded-for" in headers:
            return headers[b"x-forwarded-for"].decode("latin-1").split(",")[-1].strip() or "unknown"
        client = scope.get("client")
        return client[0] if client else "unknown"

    async def _refuse(self, send, status: int, detail: str, retry_after: int | None = None):
        body = json.dumps({"detail": detail}).encode()
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
        if retry_after:
            headers.append((b"retry-after", str(retry_after).encode()))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    def _admit(self, who: str) -> tuple[bool, str, int | None]:
        now = self.clock()
        with self._lock:
            if self._total >= self.max_total:
                return False, ("This demo has reached its total case limit, to protect the shared AI quota. "
                               "Please ask the presenters to reset it."), None
            hits = self._hits[who]
            while hits and now - hits[0] > HOUR:
                hits.popleft()
            if len(hits) >= self.max_per_ip:
                wait = int(HOUR - (now - hits[0])) + 1
                return False, (f"Demo limit reached: {self.max_per_ip} cases per hour per visitor. "
                               f"Please try again in about {max(1, wait // 60)} minute(s)."), wait
            hits.append(now)
            self._total += 1
        return True, "", None

    def status(self) -> dict:
        with self._lock:
            return {"cases_used": self._total, "cases_cap": self.max_total}

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        root = scope.get("root_path", "")  # when mounted under /api the full path still starts with it
        if root and path.startswith(root):
            path = path[len(root):]
        if scope["type"] != "http" or scope["method"] != "POST" or path.rstrip("/") != "/diagnose":
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = 0
        if length > self.max_upload:
            await self._refuse(send, 413, f"That upload is too large for the demo (limit {self.max_upload // (1024 * 1024)} MB).")
            return
        ok, detail, wait = self._admit(self._client(scope))
        if not ok:
            await self._refuse(send, 429, detail, wait)
            return
        await self.app(scope, receive, send)
