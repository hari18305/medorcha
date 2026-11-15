"""The public-demo guard: limits per visitor and in total, upload size, proxy header handling, and that it is inert by default."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.demo_guard import DemoGuard, demo_mode_enabled


def make_client(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    app = FastAPI()

    @app.post("/diagnose")
    def diagnose():
        return {"ok": True}

    @app.get("/diagnose/{case_id}/status")
    def status(case_id: str):
        return {"status": "completed"}

    @app.post("/review/{case_id}")
    def review(case_id: str):
        return {"ok": True}

    app.add_middleware(DemoGuard)
    return TestClient(app)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert demo_mode_enabled() is False
    monkeypatch.setenv("DEMO_MODE", "on")
    assert demo_mode_enabled() is True


def test_per_visitor_hourly_limit(monkeypatch):
    client = make_client(monkeypatch, DEMO_MAX_PER_IP_HOUR=2, DEMO_MAX_CASES_TOTAL=100)
    assert client.post("/diagnose").status_code == 200
    assert client.post("/diagnose").status_code == 200
    blocked = client.post("/diagnose")
    assert blocked.status_code == 429
    assert "per hour" in blocked.json()["detail"]
    assert int(blocked.headers["retry-after"]) > 0


def test_total_cap_applies_to_everyone(monkeypatch):
    client = make_client(monkeypatch, DEMO_MAX_PER_IP_HOUR=50, DEMO_MAX_CASES_TOTAL=2, DEMO_TRUST_PROXY="on")
    assert client.post("/diagnose", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert client.post("/diagnose", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
    capped = client.post("/diagnose", headers={"X-Forwarded-For": "3.3.3.3"})
    assert capped.status_code == 429
    assert "total case limit" in capped.json()["detail"]


def test_proxy_header_separates_visitors_only_when_trusted(monkeypatch):
    trusted = make_client(monkeypatch, DEMO_MAX_PER_IP_HOUR=1, DEMO_MAX_CASES_TOTAL=100, DEMO_TRUST_PROXY="on")
    assert trusted.post("/diagnose", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert trusted.post("/diagnose", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
    assert trusted.post("/diagnose", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429
    # forged leading entries do not help: only the last (proxy-appended) entry counts
    assert trusted.post("/diagnose", headers={"X-Forwarded-For": "9.9.9.9, 1.1.1.1"}).status_code == 429

    untrusted = make_client(monkeypatch, DEMO_MAX_PER_IP_HOUR=1, DEMO_MAX_CASES_TOTAL=100, DEMO_TRUST_PROXY="off")
    assert untrusted.post("/diagnose", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert untrusted.post("/diagnose", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_oversized_upload_refused_without_using_quota(monkeypatch):
    client = make_client(monkeypatch, DEMO_MAX_UPLOAD_MB=1, DEMO_MAX_PER_IP_HOUR=1, DEMO_MAX_CASES_TOTAL=100)
    too_big = client.post("/diagnose", content=b"x" * (1024 * 1024 + 10))
    assert too_big.status_code == 413
    assert "too large" in too_big.json()["detail"]
    assert client.post("/diagnose", content=b"small").status_code == 200  # the refused upload cost no quota


def test_other_routes_are_not_limited(monkeypatch):
    client = make_client(monkeypatch, DEMO_MAX_PER_IP_HOUR=1, DEMO_MAX_CASES_TOTAL=1)
    assert client.post("/diagnose").status_code == 200
    for _ in range(10):
        assert client.get("/diagnose/abc/status").status_code == 200
        assert client.post("/review/abc").status_code == 200


def test_window_expires(monkeypatch):
    now = [0.0]
    for key, value in dict(DEMO_MAX_PER_IP_HOUR=1, DEMO_MAX_CASES_TOTAL=100).items():
        monkeypatch.setenv(key, str(value))
    app = FastAPI()

    @app.post("/diagnose")
    def diagnose():
        return {"ok": True}

    app.add_middleware(DemoGuard, clock=lambda: now[0])
    client = TestClient(app)
    assert client.post("/diagnose").status_code == 200
    assert client.post("/diagnose").status_code == 429
    now[0] = 3601.0
    assert client.post("/diagnose").status_code == 200


def test_limits_still_apply_when_the_api_is_mounted_under_api(monkeypatch):
    """backend.demo_app serves the API under /api; the guard must recognise /api/diagnose."""
    monkeypatch.setenv("DEMO_MAX_PER_IP_HOUR", "1")
    monkeypatch.setenv("DEMO_MAX_CASES_TOTAL", "100")
    inner = FastAPI()

    @inner.post("/diagnose")
    def diagnose():
        return {"ok": True}

    inner.add_middleware(DemoGuard)
    outer = FastAPI()
    outer.mount("/api", inner)
    client = TestClient(outer)
    assert client.post("/api/diagnose").status_code == 200
    assert client.post("/api/diagnose").status_code == 429
