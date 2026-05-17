"""Tests for the web dashboard — the human's window into a running session.

The dashboard is a FastAPI app bound to 127.0.0.1 (exposed remotely via
Tailscale, never the open internet). It surfaces phase, current target,
recent events, and the LLM budget, plus a POST /api/estop trigger.

Tests use httpx.AsyncClient against the in-memory ASGI app — no real port.
"""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from nina_autopilot.conductor import Phase
from nina_autopilot.dashboard import create_app
from nina_autopilot.llm import BudgetState, LLMClient
from nina_autopilot.state import open_store


class _FakeConductor:
    """Minimal stand-in matching the DashboardConductor protocol."""

    def __init__(self, phase: Phase = Phase.IMAGING):
        self._phase = phase
        self.stop_calls = 0

    @property
    def phase(self) -> Phase:
        return self._phase

    async def request_stop(self) -> None:
        self.stop_calls += 1


@pytest.fixture
def store(tmp_path):
    return open_store(tmp_path / "session.sqlite")


@pytest.fixture
async def client_factory():
    clients = []

    def make(app):
        c = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        clients.append(c)
        return c

    yield make
    for c in clients:
        await c.aclose()


class TestStatusEndpoint:
    async def test_returns_phase(self, store, client_factory):
        store.start_session("x.json")
        store.set_phase(1, "IMAGING")
        cond = _FakeConductor(phase=Phase.IMAGING)
        app = create_app(conductor=cond, store=store)
        client = client_factory(app)

        r = await client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["phase"] == "IMAGING"

    async def test_returns_active_session_summary(self, store, client_factory):
        store.start_session("nightly.json")
        cond = _FakeConductor()
        app = create_app(conductor=cond, store=store)
        client = client_factory(app)

        r = await client.get("/api/status")
        data = r.json()
        assert data["session"] is not None
        assert data["session"]["sequence_file"] == "nightly.json"
        assert data["session"]["ended_at"] is None

    async def test_no_active_session_returns_null(self, store, client_factory):
        # Session started and ended → no active session
        sid = store.start_session("x.json")
        store.end_session(sid, reason="sequence_complete")
        cond = _FakeConductor(phase=Phase.DONE)
        app = create_app(conductor=cond, store=store)
        client = client_factory(app)

        r = await client.get("/api/status")
        data = r.json()
        assert data["session"] is None

    async def test_status_includes_budget_when_llm_provided(self, store, client_factory):
        # Fake LLM with non-zero spend
        class _FakeUsage:
            input_tokens = 100
            output_tokens = 50
            cache_creation_input_tokens = 0
            cache_read_input_tokens = 0

        class _FakeResp:
            content = [type("B", (), {"type": "text", "text": "x"})]
            stop_reason = "end_turn"
            usage = _FakeUsage()

        class _FakeMsgs:
            async def create(self, **kwargs):
                return _FakeResp()

        class _FakeClient:
            messages = _FakeMsgs()

        llm = LLMClient(client=_FakeClient(), nightly_budget_usd=2.00)
        await llm.complete(model="claude-haiku-4-5", system="s", user="u")
        cond = _FakeConductor()
        app = create_app(conductor=cond, store=store, llm=llm)
        client = client_factory(app)

        r = await client.get("/api/status")
        data = r.json()
        assert "budget" in data
        assert data["budget"]["budget_usd"] == 2.00
        assert data["budget"]["state"] in {"normal", "demoted", "halted"}
        assert data["budget"]["spent_usd"] > 0


class TestEventsEndpoint:
    async def test_returns_events_for_current_session(self, store, client_factory):
        sid = store.start_session("x.json")
        store.record_event(sid, kind="PHASE_CHANGE", payload={"from": "BOOT", "to": "IMAGING"})
        store.record_event(sid, kind="FAULT_DETECTED", payload={"fault_message": "x"})

        cond = _FakeConductor()
        app = create_app(conductor=cond, store=store)
        client = client_factory(app)

        r = await client.get("/api/events")
        assert r.status_code == 200
        events = r.json()
        assert len(events) == 2
        kinds = [e["kind"] for e in events]
        assert "PHASE_CHANGE" in kinds and "FAULT_DETECTED" in kinds

    async def test_limit_parameter(self, store, client_factory):
        sid = store.start_session("x.json")
        for i in range(20):
            store.record_event(sid, kind=f"E{i}", payload={})
        cond = _FakeConductor()
        app = create_app(conductor=cond, store=store)
        client = client_factory(app)

        r = await client.get("/api/events?limit=5")
        assert len(r.json()) == 5


class TestEstopEndpoint:
    async def test_estop_triggers_request_stop(self, store, client_factory):
        store.start_session("x.json")
        cond = _FakeConductor()
        app = create_app(conductor=cond, store=store)
        client = client_factory(app)

        r = await client.post("/api/estop")
        assert r.status_code == 200
        assert cond.stop_calls == 1

    async def test_estop_idempotent(self, store, client_factory):
        store.start_session("x.json")
        cond = _FakeConductor()
        app = create_app(conductor=cond, store=store)
        client = client_factory(app)

        await client.post("/api/estop")
        await client.post("/api/estop")
        assert cond.stop_calls == 2


class TestRootPage:
    async def test_root_returns_html(self, store, client_factory):
        cond = _FakeConductor()
        app = create_app(conductor=cond, store=store)
        client = client_factory(app)

        r = await client.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        # Body must contain HTMX-driven status hook and the E-STOP control
        body = r.text
        assert "E-STOP" in body or "e-stop" in body.lower()
        assert "/api/status" in body
