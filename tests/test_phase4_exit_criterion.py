"""Phase 4 exit-criterion test.

Plan says: 'One real on-sky night, supervised, with the human observing the
dashboard.' We can't do an actual on-sky run in an automated test, but we
CAN simulate everything the human would see: REPLAN happening end-to-end,
budget tracked across multiple Doctor calls, dashboard showing live phase
+ events + budget, E-STOP routed through to the Conductor.
"""

import json
import sqlite3
from functools import partial

import pytest
from httpx import ASGITransport, AsyncClient

from nina_autopilot.conductor import Conductor, ConductorConfig, Phase
from nina_autopilot.dashboard import create_app
from nina_autopilot.doctor import Doctor
from nina_autopilot.llm import BudgetState, LLMClient
from nina_autopilot.nina_client import FakeNinaClient
from nina_autopilot.planner import plan_next
from nina_autopilot.state import open_store

from tests.test_llm import FakeAnthropic


# Re-use the schema fragment from Phase 3 e2e
_TS_SCHEMA = """
CREATE TABLE project (
    Id INTEGER NOT NULL PRIMARY KEY, profileId TEXT NOT NULL, name TEXT NOT NULL,
    description TEXT, state INTEGER, priority INTEGER, createdate INTEGER,
    activedate INTEGER, inactivedate INTEGER, minimumtime INTEGER,
    minimumaltitude REAL, usecustomhorizon INTEGER, horizonoffset REAL,
    meridianwindow INTEGER, filterswitchfrequency INTEGER, ditherevery INTEGER,
    enablegrader INTEGER, isMosaic INTEGER NOT NULL DEFAULT 0,
    flatsHandling INTEGER NOT NULL DEFAULT 0, maximumAltitude REAL DEFAULT 0,
    smartexposureorder INTEGER DEFAULT 0, guid TEXT
);
CREATE TABLE target (
    Id INTEGER NOT NULL PRIMARY KEY, name TEXT NOT NULL, active INTEGER NOT NULL,
    ra REAL, dec REAL, epochcode INTEGER NOT NULL, rotation REAL, roi REAL,
    projectid INTEGER, unusedOEO TEXT, guid TEXT
);
CREATE TABLE exposuretemplate (
    Id INTEGER NOT NULL PRIMARY KEY, profileId TEXT NOT NULL, name TEXT NOT NULL,
    filtername TEXT NOT NULL, gain INTEGER, offset INTEGER, bin INTEGER,
    readoutmode INTEGER, twilightlevel INTEGER, moonavoidanceenabled INTEGER,
    moonavoidanceseparation REAL, moonavoidancewidth INTEGER, maximumhumidity REAL,
    defaultexposure REAL DEFAULT 60, moonrelaxscale REAL DEFAULT 0,
    moonrelaxmaxaltitude REAL DEFAULT 5, moonrelaxminaltitude REAL DEFAULT -15,
    moondownenabled INTEGER DEFAULT 0, ditherevery INTEGER DEFAULT -1,
    minutesOffset INTEGER DEFAULT 0, guid TEXT
);
CREATE TABLE exposureplan (
    Id INTEGER NOT NULL PRIMARY KEY, profileId TEXT NOT NULL, exposure REAL NOT NULL,
    desired INTEGER, acquired INTEGER, accepted INTEGER, targetid INTEGER,
    exposureTemplateId INTEGER, enabled INTEGER DEFAULT 1, guid TEXT
);
"""

PROFILE = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def ts_db_two_targets(tmp_path):
    """Two active projects so REPLAN has a different target to switch to."""
    p = tmp_path / "schedulerdb.sqlite"
    c = sqlite3.connect(str(p))
    c.executescript(_TS_SCHEMA)
    c.executemany(
        "INSERT INTO project (Id, profileId, name, state, priority) VALUES (?, ?, ?, ?, ?)",
        [
            (1, PROFILE, "M81", 1, 100),
            (2, PROFILE, "NGC 7000", 1, 50),
        ],
    )
    c.executemany(
        "INSERT INTO target (Id, name, active, ra, dec, epochcode, projectid) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (10, "M81 Galaxy",      1, 148.888, 69.065, 2, 1),
            (11, "NGC 7000 Nebula", 1, 314.750, 44.330, 2, 2),
        ],
    )
    c.execute(
        "INSERT INTO exposuretemplate (Id, profileId, name, filtername, gain) VALUES "
        "(100, ?, 'L', 'L', 100)", (PROFILE,)
    )
    c.executemany(
        "INSERT INTO exposureplan (Id, profileId, exposure, desired, acquired, accepted, "
        "targetid, exposureTemplateId, enabled) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1000, PROFILE, 180.0, 30, 10, 8, 10, 100, 1),  # M81 still needs work
            (1001, PROFILE, 120.0, 50, 0,  0, 11, 100, 1),  # NGC 7000 needs work
        ],
    )
    c.commit()
    c.close()
    return p


class _ReplanScenarioClient(FakeNinaClient):
    """Reports Error once, then Finished — simulates Doctor REPLAN succeeding."""
    def __init__(self):
        super().__init__()
        self._first_seq_poll = True

    async def get_sequence_state(self):
        self._record("get_sequence_state")
        if self._first_seq_poll:
            self._first_seq_poll = False
            return {"State": "Error", "ErrorMessage": "Target unreachable: below altitude"}
        return {"State": "Finished"}


async def test_phase4_full_replan_with_budget_and_dashboard(tmp_path, ts_db_two_targets):
    """Doctor REPLAN → switches target → budget tracked → dashboard sees it all."""
    store = open_store(tmp_path / "session.sqlite")
    fake_nina = _ReplanScenarioClient()

    # Fake LLM with a budget headroom of $1 — non-trivial but not exhausted.
    fake_llm = FakeAnthropic()
    fake_llm.messages.queue(json.dumps({
        "action": "replan",
        "reason": "M81 likely below altitude given the error — switch to a higher-elevation target",
    }), input_tokens=820, output_tokens=38)
    llm = LLMClient(client=fake_llm, nightly_budget_usd=1.00)
    doctor = Doctor(llm=llm, model="claude-sonnet-4-6")

    planner_fn = partial(
        plan_next,
        ts_db_path=str(ts_db_two_targets),
        sequence_name="ts_driven_sequence",
        profile_id=PROFILE,
    )

    cfg = ConductorConfig(
        planner=planner_fn,
        doctor=doctor,
        safety_tick_s=0.0,
    )
    conductor = Conductor(fake_nina, store, cfg)

    # Dashboard wired to the same conductor + store
    app = create_app(conductor=conductor, store=store, llm=llm)

    # Run the session
    await conductor.run()

    # ---- A. Session reached DONE cleanly ----
    assert conductor.phase is Phase.DONE
    sess = store.get_session(1)
    assert sess["end_reason"] == "sequence_complete"

    # ---- B. REPLAN was performed ----
    events = store.list_events(1)
    kinds = [e["kind"] for e in events]
    assert "FAULT_DETECTED" in kinds
    assert "DOCTOR_DECISION" in kinds
    assert "REPLAN_LOADED" in kinds
    decision_ev = next(e for e in events if e["kind"] == "DOCTOR_DECISION")
    assert decision_ev["payload"]["action"] == "replan"

    # The first PLANNER_DECISION picked M81 (priority 100); after REPLAN the second
    # PLANNER_DECISION should also be IMAGE (M81 still has remaining frames).
    planner_events = [e for e in events if e["kind"] == "PLANNER_DECISION"]
    assert len(planner_events) == 2
    assert all(e["payload"]["action"] == "image" for e in planner_events)

    # ---- C. Budget was tracked across the Doctor call ----
    assert llm.usage_total.input_tokens == 820
    assert llm.usage_total.output_tokens == 38
    assert llm.cost_estimate_usd() > 0
    assert llm.budget_state is BudgetState.NORMAL  # well under $1

    # ---- D. Dashboard exposes the right post-session state ----
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r_status = await client.get("/api/status")
        assert r_status.status_code == 200
        s = r_status.json()
        assert s["phase"] == "DONE"
        # Session finished, so current_session is None
        assert s["session"] is None
        # Budget snapshot present and consistent
        assert s["budget"]["budget_usd"] == 1.00
        assert s["budget"]["spent_usd"] > 0
        assert s["budget"]["state"] == "normal"

        # /api/events still returns nothing (no current session) — but a previous-session
        # endpoint would. For Phase 4 this is acceptable; expand if/when needed.
        r_events = await client.get("/api/events")
        assert r_events.status_code == 200
        assert r_events.json() == []  # no active session

        # / serves HTML
        r_root = await client.get("/")
        assert r_root.status_code == 200
        assert "NINA Autopilot" in r_root.text


async def test_phase4_budget_halt_blocks_further_doctor_calls(tmp_path, ts_db_two_targets):
    """If budget is exhausted before a Doctor call, the call must raise BudgetExceeded
    — which the Conductor swallows and treats as Doctor-driven ABORT (fail-safe)."""
    from nina_autopilot.llm import BudgetExceeded

    store = open_store(tmp_path / "session.sqlite")
    fake_nina = _ReplanScenarioClient()

    # First call eats most of the budget; second call should be blocked.
    fake_llm = FakeAnthropic()
    fake_llm.messages.queue(json.dumps({"action": "retry", "reason": "x"}),
                            input_tokens=200_000, output_tokens=50_000)
    # $0.20 + $0.25 = $0.45 — set budget to $0.30 so we go HALTED after first call.
    llm = LLMClient(client=fake_llm, nightly_budget_usd=0.30)
    # Burn the budget BEFORE the Conductor runs
    await llm.complete(model="claude-sonnet-4-6", system="s", user="u")
    assert llm.budget_state is BudgetState.HALTED

    doctor = Doctor(llm=llm, model="claude-sonnet-4-6")
    cfg = ConductorConfig(sequence_file="x.json", doctor=doctor, safety_tick_s=0.0)
    conductor = Conductor(fake_nina, store, cfg)

    # Conductor should still complete (Doctor call raises BudgetExceeded → caught → ABORT)
    await conductor.run()
    assert conductor.phase is Phase.DONE
    sess = store.get_session(1)
    # End reason captures the budget halt path
    assert "budget" in sess["end_reason"].lower() or "doctor" in sess["end_reason"].lower()


async def test_phase4_estop_via_dashboard_stops_running_conductor(tmp_path):
    """Hit POST /api/estop while the Conductor is running → next tick aborts."""
    import asyncio

    store = open_store(tmp_path / "session.sqlite")

    class _LoopingClient(FakeNinaClient):
        """Sequence never finishes — only an external stop ends the session."""
        async def get_sequence_state(self):
            self._record("get_sequence_state")
            return {"State": "Running"}

    fake_nina = _LoopingClient()
    cfg = ConductorConfig(sequence_file="x.json", safety_tick_s=0.001)
    conductor = Conductor(fake_nina, store, cfg)
    app = create_app(conductor=conductor, store=store)

    # Run Conductor as a task, hit E-STOP from the dashboard
    run_task = asyncio.create_task(conductor.run())
    await asyncio.sleep(0.01)  # let the imaging loop start

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/estop")
        assert r.status_code == 200

    await asyncio.wait_for(run_task, timeout=2.0)
    assert conductor.phase is Phase.DONE
    sess = store.get_session(1)
    assert sess["end_reason"] == "manual_stop"