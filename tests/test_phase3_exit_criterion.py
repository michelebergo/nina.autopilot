"""Phase 3 exit-criterion test.

Plan says: 'One full simulated night with synthetic faults, zero human intervention'.

This wires the REAL Planner (against a seeded Target Scheduler SQLite DB),
the REAL Doctor (against a fake Anthropic client so we don't hit the API),
and the Conductor with a FakeNinaClient that injects a plate-solve fault
followed by a clean recovery. Zero human-in-the-loop touchpoints — Planner
picks the target, Conductor loads/starts the sequence, fault appears,
Doctor decides RETRY, Conductor restarts, sequence finishes, close-down
runs cleanly, Discord gets a single INFO summary.
"""

import json
import sqlite3
from functools import partial

import pytest

from nina_autopilot.conductor import Conductor, ConductorConfig, Phase
from nina_autopilot.doctor import Doctor
from nina_autopilot.llm import LLMClient
from nina_autopilot.nina_client import FakeNinaClient
from nina_autopilot.planner import plan_next
from nina_autopilot.state import open_store

from tests.test_llm import FakeAnthropic


# Mirror the TS v5 schema enough for the Planner to walk it.
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
def ts_db(tmp_path):
    p = tmp_path / "schedulerdb.sqlite"
    c = sqlite3.connect(str(p))
    c.executescript(_TS_SCHEMA)
    c.executemany(
        "INSERT INTO project (Id, profileId, name, state, priority) VALUES (?, ?, ?, ?, ?)",
        [(1, PROFILE, "Bode's Galaxy", 1, 100)],
    )
    c.execute(
        "INSERT INTO target (Id, name, active, ra, dec, epochcode, projectid) VALUES "
        "(10, 'M81', 1, 148.888, 69.065, 2, 1)"
    )
    c.execute(
        "INSERT INTO exposuretemplate (Id, profileId, name, filtername, gain) VALUES "
        "(100, ?, 'L', 'L', 100)", (PROFILE,)
    )
    c.execute(
        "INSERT INTO exposureplan (Id, profileId, exposure, desired, acquired, accepted, "
        "targetid, exposureTemplateId, enabled) VALUES "
        "(1000, ?, 180.0, 30, 10, 8, 10, 100, 1)", (PROFILE,)
    )
    c.commit()
    c.close()
    return p


class _RecoveringClient(FakeNinaClient):
    """First sequence-state poll returns Error; subsequent polls return Finished.
    Models 'plate-solve hiccupped on first try, retry succeeded'."""

    def __init__(self):
        super().__init__()
        self._first_poll = True

    async def get_sequence_state(self):
        self._record("get_sequence_state")
        if self._first_poll:
            self._first_poll = False
            return {
                "State": "Error",
                "ErrorMessage": "Plate solve failed (ASTAP): solver returned no match",
            }
        return {"State": "Finished"}


async def test_phase3_full_night_with_synthetic_fault_zero_human_intervention(tmp_path, ts_db):
    # ---- Setup: store + fake NINA + real Planner + real Doctor (fake LLM) ----
    store = open_store(tmp_path / "session.sqlite")
    fake_nina = _RecoveringClient()

    # Fake LLM: Doctor sees one fault → returns RETRY
    fake_llm = FakeAnthropic()
    fake_llm.messages.queue(json.dumps({
        "action": "retry",
        "reason": "First plate-solve attempt failed — most likely a transient cloud or bright moon. Retry.",
    }), input_tokens=850, output_tokens=42)
    llm = LLMClient(client=fake_llm)
    doctor = Doctor(llm=llm, model="claude-sonnet-4-6")

    # Real Planner bound to the seeded TS DB
    planner = partial(
        plan_next,
        ts_db_path=str(ts_db),
        sequence_name="ts_driven_sequence.json",
        profile_id=PROFILE,
    )

    cfg = ConductorConfig(
        planner=planner,
        doctor=doctor,
        safety_tick_s=0.0, wait_for_running_seconds=0.0,
        max_retries=2,
    )
    conductor = Conductor(fake_nina, store, cfg)

    # ---- Run the simulated night ----
    await conductor.run()

    # ---- A. The session ended in DONE (full happy-path close-down ran) ----
    assert conductor.phase is Phase.DONE
    sess = store.get_session(1)
    assert sess["ended_at"] is not None
    assert sess["end_reason"] == "sequence_complete"

    # ---- B. Planner did the target selection (no human input) ----
    events = store.list_events(1)
    planner_ev = next(e for e in events if e["kind"] == "PLANNER_DECISION")
    assert planner_ev["payload"]["action"] == "image"
    assert planner_ev["payload"]["target"]["name"] == "M81"
    assert planner_ev["payload"]["sequence_name"] == "ts_driven_sequence.json"

    # ---- C. Conductor loaded the Planner's sequence (NOT a hand-written file) ----
    load_call = next(c for c in fake_nina.calls if c.method == "load_sequence")
    assert load_call.kwargs["name"] == "ts_driven_sequence.json"

    # ---- D. Doctor diagnosed the fault and decided RETRY ----
    fault_ev = next(e for e in events if e["kind"] == "FAULT_DETECTED")
    assert fault_ev["payload"]["fault_type"] == "sequence_error"
    assert "Plate solve" in fault_ev["payload"]["fault_message"]

    decision_ev = next(e for e in events if e["kind"] == "DOCTOR_DECISION")
    assert decision_ev["payload"]["action"] == "retry"

    # ---- E. Retry actually happened (start_sequence called twice) ----
    assert fake_nina.call_names().count("start_sequence") == 2

    # ---- F. Close-down chain ran in order ----
    after_recovery = fake_nina.call_names()[
        fake_nina.call_names().index("start_sequence") + 1:
    ]
    # Find LAST close_dome_shutter / park_mount / stop_cooling in order
    cd = [c for c in after_recovery if c in {"close_dome_shutter", "park_mount", "stop_cooling"}]
    assert cd == ["close_dome_shutter", "park_mount", "stop_cooling"]

    # ---- G. The human gets exactly ONE Discord message: info "session complete" ----
    assert len(fake_nina.alerts) == 1
    assert fake_nina.alerts[0]["severity"] == "info"
    assert "sequence finished" in fake_nina.alerts[0]["message"].lower()

    # ---- H. LLM cost tracked (proves Doctor went through the budget channel) ----
    assert llm.usage_total.input_tokens == 850
    assert llm.usage_total.output_tokens == 42
    assert llm.cost_estimate_usd() > 0


async def test_phase3_doctor_abort_decision_ends_session_cleanly(tmp_path, ts_db):
    """Same setup but Doctor decides ABORT — close-down runs, ALERT severity, no panic."""
    store = open_store(tmp_path / "session.sqlite")
    fake_nina = _RecoveringClient()

    fake_llm = FakeAnthropic()
    fake_llm.messages.queue(json.dumps({
        "action": "abort",
        "reason": "Plate-solve failure pattern indicates equipment misalignment; cannot continue safely.",
    }))
    llm = LLMClient(client=fake_llm)
    doctor = Doctor(llm=llm, model="claude-sonnet-4-6")

    planner = partial(
        plan_next, ts_db_path=str(ts_db),
        sequence_name="x.json", profile_id=PROFILE,
    )
    cfg = ConductorConfig(planner=planner, doctor=doctor, safety_tick_s=0.0, wait_for_running_seconds=0.0)
    conductor = Conductor(fake_nina, store, cfg)
    await conductor.run()

    sess = store.get_session(1)
    assert sess["end_reason"].startswith("doctor_abort")
    # No restart attempted
    assert fake_nina.call_names().count("start_sequence") == 1
    # Discord alert (not panic — safety isn't tripped)
    assert any(a["severity"] == "alert" for a in fake_nina.alerts)
    assert not any(a["severity"] == "panic" for a in fake_nina.alerts)


async def test_phase3_no_work_session_ends_without_load(tmp_path, tmp_path_factory):
    """Empty TS DB → Planner returns NO_WORK → Conductor exits cleanly without imaging."""
    # Fresh empty DB
    empty_db = tmp_path / "empty.sqlite"
    c = sqlite3.connect(str(empty_db))
    c.executescript(_TS_SCHEMA)
    c.commit()
    c.close()

    store = open_store(tmp_path / "session.sqlite")
    fake_nina = FakeNinaClient()

    planner = partial(plan_next, ts_db_path=str(empty_db), sequence_name="x.json")
    cfg = ConductorConfig(planner=planner, safety_tick_s=0.0, wait_for_running_seconds=0.0)
    conductor = Conductor(fake_nina, store, cfg)
    await conductor.run()

    sess = store.get_session(1)
    assert sess["end_reason"] == "no_work"
    assert "load_sequence" not in fake_nina.call_names()
    assert "start_sequence" not in fake_nina.call_names()
    # Discord INFO message for "no actionable targets"
    assert fake_nina.alerts[0]["severity"] == "info"
    assert "no actionable" in fake_nina.alerts[0]["message"].lower()
