"""Phase 4.2 exit-criterion test.

Wires ALL FOUR agents into a simulated night:
  - Planner   (algorithmic — reads a seeded Target Scheduler DB)
  - Doctor    (LLM — fake Anthropic, returns RETRY)
  - Operator  (rule-based — runs on every new sub)
  - Scout     (rule-based — runs on every safety tick)

The session walks through: planner picks a target → sequence loads/starts →
clean sub captured → operator records ACCEPT → cloud cover rises (scout WARN,
logged but not panicked) → fault injected → doctor RETRY → recovery → finish.

Every transition is observable via the session event log and the dashboard,
which is the Phase 4 promise: 'one real on-sky night, supervised, with the
human observing the dashboard' — now with the agents that make supervision
actually useful.
"""

import json
import sqlite3
from functools import partial

import pytest

from nina_autopilot.conductor import Conductor, ConductorConfig, Phase
from nina_autopilot.doctor import Doctor
from nina_autopilot.llm import LLMClient
from nina_autopilot.nina_client import FakeNinaClient, SubFrameStats
from nina_autopilot.operator import Operator
from nina_autopilot.planner import plan_next
from nina_autopilot.safety import SafetyReading
from nina_autopilot.scout import Scout
from nina_autopilot.state import open_store

from tests.test_llm import FakeAnthropic


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
    c.execute(
        "INSERT INTO project (Id, profileId, name, state, priority) VALUES (1, ?, 'M81 Project', 1, 100)",
        (PROFILE,),
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


class _FullNightClient(FakeNinaClient):
    """Scripted client that walks through a representative night:

      poll 1: safety SAFE     · seq Running · sub 1 stats (clean)
      poll 2: safety WARN     · seq Running · sub 2 stats (clean)
              (high humidity — Scout logs WARN but does NOT abort)
      poll 3: safety SAFE     · seq Error   · plate-solve message
              (Doctor: RETRY → stop+restart)
      poll 4: safety SAFE     · seq Finished · close-down
    """

    def __init__(self):
        super().__init__()
        self._safety_polls = 0
        self._seq_polls = 0

    async def get_safety_reading(self):
        self._safety_polls += 1
        if self._safety_polls == 2:
            return SafetyReading(humidity_pct=97.0)  # WARN — humidity above 95
        return SafetyReading(humidity_pct=60.0)

    async def get_sequence_state(self):
        self._record("get_sequence_state")
        self._seq_polls += 1
        if self._seq_polls == 3:
            return {"State": "Error", "ErrorMessage": "Plate solve failed (ASTAP)"}
        if self._seq_polls >= 4:
            return {"State": "Finished"}
        return {"State": "Running"}


async def test_phase4_2_full_night_with_all_four_agents(tmp_path, ts_db):
    store = open_store(tmp_path / "session.sqlite")
    fake = _FullNightClient()
    fake.sub_stats_queue = [
        {"index": 1, "stats": SubFrameStats(hfr=2.4, star_count=600, guide_rms_total=0.5,
                                             filter_name="L", exposure_s=180.0)},
        {"index": 2, "stats": SubFrameStats(hfr=2.5, star_count=620, guide_rms_total=0.55,
                                             filter_name="L", exposure_s=180.0)},
    ]

    # Real Doctor against fake Anthropic — returns RETRY for the plate-solve fault
    fake_llm = FakeAnthropic()
    fake_llm.messages.queue(json.dumps({
        "action": "retry",
        "reason": "First plate-solve attempt failed — likely transient, retry.",
    }), input_tokens=830, output_tokens=40)
    llm = LLMClient(client=fake_llm, nightly_budget_usd=2.00)
    doctor = Doctor(llm=llm, model="claude-sonnet-4-6")

    planner_fn = partial(
        plan_next,
        ts_db_path=str(ts_db),
        sequence_name="ts_driven_sequence",
        profile_id=PROFILE,
    )

    cfg = ConductorConfig(
        planner=planner_fn,
        doctor=doctor,
        operator=Operator(),
        scout=Scout(),
        safety_tick_s=0.0,
    )
    conductor = Conductor(fake, store, cfg)
    await conductor.run()

    events = store.list_events(1)
    kinds = [e["kind"] for e in events]

    # ---- A. Session reached DONE via clean recovery ----
    assert conductor.phase is Phase.DONE
    assert store.get_session(1)["end_reason"] == "sequence_complete"

    # ---- B. Planner ran and picked M81 ----
    planner_ev = next(e for e in events if e["kind"] == "PLANNER_DECISION")
    assert planner_ev["payload"]["target"]["name"] == "M81"

    # ---- C. Operator evaluated both subs (separate OPERATOR_DECISION per index) ----
    op_events = [e for e in events if e["kind"] == "OPERATOR_DECISION"]
    indices = sorted(e["payload"]["sub_index"] for e in op_events)
    assert indices == [1, 2]
    for ev in op_events:
        # Both subs were nominal — Operator should ACCEPT both
        assert ev["payload"]["action"] == "accept"

    # ---- D. Scout logged at least the initial observation; high-humidity tick is WARN ----
    scout_events = [e for e in events if e["kind"] == "SCOUT_SUMMARY"]
    assert len(scout_events) >= 1
    # The humidity-tick summary records the cloud→humid change with WARN severity
    assert any(e["payload"]["severity"] == "warn" for e in scout_events), \
        f"Expected at least one WARN scout summary, got: {[e['payload']['severity'] for e in scout_events]}"

    # ---- E. Doctor caught the fault and decided RETRY ----
    fault_ev = next(e for e in events if e["kind"] == "FAULT_DETECTED")
    assert "Plate solve" in fault_ev["payload"]["fault_message"]
    decision_ev = next(e for e in events if e["kind"] == "DOCTOR_DECISION")
    assert decision_ev["payload"]["action"] == "retry"

    # ---- F. Sequence was restarted exactly once (initial + 1 retry) ----
    assert fake.call_names().count("start_sequence") == 2

    # ---- G. Close-down ran in order, single INFO Discord message ----
    assert any(c == "close_dome_shutter" for c in fake.call_names())
    assert any(c == "park_mount" for c in fake.call_names())
    final_alerts = [a for a in fake.alerts if a["severity"] == "info"]
    assert len(final_alerts) >= 1, "Expected at least one INFO Discord summary at end of session"

    # ---- H. LLM budget tracked (Doctor went through the budget channel) ----
    assert llm.usage_total.input_tokens == 830
    assert llm.cost_estimate_usd() > 0


async def test_phase4_2_operator_flags_bad_sub_in_log(tmp_path, ts_db):
    """A single bad sub (high HFR) → OPERATOR_DECISION records RESHOOT in the log."""
    store = open_store(tmp_path / "session.sqlite")

    class _OneBadSubClient(FakeNinaClient):
        def __init__(self):
            super().__init__()
            self._seq_polls = 0

        async def get_sequence_state(self):
            self._record("get_sequence_state")
            self._seq_polls += 1
            return {"State": "Finished"} if self._seq_polls >= 2 else {"State": "Running"}

    fake = _OneBadSubClient()
    fake.sub_stats_queue = [
        {"index": 99, "stats": SubFrameStats(hfr=6.5, star_count=120, guide_rms_total=0.6)},
    ]

    cfg = ConductorConfig(
        sequence_file="x.json",
        operator=Operator(),  # default hfr_max=5.0 → 6.5 triggers RESHOOT
        safety_tick_s=0.0,
    )
    conductor = Conductor(fake, store, cfg)
    await conductor.run()

    op_ev = next(e for e in store.list_events(1) if e["kind"] == "OPERATOR_DECISION")
    assert op_ev["payload"]["action"] == "reshoot"
    assert op_ev["payload"]["sub_index"] == 99
    assert op_ev["payload"]["metrics"]["hfr"] == 6.5


async def test_phase4_2_scout_alert_severity_posts_discord(tmp_path):
    """A safety-monitor flip → Scout ALERT → Discord alert (separately from safety panic)."""
    store = open_store(tmp_path / "session.sqlite")

    class _SafetyFlipClient(FakeNinaClient):
        def __init__(self):
            super().__init__()
            self._safety_polls = 0
            self._seq_polls = 0

        async def get_safety_reading(self):
            self._safety_polls += 1
            return SafetyReading(safety_is_safe=(self._safety_polls == 1))

        async def get_sequence_state(self):
            self._record("get_sequence_state")
            self._seq_polls += 1
            return {"State": "Running"}  # never finishes; scout/safety must end it

    fake = _SafetyFlipClient()
    cfg = ConductorConfig(sequence_file="x.json", scout=Scout(), safety_tick_s=0.0)
    conductor = Conductor(fake, store, cfg)
    await conductor.run()

    # Two alerts: scout 'alert' (severity) + safety supervisor 'panic'
    severities = [a["severity"] for a in fake.alerts]
    assert "alert" in severities  # scout-driven
    assert "panic" in severities  # safety supervisor close-down
    # Order: scout ALERT must have fired BEFORE the safety panic
    assert severities.index("alert") < severities.index("panic")
