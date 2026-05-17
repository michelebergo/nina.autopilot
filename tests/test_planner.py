"""Tests for planner.py — Phase 3 algorithmic Planner.

The Planner queries the Target Scheduler v5 DB at session start. It does not
generate sequencer JSON in Phase 3 — the user runs a Target-Scheduler-driven
NINA sequence, and the Planner's job is to verify there's actionable work
and return a structured decision the Conductor can act on.

The LLM layer (re-planning, altitude/moon scoring) arrives later. For now,
this is deterministic Python: TS DB read → pick top-priority target with
remaining frames → return decision.
"""

import sqlite3

import pytest

from nina_autopilot.planner import (
    PlannerAction,
    PlannerDecision,
    plan_next,
)


# Mirror the TS v5 schema (kept compact here — fixture used only by planner tests)
_TS_SCHEMA = """
CREATE TABLE project (
    Id INTEGER NOT NULL PRIMARY KEY,
    profileId TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    state INTEGER,
    priority INTEGER,
    createdate INTEGER,
    activedate INTEGER,
    inactivedate INTEGER,
    minimumtime INTEGER,
    minimumaltitude REAL,
    usecustomhorizon INTEGER,
    horizonoffset REAL,
    meridianwindow INTEGER,
    filterswitchfrequency INTEGER,
    ditherevery INTEGER,
    enablegrader INTEGER,
    isMosaic INTEGER NOT NULL DEFAULT 0,
    flatsHandling INTEGER NOT NULL DEFAULT 0,
    maximumAltitude REAL DEFAULT 0,
    smartexposureorder INTEGER DEFAULT 0,
    guid TEXT
);
CREATE TABLE target (
    Id INTEGER NOT NULL PRIMARY KEY,
    name TEXT NOT NULL,
    active INTEGER NOT NULL,
    ra REAL, dec REAL, epochcode INTEGER NOT NULL,
    rotation REAL, roi REAL, projectid INTEGER,
    unusedOEO TEXT, guid TEXT
);
CREATE TABLE exposuretemplate (
    Id INTEGER NOT NULL PRIMARY KEY,
    profileId TEXT NOT NULL, name TEXT NOT NULL,
    filtername TEXT NOT NULL,
    gain INTEGER, offset INTEGER, bin INTEGER,
    readoutmode INTEGER, twilightlevel INTEGER,
    moonavoidanceenabled INTEGER, moonavoidanceseparation REAL,
    moonavoidancewidth INTEGER, maximumhumidity REAL,
    defaultexposure REAL DEFAULT 60, moonrelaxscale REAL DEFAULT 0,
    moonrelaxmaxaltitude REAL DEFAULT 5, moonrelaxminaltitude REAL DEFAULT -15,
    moondownenabled INTEGER DEFAULT 0, ditherevery INTEGER DEFAULT -1,
    minutesOffset INTEGER DEFAULT 0, guid TEXT
);
CREATE TABLE exposureplan (
    Id INTEGER NOT NULL PRIMARY KEY,
    profileId TEXT NOT NULL, exposure REAL NOT NULL,
    desired INTEGER, acquired INTEGER, accepted INTEGER,
    targetid INTEGER, exposureTemplateId INTEGER,
    enabled INTEGER DEFAULT 1, guid TEXT
);
CREATE TABLE acquiredimage (
    Id INTEGER NOT NULL PRIMARY KEY, projectId INTEGER NOT NULL,
    targetId INTEGER NOT NULL, acquireddate INTEGER,
    filtername TEXT NOT NULL, gradingStatus INTEGER NOT NULL,
    metadata TEXT NOT NULL, rejectreason TEXT, profileId TEXT,
    exposureId INTEGER DEFAULT 0, guid TEXT
);
"""

PROFILE_A = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def empty_ts_db(tmp_path):
    p = tmp_path / "schedulerdb.sqlite"
    c = sqlite3.connect(str(p))
    c.executescript(_TS_SCHEMA)
    c.commit()
    c.close()
    return p


@pytest.fixture
def seeded_ts_db(empty_ts_db):
    c = sqlite3.connect(str(empty_ts_db))
    c.executemany(
        "INSERT INTO project (Id, profileId, name, state, priority) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (1, PROFILE_A, "M81",       1, 100),   # Active, high priority
            (2, PROFILE_A, "NGC 7000",  1, 50),    # Active, lower priority
            (3, PROFILE_A, "Drafty",    0, 200),   # Draft — must be ignored
        ],
    )
    c.executemany(
        "INSERT INTO target (Id, name, active, ra, dec, epochcode, projectid) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (10, "M81 Galaxy",      1, 148.888, 69.065, 2, 1),
            (11, "NGC 7000 Nebula", 1, 314.750, 44.330, 2, 2),
        ],
    )
    c.executemany(
        "INSERT INTO exposuretemplate (Id, profileId, name, filtername, gain) "
        "VALUES (?, ?, ?, ?, ?)",
        [(100, PROFILE_A, "L", "L", 100)],
    )
    c.executemany(
        "INSERT INTO exposureplan (Id, profileId, exposure, desired, acquired, accepted, "
        "targetid, exposureTemplateId, enabled) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1000, PROFILE_A, 180.0, 30, 10, 8, 10, 100, 1),  # M81 still needs work
            (1001, PROFILE_A, 120.0, 50, 0,  0, 11, 100, 1),  # NGC 7000 not started
        ],
    )
    c.commit()
    c.close()
    return empty_ts_db


class TestPlannerEmpty:
    def test_empty_db_returns_no_work(self, empty_ts_db):
        d = plan_next(ts_db_path=str(empty_ts_db), sequence_name="any.json")
        assert d.action is PlannerAction.NO_WORK
        assert d.target is None


class TestPlannerWithData:
    def test_picks_highest_priority_actionable_target(self, seeded_ts_db):
        d = plan_next(ts_db_path=str(seeded_ts_db), sequence_name="ts_driven.json")
        assert d.action is PlannerAction.IMAGE
        assert d.target["name"] == "M81 Galaxy"
        assert d.project["name"] == "M81"

    def test_sequence_name_threaded_through(self, seeded_ts_db):
        d = plan_next(ts_db_path=str(seeded_ts_db), sequence_name="my_ts_sequence.json")
        assert d.sequence_name == "my_ts_sequence.json"

    def test_decision_includes_plan_summary(self, seeded_ts_db):
        """The Conductor logs `summary` to the session DB — must be human-readable."""
        d = plan_next(ts_db_path=str(seeded_ts_db), sequence_name="x.json")
        assert d.summary
        assert "M81" in d.summary
        # Plan summary should mention the L filter and remaining frame count
        assert "L" in d.summary
        assert "20" in d.summary  # 30 desired - 10 acquired

    def test_profile_filter(self, seeded_ts_db):
        """Restricting to a non-existent profile yields NO_WORK."""
        d = plan_next(
            ts_db_path=str(seeded_ts_db),
            sequence_name="x.json",
            profile_id="ffffffff-ffff-ffff-ffff-ffffffffffff",
        )
        assert d.action is PlannerAction.NO_WORK


class TestPlannerDecisionShape:
    def test_no_work_decision_has_no_target_or_plans(self, empty_ts_db):
        d = plan_next(ts_db_path=str(empty_ts_db), sequence_name="x.json")
        assert d.target is None
        assert d.project is None
        assert d.plans == []

    def test_image_decision_lists_remaining_plans(self, seeded_ts_db):
        d = plan_next(ts_db_path=str(seeded_ts_db), sequence_name="x.json")
        assert d.action is PlannerAction.IMAGE
        # M81 has one enabled plan with remaining=20
        assert len(d.plans) == 1
        assert d.plans[0]["template_name"] == "L"
        assert d.plans[0]["remaining"] == 20
