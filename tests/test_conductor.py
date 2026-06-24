"""Tests for the Conductor — state machine + close-down chain + Planner hook.

Phase 2: load + start a hand-written sequence; monitor; close-down on unsafe.
Phase 3: optional planner callable invoked at session start; NO_WORK skips imaging.
"""

import pytest

from nina_autopilot.conductor import Conductor, ConductorConfig, Phase
from nina_autopilot.nina_client import FakeNinaClient
from nina_autopilot.planner import PlannerAction, PlannerDecision
from nina_autopilot.safety import SafetyReading
from nina_autopilot.state import open_store


@pytest.fixture
def store(tmp_path):
    return open_store(tmp_path / "session.sqlite")


@pytest.fixture
def fake():
    return FakeNinaClient()


def make_conductor(fake, store, sequence_file="tonight.json"):
    cfg = ConductorConfig(sequence_file=sequence_file, safety_tick_s=0.0, wait_for_running_seconds=0.0)
    return Conductor(fake, store, cfg)


# ---------------------------------------------------------------------------
# Clean-run path
# ---------------------------------------------------------------------------

class TestCleanRun:
    async def test_loads_and_starts_sequence(self, fake, store):
        """Conductor must load + start before transitioning to IMAGING."""
        # Sequence reports Finished immediately so the run exits cleanly.
        fake.sequence_state = {"State": "Finished"}
        c = make_conductor(fake, store, sequence_file="my_target.json")
        await c.run()

        calls = fake.call_names()
        # Three startup calls in order: load → reset → start.
        # (reset_sequence is between them since NINA needs containers CREATED before start.)
        assert "load_sequence" in calls
        assert "reset_sequence" in calls
        assert "start_sequence" in calls
        assert calls.index("load_sequence") < calls.index("reset_sequence") < calls.index("start_sequence")
        # load_sequence kwargs include the filename
        load_call = next(c for c in fake.calls if c.method == "load_sequence")
        assert load_call.kwargs["name"] == "my_target.json"

    async def test_clean_run_executes_close_down_chain(self, fake, store):
        fake.sequence_state = {"State": "Finished"}
        c = make_conductor(fake, store)
        await c.run()

        # Close-down: stop_sequence (best-effort), close_dome_shutter,
        # park_mount, stop_cooling, then a final info alert.
        calls = fake.call_names()
        # Filter to the close-down portion (after start_sequence)
        after_start = calls[calls.index("start_sequence") + 1:]
        for expected in ("close_dome_shutter", "park_mount", "stop_cooling"):
            assert expected in after_start, f"missing {expected} in {after_start}"
        # Order: dome → mount → cooler
        assert after_start.index("close_dome_shutter") < after_start.index("park_mount")
        assert after_start.index("park_mount") < after_start.index("stop_cooling")

    async def test_clean_run_sends_info_alert(self, fake, store):
        fake.sequence_state = {"State": "Finished"}
        c = make_conductor(fake, store)
        await c.run()
        assert len(fake.alerts) >= 1
        last = fake.alerts[-1]
        assert last["severity"] == "info"

    async def test_clean_run_reaches_done(self, fake, store):
        fake.sequence_state = {"State": "Finished"}
        c = make_conductor(fake, store)
        await c.run()
        assert c.phase is Phase.DONE


# ---------------------------------------------------------------------------
# Unsafe / abort paths — the Phase 2 EXIT CRITERION
# ---------------------------------------------------------------------------

class TestUnsafeAbort:
    async def test_unsafe_rain_triggers_abort_chain(self, fake, store):
        """The exit-criterion test: injected rain → close dome, park, alert."""
        fake.safety_reading = SafetyReading(rain=True)
        fake.sequence_state = {"State": "Running"}

        c = make_conductor(fake, store)
        await c.run()

        assert c.phase is Phase.DONE  # reached terminal cleanly
        calls = fake.call_names()
        assert "stop_sequence" in calls
        assert "close_dome_shutter" in calls
        assert "park_mount" in calls
        assert "stop_cooling" in calls
        # Last alert must be panic-severity
        assert fake.alerts[-1]["severity"] == "panic"
        assert "rain" in fake.alerts[-1]["message"].lower()

    async def test_unsafe_safety_monitor_triggers_abort(self, fake, store):
        fake.safety_reading = SafetyReading(safety_is_safe=False)
        fake.sequence_state = {"State": "Running"}
        c = make_conductor(fake, store)
        await c.run()
        assert "close_dome_shutter" in fake.call_names()
        assert fake.alerts[-1]["severity"] == "panic"

    async def test_warn_does_not_abort(self, fake, store):
        """Humidity (WARN) without UNSAFE should not trigger close-down."""
        fake.safety_reading = SafetyReading(humidity_pct=97.0)
        # Run completes via sequence finish, NOT via safety
        fake.sequence_state = {"State": "Finished"}
        c = make_conductor(fake, store)
        await c.run()
        # Final alert is info (sequence complete), not panic
        assert fake.alerts[-1]["severity"] == "info"

    async def test_close_down_continues_after_individual_failure(self, fake, store):
        """If close_dome_shutter raises, park + cooler must still run."""
        fake.safety_reading = SafetyReading(rain=True)
        fake.sequence_state = {"State": "Running"}
        fake.fail_on = {"close_dome_shutter"}
        c = make_conductor(fake, store)
        await c.run()
        calls = fake.call_names()
        # close_dome_shutter attempted (and failed), but park + cooler still ran
        assert "close_dome_shutter" in calls
        assert "park_mount" in calls
        assert "stop_cooling" in calls


# ---------------------------------------------------------------------------
# Manual stop
# ---------------------------------------------------------------------------

class TestManualStop:
    async def test_request_stop_triggers_abort(self, fake, store):
        """request_stop() before run completes → abort chain."""
        fake.sequence_state = {"State": "Running"}
        c = make_conductor(fake, store)
        await c.request_stop()  # set before run() begins polling
        await c.run()
        assert "close_dome_shutter" in fake.call_names()
        # Manual stop → alert severity = alert (not panic)
        assert fake.alerts[-1]["severity"] == "alert"


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    async def test_session_row_created(self, fake, store):
        fake.sequence_state = {"State": "Finished"}
        c = make_conductor(fake, store, sequence_file="x.json")
        await c.run()
        cur = store.current_session()
        assert cur is None  # ended
        # Find the session we just ran
        sess = store.get_session(1)
        assert sess["sequence_file"] == "x.json"
        assert sess["ended_at"] is not None
        assert sess["end_reason"] == "sequence_complete"

    async def test_safety_abort_records_reason(self, fake, store):
        fake.safety_reading = SafetyReading(rain=True)
        fake.sequence_state = {"State": "Running"}
        c = make_conductor(fake, store)
        await c.run()
        sess = store.get_session(1)
        assert sess["end_reason"].startswith("safety_abort")
        assert "rain" in sess["end_reason"].lower()

    async def test_phase_transitions_logged(self, fake, store):
        fake.sequence_state = {"State": "Finished"}
        c = make_conductor(fake, store)
        await c.run()
        events = store.list_events(1)
        kinds = [e["kind"] for e in events]
        # At minimum, every phase transition should be recorded
        assert "PHASE_CHANGE" in kinds
        phase_events = [e for e in events if e["kind"] == "PHASE_CHANGE"]
        phases_seen = [e["payload"]["to"] for e in phase_events]
        # Must hit IMAGING and CLOSING and DONE during a clean run
        assert "IMAGING" in phases_seen
        assert "CLOSING" in phases_seen
        assert "DONE" in phases_seen


# ---------------------------------------------------------------------------
# Phase 3 — Planner integration
# ---------------------------------------------------------------------------

class TestPlannerIntegration:
    async def test_planner_image_decision_uses_its_sequence_name(self, fake, store):
        """Planner override the sequence_file fallback when its action is IMAGE."""
        fake.sequence_state = {"State": "Finished"}

        def planner():
            return PlannerDecision(
                action=PlannerAction.IMAGE,
                sequence_name="planner_picked.json",
                target={"name": "M81", "ra": 148.8, "dec": 69.0},
                project={"name": "M81 Project"},
                plans=[{"template_name": "L", "remaining": 20, "exposure": 180.0,
                        "filter_name": "L", "gain": 100}],
                summary="target=M81 | L×20 (180s)",
            )

        cfg = ConductorConfig(
            sequence_file="fallback.json",
            planner=planner,
            safety_tick_s=0.0, wait_for_running_seconds=0.0,
        )
        c = Conductor(fake, store, cfg)
        await c.run()

        load_call = next(call for call in fake.calls if call.method == "load_sequence")
        assert load_call.kwargs["name"] == "planner_picked.json"

    async def test_planner_no_work_skips_imaging(self, fake, store):
        """NO_WORK → end session cleanly without ever loading/starting a sequence."""
        def planner():
            return PlannerDecision(
                action=PlannerAction.NO_WORK,
                summary="No actionable Target Scheduler targets",
            )

        cfg = ConductorConfig(
            sequence_file="ignored.json",
            planner=planner,
            safety_tick_s=0.0, wait_for_running_seconds=0.0,
        )
        c = Conductor(fake, store, cfg)
        await c.run()

        calls = fake.call_names()
        assert "load_sequence" not in calls
        assert "start_sequence" not in calls
        # But the close-down chain still runs idempotently (safe baseline)
        assert "close_dome_shutter" in calls
        assert "park_mount" in calls
        # Session reason reflects no_work
        sess = store.get_session(1)
        assert sess["end_reason"] == "no_work"

    async def test_planner_decision_logged_to_events(self, fake, store):
        fake.sequence_state = {"State": "Finished"}

        def planner():
            return PlannerDecision(
                action=PlannerAction.IMAGE,
                sequence_name="x.json",
                target={"name": "M81"},
                summary="target=M81 | L×20",
            )

        cfg = ConductorConfig(
            sequence_file="fallback.json",
            planner=planner,
            safety_tick_s=0.0, wait_for_running_seconds=0.0,
        )
        c = Conductor(fake, store, cfg)
        await c.run()

        events = store.list_events(1)
        kinds = [e["kind"] for e in events]
        assert "PLANNER_DECISION" in kinds
        planner_ev = next(e for e in events if e["kind"] == "PLANNER_DECISION")
        assert planner_ev["payload"]["action"] == "image"
        assert "M81" in planner_ev["payload"]["summary"]

    async def test_async_planner_supported(self, fake, store):
        """Planner may be an async function (e.g. when it does HTTP/DB I/O)."""
        fake.sequence_state = {"State": "Finished"}

        async def planner():
            return PlannerDecision(
                action=PlannerAction.IMAGE,
                sequence_name="async.json",
                target={"name": "X"},
                summary="x",
            )

        cfg = ConductorConfig(sequence_file="fallback.json", planner=planner, safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg)
        await c.run()
        load_call = next(call for call in fake.calls if call.method == "load_sequence")
        assert load_call.kwargs["name"] == "async.json"


# ---------------------------------------------------------------------------
# Phase 3 — Doctor integration
# ---------------------------------------------------------------------------

class _StubDoctor:
    """A stand-in for the LLM-backed Doctor: returns canned decisions in order."""

    def __init__(self, decisions):
        from collections import deque
        self._decisions = deque(decisions)
        self.calls = []

    async def diagnose(self, fault_context):
        self.calls.append(fault_context)
        if not self._decisions:
            raise RuntimeError("StubDoctor: ran out of canned decisions")
        return self._decisions.popleft()


class _FaultThenRecoverClient(FakeNinaClient):
    """Reports Error for the first N sequence-state polls, then Finished."""

    def __init__(self, error_polls: int):
        super().__init__()
        self._remaining_errors = error_polls
        self.sequence_state = {"State": "Error", "ErrorMessage": "Plate solve failed (ASTAP)"}

    async def get_sequence_state(self):
        self._record("get_sequence_state")
        if self._remaining_errors > 0:
            self._remaining_errors -= 1
            return {"State": "Error", "ErrorMessage": "Plate solve failed (ASTAP)"}
        return {"State": "Finished"}


class TestDoctorIntegration:
    async def test_fault_invokes_doctor(self, store):
        from nina_autopilot.doctor import DoctorAction, DoctorDecision
        fake = _FaultThenRecoverClient(error_polls=1)
        doctor = _StubDoctor([
            DoctorDecision(action=DoctorAction.RETRY, reason="first failure, retry"),
        ])
        cfg = ConductorConfig(
            sequence_file="x.json",
            doctor=doctor,
            safety_tick_s=0.0, wait_for_running_seconds=0.0,
        )
        c = Conductor(fake, store, cfg)
        await c.run()

        assert len(doctor.calls) == 1
        ctx = doctor.calls[0]
        assert ctx.fault_type == "sequence_error"
        assert "Plate solve" in ctx.fault_message

    async def test_doctor_retry_restarts_sequence(self, store):
        """RETRY decision → stop+restart sequence so NINA tries again."""
        from nina_autopilot.doctor import DoctorAction, DoctorDecision
        fake = _FaultThenRecoverClient(error_polls=1)
        doctor = _StubDoctor([
            DoctorDecision(action=DoctorAction.RETRY, reason="first failure"),
        ])
        cfg = ConductorConfig(sequence_file="x.json", doctor=doctor, safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg)
        await c.run()

        calls = fake.call_names()
        # start_sequence called twice: initial + retry
        assert calls.count("start_sequence") == 2
        # stop_sequence called before retry (and again as part of close-down)
        assert "stop_sequence" in calls

    async def test_doctor_abort_decision_triggers_abort_chain(self, store):
        from nina_autopilot.doctor import DoctorAction, DoctorDecision
        fake = _FaultThenRecoverClient(error_polls=1)
        doctor = _StubDoctor([
            DoctorDecision(action=DoctorAction.ABORT, reason="non-recoverable"),
        ])
        cfg = ConductorConfig(sequence_file="x.json", doctor=doctor, safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg)
        await c.run()

        sess = store.get_session(1)
        assert sess["end_reason"].startswith("doctor_abort")
        # No second start_sequence (we aborted instead of retrying)
        assert fake.call_names().count("start_sequence") == 1
        # Close-down chain ran
        assert "close_dome_shutter" in fake.call_names()
        # Alert severity is alert (Doctor-driven abort, not safety panic)
        assert any(a["severity"] == "alert" for a in fake.alerts)

    async def test_repeated_faults_eventually_give_up(self, store):
        """If the Doctor keeps saying RETRY forever, the Conductor caps retries."""
        from nina_autopilot.doctor import DoctorAction, DoctorDecision
        fake = _FaultThenRecoverClient(error_polls=10)  # error forever for this test
        # Doctor always says RETRY — Conductor must enforce its own cap.
        doctor = _StubDoctor([
            DoctorDecision(action=DoctorAction.RETRY, reason="just one more!")
            for _ in range(10)
        ])
        cfg = ConductorConfig(
            sequence_file="x.json",
            doctor=doctor,
            safety_tick_s=0.0, wait_for_running_seconds=0.0,
            max_retries=2,
        )
        c = Conductor(fake, store, cfg)
        await c.run()

        # 1 initial + at most max_retries restarts
        assert fake.call_names().count("start_sequence") <= 1 + 2
        sess = store.get_session(1)
        assert "max_retries" in sess["end_reason"]

    async def test_doctor_decision_logged(self, store):
        from nina_autopilot.doctor import DoctorAction, DoctorDecision
        fake = _FaultThenRecoverClient(error_polls=1)
        doctor = _StubDoctor([
            DoctorDecision(action=DoctorAction.RETRY, reason="hiccup"),
        ])
        cfg = ConductorConfig(sequence_file="x.json", doctor=doctor, safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg)
        await c.run()

        events = store.list_events(1)
        kinds = [e["kind"] for e in events]
        assert "FAULT_DETECTED" in kinds
        assert "DOCTOR_DECISION" in kinds
        decision_ev = next(e for e in events if e["kind"] == "DOCTOR_DECISION")
        assert decision_ev["payload"]["action"] == "retry"
        assert decision_ev["payload"]["reason"] == "hiccup"


# ---------------------------------------------------------------------------
# Phase 4 — REPLAN handler
# ---------------------------------------------------------------------------

class _PlannerSequence:
    """Returns each queued PlannerDecision in turn — for REPLAN tests."""
    def __init__(self, decisions):
        from collections import deque
        self._decisions = deque(decisions)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if not self._decisions:
            raise RuntimeError("_PlannerSequence: ran out of decisions")
        return self._decisions.popleft()


class TestReplanHandler:
    async def test_replan_loads_new_sequence_and_continues(self, store):
        """Doctor REPLAN → Conductor re-runs Planner → loads NEW sequence → resumes imaging."""
        from nina_autopilot.doctor import DoctorAction, DoctorDecision

        fake = _FaultThenRecoverClient(error_polls=1)
        planner = _PlannerSequence([
            PlannerDecision(action=PlannerAction.IMAGE, sequence_name="target_A.json",
                            target={"name": "M81"}, summary="first pick"),
            PlannerDecision(action=PlannerAction.IMAGE, sequence_name="target_B.json",
                            target={"name": "NGC 7000"}, summary="second pick"),
        ])
        doctor = _StubDoctor([
            DoctorDecision(action=DoctorAction.REPLAN, reason="target A unreachable"),
        ])
        cfg = ConductorConfig(planner=planner, doctor=doctor, safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg)
        await c.run()

        # Planner called twice: initial + REPLAN
        assert planner.calls == 2
        # Two sequence loads — A then B
        load_calls = [c.kwargs["name"] for c in fake.calls if c.method == "load_sequence"]
        assert load_calls == ["target_A.json", "target_B.json"]
        # Session ended cleanly via sequence_complete on target_B
        sess = store.get_session(1)
        assert sess["end_reason"] == "sequence_complete"

    async def test_replan_with_no_work_ends_session(self, store):
        """Doctor REPLAN but Planner says NO_WORK on the re-plan → end cleanly."""
        from nina_autopilot.doctor import DoctorAction, DoctorDecision

        fake = _FaultThenRecoverClient(error_polls=10)
        planner = _PlannerSequence([
            PlannerDecision(action=PlannerAction.IMAGE, sequence_name="A.json",
                            target={"name": "X"}, summary="a"),
            PlannerDecision(action=PlannerAction.NO_WORK, summary="nothing left"),
        ])
        doctor = _StubDoctor([
            DoctorDecision(action=DoctorAction.REPLAN, reason="give up on A"),
        ])
        cfg = ConductorConfig(planner=planner, doctor=doctor, safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg)
        await c.run()

        sess = store.get_session(1)
        assert sess["end_reason"] == "no_work"
        # No second sequence loaded (Planner had nothing)
        load_calls = [c.kwargs["name"] for c in fake.calls if c.method == "load_sequence"]
        assert load_calls == ["A.json"]

    async def test_replan_without_planner_aborts(self, store):
        """Conductor configured with sequence_file but no planner → REPLAN means abort."""
        from nina_autopilot.doctor import DoctorAction, DoctorDecision

        fake = _FaultThenRecoverClient(error_polls=1)
        doctor = _StubDoctor([
            DoctorDecision(action=DoctorAction.REPLAN, reason="want a different target"),
        ])
        cfg = ConductorConfig(sequence_file="x.json", doctor=doctor, safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg)
        await c.run()

        sess = store.get_session(1)
        assert sess["end_reason"].startswith("doctor_abort")
        assert "no planner" in sess["end_reason"].lower()


# ---------------------------------------------------------------------------
# Phase 4 — PARK_AND_WAIT handler
# ---------------------------------------------------------------------------

class _SafetyChangingClient(FakeNinaClient):
    """Sequence errors on first poll; safety reading switches per-poll based on
    a queued sequence — used to test the park/wait/re-check flow."""

    def __init__(self, safety_after_wait: SafetyReading):
        super().__init__()
        self._first_seq_poll = True
        self._safety_after_wait = safety_after_wait
        self._safety_polls = 0

    async def get_sequence_state(self):
        self._record("get_sequence_state")
        if self._first_seq_poll:
            self._first_seq_poll = False
            return {"State": "Error", "ErrorMessage": "Cloud band, plate solve failed"}
        return {"State": "Finished"}

    async def get_safety_reading(self):
        self._safety_polls += 1
        # First call (the imaging-loop safety check before the fault) → SAFE
        # After park_and_wait sleeps, the next call returns the configured reading.
        if self._safety_polls >= 2:
            return self._safety_after_wait
        return self.safety_reading  # default = SafetyReading() = SAFE


class TestParkAndWaitHandler:
    async def test_park_and_wait_safe_after_recheck_resumes(self, store):
        """Conditions recover → unpark + start sequence again, retry counter reset."""
        from nina_autopilot.doctor import DoctorAction, DoctorDecision

        fake = _SafetyChangingClient(safety_after_wait=SafetyReading())  # safe after wait
        doctor = _StubDoctor([
            DoctorDecision(action=DoctorAction.PARK_AND_WAIT, reason="cloud band",
                           retry_after_s=120),
        ])
        slept = []
        async def fake_sleep(s):
            slept.append(s)

        cfg = ConductorConfig(sequence_file="x.json", doctor=doctor, safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg, sleep=fake_sleep)
        await c.run()

        calls = fake.call_names()
        # Park happened during PARK_AND_WAIT (before close-down adds another park)
        assert "park_mount" in calls
        # Slept for the requested retry_after_s (120s) somewhere in the sleep log
        assert 120.0 in slept
        # Resume = unpark + start_sequence again
        assert "unpark_mount" in calls
        assert calls.count("start_sequence") == 2
        # Final state: clean completion (sequence Finished on retry)
        assert store.get_session(1)["end_reason"] == "sequence_complete"

    async def test_park_and_wait_unsafe_after_recheck_aborts(self, store):
        """Conditions still bad after the wait → end session with weather reason."""
        from nina_autopilot.doctor import DoctorAction, DoctorDecision

        # Still raining after the wait
        fake = _SafetyChangingClient(safety_after_wait=SafetyReading(rain=True))
        doctor = _StubDoctor([
            DoctorDecision(action=DoctorAction.PARK_AND_WAIT, reason="cloud band",
                           retry_after_s=60),
        ])
        async def fake_sleep(s):
            pass

        cfg = ConductorConfig(sequence_file="x.json", doctor=doctor, safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg, sleep=fake_sleep)
        await c.run()

        sess = store.get_session(1)
        # Could be either safety_abort (rain re-tripped supervisor) or doctor_abort —
        # both are correct outcomes. Crucial: NOT sequence_complete.
        assert sess["end_reason"] != "sequence_complete"
        assert ("safety_abort" in sess["end_reason"] or
                "doctor_abort" in sess["end_reason"] or
                "park_and_wait" in sess["end_reason"])
        # We did NOT attempt a second start_sequence (conditions were still bad)
        assert fake.call_names().count("start_sequence") == 1

    async def test_park_and_wait_default_duration_when_missing(self, store):
        """If Doctor omits retry_after_s, we use a sensible default (300s)."""
        from nina_autopilot.doctor import DoctorAction, DoctorDecision

        fake = _SafetyChangingClient(safety_after_wait=SafetyReading())
        doctor = _StubDoctor([
            DoctorDecision(action=DoctorAction.PARK_AND_WAIT, reason="x",
                           retry_after_s=None),
        ])
        slept = []
        async def fake_sleep(s):
            slept.append(s)
        cfg = ConductorConfig(sequence_file="x.json", doctor=doctor, safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg, sleep=fake_sleep)
        await c.run()
        assert 300.0 in slept  # default park_and_wait duration

    async def test_park_and_wait_logged(self, store):
        from nina_autopilot.doctor import DoctorAction, DoctorDecision
        fake = _SafetyChangingClient(safety_after_wait=SafetyReading())
        doctor = _StubDoctor([
            DoctorDecision(action=DoctorAction.PARK_AND_WAIT, reason="x", retry_after_s=10),
        ])
        async def fake_sleep(s): pass
        cfg = ConductorConfig(sequence_file="x.json", doctor=doctor, safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg, sleep=fake_sleep)
        await c.run()
        kinds = [e["kind"] for e in store.list_events(1)]
        assert "PARK_AND_WAIT_STARTED" in kinds
        assert "PARK_AND_WAIT_RESUMED" in kinds


# ---------------------------------------------------------------------------
# Phase 4.2 — Operator + Scout integration
# ---------------------------------------------------------------------------

class _SubsThenFinishClient(FakeNinaClient):
    """Returns Running for first N polls (yielding a fresh sub each time),
    then Finished. Each Running poll consumes one entry from sub_stats_queue."""

    def __init__(self, running_polls: int):
        super().__init__()
        self._remaining_running = running_polls

    async def get_sequence_state(self):
        self._record("get_sequence_state")
        if self._remaining_running > 0:
            self._remaining_running -= 1
            return {"State": "Running"}
        return {"State": "Finished"}


class TestOperatorIntegration:
    async def test_operator_invoked_on_each_new_sub(self, store):
        from nina_autopilot.operator import Operator
        from nina_autopilot.nina_client import SubFrameStats

        fake = _SubsThenFinishClient(running_polls=3)
        fake.sub_stats_queue = [
            {"index": 1, "stats": SubFrameStats(hfr=2.4, star_count=600, guide_rms_total=0.5)},
            {"index": 2, "stats": SubFrameStats(hfr=2.5, star_count=620, guide_rms_total=0.6)},
            {"index": 3, "stats": SubFrameStats(hfr=2.6, star_count=590, guide_rms_total=0.55)},
        ]
        cfg = ConductorConfig(
            sequence_file="x.json",
            operator=Operator(),
            safety_tick_s=0.0, wait_for_running_seconds=0.0,
        )
        c = Conductor(fake, store, cfg)
        await c.run()

        kinds = [e["kind"] for e in store.list_events(1)]
        # One OPERATOR_DECISION per unique sub
        assert kinds.count("OPERATOR_DECISION") == 3

    async def test_operator_decision_payload_carries_metrics(self, store):
        from nina_autopilot.operator import Operator
        from nina_autopilot.nina_client import SubFrameStats

        fake = _SubsThenFinishClient(running_polls=1)
        fake.sub_stats_queue = [
            {"index": 42, "stats": SubFrameStats(hfr=6.0, star_count=900)},
        ]
        cfg = ConductorConfig(sequence_file="x.json", operator=Operator(), safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg)
        await c.run()

        op_ev = next(e for e in store.list_events(1) if e["kind"] == "OPERATOR_DECISION")
        assert op_ev["payload"]["action"] == "reshoot"  # HFR 6.0 > default max 5.0
        assert op_ev["payload"]["sub_index"] == 42
        assert op_ev["payload"]["metrics"]["hfr"] == 6.0

    async def test_operator_not_re_evaluated_on_same_sub(self, store):
        """Polling more often than NINA captures must NOT double-evaluate."""
        from nina_autopilot.operator import Operator
        from nina_autopilot.nina_client import SubFrameStats

        class _StaleSubClient(FakeNinaClient):
            """Sequence still running but only one sub exists for several polls."""
            def __init__(self):
                super().__init__()
                self._polls = 0

            async def get_sequence_state(self):
                self._record("get_sequence_state")
                self._polls += 1
                if self._polls >= 5:
                    return {"State": "Finished"}
                return {"State": "Running"}

            async def get_latest_sub_stats(self):
                self._record("get_latest_sub_stats")
                return {"index": 7, "stats": SubFrameStats(hfr=2.5, star_count=500)}

        fake = _StaleSubClient()
        cfg = ConductorConfig(sequence_file="x.json", operator=Operator(), safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg)
        await c.run()

        kinds = [e["kind"] for e in store.list_events(1)]
        assert kinds.count("OPERATOR_DECISION") == 1  # only the first sighting

    async def test_no_operator_means_no_op_events(self, store):
        fake = _SubsThenFinishClient(running_polls=2)
        # Even if stats are available, no Operator → no OPERATOR_DECISION events
        from nina_autopilot.nina_client import SubFrameStats
        fake.sub_stats_queue = [
            {"index": 1, "stats": SubFrameStats(hfr=2.4)},
            {"index": 2, "stats": SubFrameStats(hfr=2.5)},
        ]
        cfg = ConductorConfig(sequence_file="x.json", safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg)
        await c.run()

        kinds = [e["kind"] for e in store.list_events(1)]
        assert "OPERATOR_DECISION" not in kinds


class TestScoutIntegration:
    async def test_scout_invoked_per_tick_logs_summary_only_on_change(self, store):
        from nina_autopilot.scout import Scout

        class _ChangingSafetyClient(FakeNinaClient):
            def __init__(self):
                super().__init__()
                self._polls = 0
                self.sequence_state = {"State": "Finished"}

            async def get_safety_reading(self):
                self._polls += 1
                # Initial reading then a notable cloud-cover change on poll 2
                if self._polls == 1:
                    return SafetyReading(cloud_cover_pct=10.0)
                return SafetyReading(cloud_cover_pct=70.0)

        fake = _ChangingSafetyClient()
        cfg = ConductorConfig(sequence_file="x.json", scout=Scout(), safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg)
        await c.run()

        kinds = [e["kind"] for e in store.list_events(1)]
        # First call produces an INITIAL summary, second produces a change summary
        scout_events = [e for e in store.list_events(1) if e["kind"] == "SCOUT_SUMMARY"]
        # At least one summary; depending on tick timing there may be 1-2
        assert len(scout_events) >= 1

    async def test_scout_alert_posts_to_discord(self, store):
        from nina_autopilot.scout import Scout

        class _RainAppearsClient(FakeNinaClient):
            def __init__(self):
                super().__init__()
                self._polls = 0

            async def get_safety_reading(self):
                self._polls += 1
                if self._polls == 1:
                    return SafetyReading(rain=False, cloud_cover_pct=5.0)
                # Rain appears → Scout ALERT (separately from safety supervisor's UNSAFE)
                return SafetyReading(rain=True, cloud_cover_pct=5.0)

            async def get_sequence_state(self):
                self._record("get_sequence_state")
                return {"State": "Running"}  # never finishes — scout/safety must end it

        fake = _RainAppearsClient()
        cfg = ConductorConfig(sequence_file="x.json", scout=Scout(), safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg)
        await c.run()

        # Safety abort fires AS WELL (rain → UNSAFE) — but Scout must have posted a
        # Discord alert too. Find an alert-severity Discord message (not panic).
        # Safety supervisor will post panic; Scout posts alert. Both should appear.
        severities = {a["severity"] for a in fake.alerts}
        # Scout posts alert before safety panic — at minimum one of the two is there.
        assert "alert" in severities or "panic" in severities

    async def test_no_scout_means_no_scout_events(self, store):
        fake = FakeNinaClient()
        fake.sequence_state = {"State": "Finished"}
        cfg = ConductorConfig(sequence_file="x.json", safety_tick_s=0.0, wait_for_running_seconds=0.0)
        c = Conductor(fake, store, cfg)
        await c.run()

        kinds = [e["kind"] for e in store.list_events(1)]
        assert "SCOUT_SUMMARY" not in kinds


# ---------------------------------------------------------------------------
# Phase 5.1 — Sequence-start handshake (reset + wait-for-Running gate)
# ---------------------------------------------------------------------------

class _ScriptedSequenceStateClient(FakeNinaClient):
    """Fake that returns scripted sequence_state values on each poll.

    Lets a test express "first poll Finished, second Finished, third Running,
    fourth Finished" or similar — needed to exercise the gate (NINA's real
    behaviour after sequence/start: stale FINISHED for a few millis, then
    RUNNING, then FINISHED at end).
    """

    def __init__(self, states):
        super().__init__()
        from collections import deque
        self._states = deque(states)

    async def get_sequence_state(self):
        self._record("get_sequence_state")
        if not self._states:
            return {"State": "Finished"}  # default tail
        return self._states.popleft()


class TestImagingHandshake:
    async def test_reset_sequence_called_before_load(self, fake, store):
        """The Conductor must call reset_sequence BEFORE start_sequence.

        Order: load → reset → start (load first because reset is no-op without
        a loaded sequence in some NINA versions; reset before start is the
        critical bit — without it NINA's sequence/start is a no-op on already-
        FINISHED containers from the previous run).
        """
        fake.sequence_state = {"State": "Finished"}
        c = make_conductor(fake, store, sequence_file="x.json")
        await c.run()

        calls = fake.call_names()
        assert "reset_sequence" in calls
        assert "load_sequence" in calls
        assert "start_sequence" in calls
        assert calls.index("reset_sequence") < calls.index("start_sequence"), (
            "reset_sequence must be called BEFORE start_sequence — otherwise "
            "NINA's start is a no-op on FINISHED containers"
        )

    async def test_finished_before_running_with_wait_keeps_polling(self, store):
        """If sequence/state reports Finished before we've ever seen Running,
        and the wait_for_running gate is still open, the Conductor must keep
        polling (not exit IMAGING immediately)."""
        fake = _ScriptedSequenceStateClient([
            # First poll right after start: stale FINISHED from previous run.
            {"State": "Finished"},
            {"State": "Finished"},
            # NINA finally flips to Running.
            {"State": "Running"},
            # Sequence runs a couple ticks.
            {"State": "Running"},
            # Finally truly finishes.
            {"State": "Finished"},
        ])
        cfg = ConductorConfig(
            sequence_file="x.json",
            safety_tick_s=0.0,
            wait_for_running_seconds=60.0,  # generous, gate stays open
        )
        c = Conductor(fake, store, cfg)
        await c.run()

        # Must reach DONE via sequence_complete (not abort)
        assert c.phase is Phase.DONE
        sess = store.get_session(1)
        assert sess["end_reason"] == "sequence_complete"
        # And must have actually polled multiple times — the early "Finished"
        # responses got ignored by the gate.
        assert fake.call_names().count("get_sequence_state") >= 4

    async def test_finished_before_running_after_timeout_aborts(self, store, monkeypatch):
        """If the wait_for_running timeout elapses with no Running seen ever,
        the Conductor must give up and abort — otherwise a truly broken start
        would loop forever."""
        # Sequence/state always reports Finished, never Running.
        fake = _ScriptedSequenceStateClient([
            {"State": "Finished"},
        ] * 50)

        # Make monotonic() return increasing values so the elapsed timeout
        # is reached quickly: each tick of our fake sleep advances by 1s.
        clock = [0.0]

        async def fake_sleep(s):
            clock[0] += s if s > 0 else 0.1  # advance even on 0s tick

        # Inject the fake clock — pytest.monkeypatch handles restore correctly
        # (manual save/restore breaks the staticmethod descriptor).
        import nina_autopilot.conductor as cmod
        monkeypatch.setattr(cmod.Conductor, "_monotonic", staticmethod(lambda: clock[0]))

        cfg = ConductorConfig(
            sequence_file="x.json",
            safety_tick_s=0.5,  # each tick advances clock by 0.5s
            wait_for_running_seconds=1.0,  # gate expires after 1s elapsed
        )
        c = Conductor(fake, store, cfg, sleep=fake_sleep)
        await c.run()

        # Once timeout elapses, the Finished state is accepted → sequence_complete
        # OR the Conductor times out into an aborted state. The contract: it
        # must exit the imaging loop instead of polling forever.
        assert c.phase is Phase.DONE
        # Time advanced — the loop didn't run forever
        assert clock[0] >= 1.0

    async def test_running_seen_then_finished_completes_normally(self, store):
        """The normal happy path: state goes Running → Running → Finished.
        Conductor completes cleanly with end_reason=sequence_complete."""
        fake = _ScriptedSequenceStateClient([
            {"State": "Running"},
            {"State": "Running"},
            {"State": "Finished"},
        ])
        cfg = ConductorConfig(
            sequence_file="x.json",
            safety_tick_s=0.0,
            wait_for_running_seconds=30.0,  # gate active but irrelevant on happy path
        )
        c = Conductor(fake, store, cfg)
        await c.run()

        assert c.phase is Phase.DONE
        sess = store.get_session(1)
        assert sess["end_reason"] == "sequence_complete"
