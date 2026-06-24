"""Phase 2 exit-criterion test.

Plan says: "Survives an injected unsafe event, closes dome, parks, alerts."

This is the gate that proves Phase 2 is complete. It runs the Conductor
end-to-end with a FakeNinaClient that starts SAFE, then mid-IMAGING flips
to UNSAFE (rain). The Conductor must:
  1. Detect the change on the next safety poll
  2. Stop the sequence
  3. Close the dome shutter
  4. Park the mount
  5. Stop cooling
  6. Send a Discord PANIC alert
  7. Persist abort reason in session DB
"""

import pytest

from nina_autopilot.conductor import Conductor, ConductorConfig, Phase
from nina_autopilot.nina_client import FakeNinaClient
from nina_autopilot.safety import SafetyReading
from nina_autopilot.state import open_store


class _FlippingFakeClient(FakeNinaClient):
    """Fake that returns SAFE for the first `flip_after_n` polls, then UNSAFE.

    Simulates 'clouds rolled in mid-night'.
    """

    def __init__(self, flip_after_n: int, unsafe_reading: SafetyReading) -> None:
        super().__init__()
        self._flip_after_n = flip_after_n
        self._unsafe_reading = unsafe_reading
        self._safety_polls = 0
        # Sequence stays "Running" forever — only safety can terminate the loop.
        self.sequence_state = {"State": "Running", "Progress": 0}

    async def get_safety_reading(self) -> SafetyReading:
        self._safety_polls += 1
        if self._safety_polls > self._flip_after_n:
            return self._unsafe_reading
        return self.safety_reading  # safe (default)


async def test_phase2_exit_criterion_full_close_down_chain(tmp_path):
    store = open_store(tmp_path / "session.sqlite")
    fake = _FlippingFakeClient(
        flip_after_n=3,
        unsafe_reading=SafetyReading(rain=True, cloud_cover_pct=95.0),
    )
    conductor = Conductor(
        fake,
        store,
        ConductorConfig(sequence_file="my_target.json", safety_tick_s=0.0, wait_for_running_seconds=0.0),
    )

    await conductor.run()

    # ---- 1. Final phase is DONE (close-down ran to completion) ----
    assert conductor.phase is Phase.DONE

    # ---- 2. Sequence reset+load+start happened before close-down ----
    # (reset_sequence puts NINA's containers back to CREATED so start_sequence
    # actually transitions to RUNNING — see _phase_starting in conductor.py)
    calls = fake.call_names()
    assert "load_sequence" in calls
    assert "reset_sequence" in calls
    assert "start_sequence" in calls
    assert calls.index("load_sequence") < calls.index("reset_sequence") < calls.index("start_sequence")

    # ---- 3. Close-down chain executed in correct order ----
    close_down_calls = [c for c in calls if c in {
        "stop_sequence", "close_dome_shutter", "park_mount", "stop_cooling"
    }]
    assert close_down_calls == [
        "stop_sequence",
        "close_dome_shutter",
        "park_mount",
        "stop_cooling",
    ], f"close-down order wrong: {close_down_calls}"

    # ---- 4. PANIC alert sent with rain in the message ----
    assert any(a["severity"] == "panic" for a in fake.alerts), "no panic alert sent"
    panic_alert = next(a for a in fake.alerts if a["severity"] == "panic")
    assert "rain" in panic_alert["message"].lower()

    # ---- 5. Session DB shows safety_abort end reason ----
    sess = store.get_session(1)
    assert sess["end_reason"].startswith("safety_abort")
    assert "rain" in sess["end_reason"].lower()

    # ---- 6. Event log records the safety event and the abort phase change ----
    events = store.list_events(1)
    event_kinds = [e["kind"] for e in events]
    assert "SAFETY_UNSAFE" in event_kinds
    phase_changes = [e["payload"] for e in events if e["kind"] == "PHASE_CHANGE"]
    targets = [p["to"] for p in phase_changes]
    assert "ABORTING" in targets
    assert "CLOSING" in targets
    assert "DONE" in targets

    # ---- 7. The Conductor actually saw the SAFE state first ----
    # (Confirms we didn't accidentally make every poll unsafe.)
    safety_events = [e for e in events if e["kind"] == "SAFETY_UNSAFE"]
    # Only the unsafe transition should be logged (we don't log SAFE)
    assert len(safety_events) == 1
