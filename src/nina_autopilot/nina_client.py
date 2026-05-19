"""Abstraction over NINA + autopilot-extension tools.

The Conductor depends on the NinaClient Protocol, not a concrete class. This
lets us swap implementations: HttpNinaClient (real NINA via Advanced API) for
production, FakeNinaClient for tests.

Only methods the Conductor actually calls live here — YAGNI on the full 176
tool surface. New needs add new methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from .operator import SubFrameStats
from .safety import SafetyReading


@runtime_checkable
class NinaClient(Protocol):
    """The minimal control surface the Phase 2 Conductor needs."""

    async def get_safety_reading(self) -> SafetyReading: ...
    async def get_sequence_state(self) -> dict[str, Any]: ...
    async def load_sequence(self, name: str) -> dict[str, Any]: ...
    async def start_sequence(self) -> dict[str, Any]: ...
    async def stop_sequence(self) -> dict[str, Any]: ...
    async def close_dome_shutter(self) -> dict[str, Any]: ...
    async def park_mount(self) -> dict[str, Any]: ...
    async def unpark_mount(self) -> dict[str, Any]: ...
    async def stop_cooling(self) -> dict[str, Any]: ...
    async def alert(
        self,
        severity: str,
        message: str,
        image_path: Optional[str] = None,
    ) -> dict[str, Any]: ...
    async def get_latest_sub_stats(self) -> Optional[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Test double — records every call for assertions, returns canned responses.
# ---------------------------------------------------------------------------

@dataclass
class FakeCall:
    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeNinaClient:
    """In-memory NinaClient for tests. Set attributes to control responses."""

    def __init__(self) -> None:
        self.calls: list[FakeCall] = []
        # Canned responses — tests mutate these before invoking the Conductor.
        self.safety_reading: SafetyReading = SafetyReading()
        self.sequence_state: dict[str, Any] = {"State": "Idle", "Progress": 0}
        self.alerts: list[dict[str, Any]] = []
        # Sub-stats queue for Operator integration tests. Each get_latest_sub_stats
        # call pops the first entry (or returns None if empty).
        self.sub_stats_queue: list[dict[str, Any]] = []
        # Method names that should raise instead of returning normally.
        self.fail_on: set[str] = set()

    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append(FakeCall(method, kwargs))
        if method in self.fail_on:
            raise RuntimeError(f"FakeNinaClient: simulated failure in {method}")

    def call_names(self) -> list[str]:
        return [c.method for c in self.calls]

    async def get_safety_reading(self) -> SafetyReading:
        self._record("get_safety_reading")
        return self.safety_reading

    async def get_sequence_state(self) -> dict[str, Any]:
        self._record("get_sequence_state")
        return dict(self.sequence_state)

    async def load_sequence(self, name: str) -> dict[str, Any]:
        self._record("load_sequence", name=name)
        return {"Success": True, "Message": f"loaded {name}"}

    async def start_sequence(self) -> dict[str, Any]:
        self._record("start_sequence")
        # NOTE: deliberately NO state mutation — tests own sequence_state so they
        # can pre-set "Finished" to exit the IMAGING loop on the first poll.
        return {"Success": True}

    async def stop_sequence(self) -> dict[str, Any]:
        self._record("stop_sequence")
        return {"Success": True}

    async def close_dome_shutter(self) -> dict[str, Any]:
        self._record("close_dome_shutter")
        return {"Success": True}

    async def park_mount(self) -> dict[str, Any]:
        self._record("park_mount")
        return {"Success": True}

    async def unpark_mount(self) -> dict[str, Any]:
        self._record("unpark_mount")
        return {"Success": True}

    async def stop_cooling(self) -> dict[str, Any]:
        self._record("stop_cooling")
        return {"Success": True}

    async def alert(
        self,
        severity: str,
        message: str,
        image_path: Optional[str] = None,
    ) -> dict[str, Any]:
        self._record("alert", severity=severity, message=message, image_path=image_path)
        self.alerts.append({"severity": severity, "message": message, "image_path": image_path})
        return {"Success": True}

    async def get_latest_sub_stats(self) -> Optional[dict[str, Any]]:
        self._record("get_latest_sub_stats")
        if self.sub_stats_queue:
            return self.sub_stats_queue.pop(0)
        return None
