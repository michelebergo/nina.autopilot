"""Conductor — Phase 2 state machine + close-down chain.

State flow:

    BOOT → STARTING (load+start sequence) → IMAGING ─┬─ sequence done → CLOSING
                                                      ├─ unsafe       → ABORTING
                                                      └─ manual stop  → ABORTING
                                            ABORTING → CLOSING → DONE
                                            CLOSING  → DONE

The IMAGING loop polls the safety supervisor and the sequencer state at
`safety_tick_s` cadence. UNSAFE preempts everything — close-down is then
best-effort (one step's failure does not stop the others).

No LLM in Phase 2. Agents arrive in Phase 3.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from .doctor import DoctorAction, FaultContext
from .llm import BudgetExceeded
from .nina_client import NinaClient
from .planner import PlannerAction, PlannerDecision
from .safety import SafetyDecision, SafetyLevel, SafetyThresholds, evaluate
from .state import SessionStore


logger = logging.getLogger(__name__)


class Phase(str, Enum):
    BOOT = "BOOT"
    STARTING = "STARTING"
    IMAGING = "IMAGING"
    ABORTING = "ABORTING"
    CLOSING = "CLOSING"
    DONE = "DONE"
    FAULT = "FAULT"


_SEQUENCE_DONE_STATES = {"Idle", "Finished", "Completed", "Stopped"}
_SEQUENCE_ERROR_STATE = "Error"


PlannerCallable = Callable[[], Awaitable[PlannerDecision]]  # sync also accepted


@dataclass
class ConductorConfig:
    sequence_file: Optional[str] = None
    safety_tick_s: float = 5.0
    thresholds: SafetyThresholds = field(default_factory=SafetyThresholds)
    planner: Optional[Callable[[], Any]] = None  # sync or async; runs at session start
    doctor: Optional[Any] = None  # any object with `async def diagnose(ctx) -> DoctorDecision`
    max_retries: int = 2  # cap on Doctor RETRY actions before forced abort

    def __post_init__(self):
        if self.sequence_file is None and self.planner is None:
            raise ValueError(
                "ConductorConfig requires either sequence_file (Phase 2 hand-written) "
                "or planner (Phase 3 TS-driven)."
            )


class Conductor:
    def __init__(
        self,
        client: NinaClient,
        store: SessionStore,
        config: ConductorConfig,
        *,
        sleep: Optional[Callable[[float], Awaitable[None]]] = None,
    ) -> None:
        self._client = client
        self._store = store
        self._config = config
        self._sleep = sleep or asyncio.sleep
        self._phase: Phase = Phase.BOOT
        self._end_reason: str = "unknown"
        self._stop_requested = False
        self._session_id: Optional[int] = None
        self._retry_count = 0

    @property
    def phase(self) -> Phase:
        return self._phase

    async def request_stop(self) -> None:
        """External e-stop — picked up on next IMAGING tick."""
        self._stop_requested = True

    async def run(self) -> None:
        self._session_id = self._store.start_session(self._config.sequence_file)
        try:
            await self._phase_starting()
            if self._phase is Phase.IMAGING:
                await self._phase_imaging()
            if self._phase is Phase.ABORTING:
                await self._phase_aborting()
            if self._phase is Phase.CLOSING:
                await self._phase_closing()
        except Exception as e:
            logger.exception("Conductor run() crashed")
            self._end_reason = f"fault: {e}"
            self._transition(Phase.FAULT)
        finally:
            if self._session_id is not None:
                self._store.end_session(self._session_id, self._end_reason)

    # ---- phase handlers ----

    async def _phase_starting(self) -> None:
        self._transition(Phase.STARTING)
        sequence_name = await self._resolve_sequence()
        if sequence_name is None:
            # Planner said NO_WORK — skip imaging, close-down idempotently.
            self._end_reason = "no_work"
            self._transition(Phase.CLOSING)
            return
        await self._client.load_sequence(sequence_name)
        await self._client.start_sequence()
        self._transition(Phase.IMAGING)

    async def _resolve_sequence(self) -> Optional[str]:
        """Phase 3: invoke Planner if configured. Returns the sequence name to
        load, or None for NO_WORK. Falls back to ConductorConfig.sequence_file."""
        if self._config.planner is not None:
            raw = self._config.planner()
            decision = await raw if asyncio.iscoroutine(raw) else raw
            if not isinstance(decision, PlannerDecision):
                raise TypeError(
                    f"planner returned {type(decision).__name__}, expected PlannerDecision"
                )
            if self._session_id is not None:
                self._store.record_event(
                    self._session_id,
                    kind="PLANNER_DECISION",
                    payload={
                        "action": decision.action.value,
                        "sequence_name": decision.sequence_name,
                        "summary": decision.summary,
                        "target": decision.target,
                        "project": decision.project,
                    },
                )
            if decision.action is PlannerAction.NO_WORK:
                return None
            return decision.sequence_name
        return self._config.sequence_file

    async def _phase_imaging(self) -> None:
        while True:
            # 1) Safety first — pre-empts everything else.
            reading = await self._client.get_safety_reading()
            decision = evaluate(reading, self._config.thresholds)
            if decision.is_unsafe:
                self._record_safety(decision)
                self._end_reason = f"safety_abort: {'; '.join(decision.reasons)}"
                self._transition(Phase.ABORTING)
                return
            if decision.level is SafetyLevel.WARN:
                self._record_safety(decision)  # warn-level logged but not aborting

            # 2) Manual stop check
            if self._stop_requested:
                self._end_reason = "manual_stop"
                self._transition(Phase.ABORTING)
                return

            # 3) Sequence state check — completion OR fault
            seq_state = await self._client.get_sequence_state()
            state_name = seq_state.get("State")
            if state_name in _SEQUENCE_DONE_STATES:
                self._end_reason = "sequence_complete"
                self._transition(Phase.CLOSING)
                return
            if state_name == _SEQUENCE_ERROR_STATE:
                continue_imaging = await self._handle_fault(seq_state)
                if not continue_imaging:
                    return

            await self._sleep(self._config.safety_tick_s)

    async def _phase_aborting(self) -> None:
        # Stop the sequence first (best-effort)
        await self._safe_call("stop_sequence", self._client.stop_sequence())
        self._transition(Phase.CLOSING)

    async def _handle_fault(self, seq_state: dict[str, Any]) -> bool:
        """Invoke Doctor on a sequencer error. Returns True to continue imaging
        (retry started), False if the session is ending (transition queued)."""
        fault_msg = seq_state.get("ErrorMessage") or "Sequence reported error state"

        # No doctor configured → fail safe = abort immediately
        if self._config.doctor is None:
            self._end_reason = f"sequence_error: {fault_msg}"
            self._record_event("FAULT_DETECTED", {
                "fault_message": fault_msg, "doctor_configured": False,
            })
            self._transition(Phase.ABORTING)
            return False

        self._retry_count += 1
        self._record_event("FAULT_DETECTED", {
            "fault_type": "sequence_error",
            "fault_message": fault_msg,
            "attempt": self._retry_count,
        })

        if self._retry_count > self._config.max_retries:
            self._end_reason = (
                f"max_retries ({self._config.max_retries}) exceeded for sequence_error"
            )
            self._transition(Phase.ABORTING)
            return False

        ctx = FaultContext(
            fault_type="sequence_error",
            fault_message=fault_msg,
            consecutive_count=self._retry_count,
        )
        try:
            decision = await self._config.doctor.diagnose(ctx)
        except BudgetExceeded as e:
            self._record_event("BUDGET_HALT", {"message": str(e)})
            self._end_reason = f"doctor_abort: nightly budget exhausted — {e}"
            self._transition(Phase.ABORTING)
            return False
        self._record_event("DOCTOR_DECISION", {
            "action": decision.action.value,
            "reason": decision.reason,
            "retry_after_s": decision.retry_after_s,
        })

        if decision.action is DoctorAction.RETRY:
            # Restart the sequence — NINA's sequencer will re-attempt from the top.
            await self._safe_call("stop_sequence", self._client.stop_sequence())
            await self._safe_call("start_sequence", self._client.start_sequence())
            return True  # continue imaging loop

        if decision.action is DoctorAction.REPLAN:
            return await self._handle_replan(decision)

        if decision.action is DoctorAction.PARK_AND_WAIT:
            return await self._handle_park_and_wait(decision)

        # ABORT (or any unknown action — fail safe)
        self._end_reason = f"doctor_abort: {decision.reason}"
        self._transition(Phase.ABORTING)
        return False

    async def _handle_park_and_wait(self, decision) -> bool:
        """Doctor wants the rig idle for a while (typically transient weather).

        Flow: stop sequence → park mount → sleep(retry_after_s) → re-check safety.
        If safe → unpark + start sequence (resume), reset retry counter.
        If unsafe → end session with a park_and_wait_unsafe reason.
        """
        wait_s = decision.retry_after_s if decision.retry_after_s is not None else 300.0
        self._record_event("PARK_AND_WAIT_STARTED", {
            "wait_s": wait_s,
            "reason": decision.reason,
        })
        await self._safe_call("stop_sequence", self._client.stop_sequence())
        await self._safe_call("park_mount", self._client.park_mount())
        await self._sleep(wait_s)

        # Re-check safety after the wait
        reading = await self._client.get_safety_reading()
        post_decision = evaluate(reading, self._config.thresholds)
        if post_decision.is_unsafe:
            self._record_safety(post_decision)
            self._end_reason = (
                f"park_and_wait_unsafe: {'; '.join(post_decision.reasons)}"
            )
            self._transition(Phase.ABORTING)
            return False

        # Resume — reset retry counter so the resumed run gets a fresh fault budget.
        self._retry_count = 0
        await self._safe_call("unpark_mount", self._client.unpark_mount())
        await self._safe_call("start_sequence", self._client.start_sequence())
        self._record_event("PARK_AND_WAIT_RESUMED", {
            "post_wait_level": post_decision.level.value,
        })
        return True

    async def _handle_replan(self, decision) -> bool:
        """Doctor wants a different target. Re-run Planner, switch sequences."""
        if self._config.planner is None:
            self._end_reason = (
                f"doctor_abort: REPLAN requested but no planner configured ({decision.reason})"
            )
            self._transition(Phase.ABORTING)
            return False

        # Stop the failing sequence before re-planning
        await self._safe_call("stop_sequence", self._client.stop_sequence())

        new_sequence = await self._resolve_sequence()
        if new_sequence is None:
            # Planner says NO_WORK on the replan — end cleanly
            self._end_reason = "no_work"
            self._transition(Phase.CLOSING)
            return False

        # Fresh sequence — reset the retry counter so the new target gets its own budget
        self._retry_count = 0
        await self._safe_call("load_sequence", self._client.load_sequence(new_sequence))
        await self._safe_call("start_sequence", self._client.start_sequence())
        self._record_event("REPLAN_LOADED", {"sequence_name": new_sequence})
        return True  # continue imaging loop

    def _record_event(self, kind: str, payload: dict[str, Any]) -> None:
        if self._session_id is not None:
            self._store.record_event(self._session_id, kind=kind, payload=payload)

    async def _phase_closing(self) -> None:
        await self._safe_call("close_dome_shutter", self._client.close_dome_shutter())
        await self._safe_call("park_mount", self._client.park_mount())
        await self._safe_call("stop_cooling", self._client.stop_cooling())

        # Alert severity depends on how we got here.
        if self._end_reason == "sequence_complete":
            await self._safe_call(
                "alert",
                self._client.alert(severity="info", message="Session complete — sequence finished cleanly."),
            )
        elif self._end_reason == "manual_stop":
            await self._safe_call(
                "alert",
                self._client.alert(severity="alert", message="Session ended by manual stop."),
            )
        elif self._end_reason.startswith("safety_abort"):
            reasons = self._end_reason.removeprefix("safety_abort:").strip()
            await self._safe_call(
                "alert",
                self._client.alert(severity="panic", message=f"Safety abort: {reasons}"),
            )
        elif self._end_reason == "no_work":
            await self._safe_call(
                "alert",
                self._client.alert(severity="info", message="No actionable targets — session ended cleanly."),
            )
        elif self._end_reason.startswith("doctor_abort") or self._end_reason.startswith("max_retries"):
            await self._safe_call(
                "alert",
                self._client.alert(severity="alert", message=f"Session ended: {self._end_reason}"),
            )
        else:
            await self._safe_call(
                "alert",
                self._client.alert(severity="alert", message=f"Session ended: {self._end_reason}"),
            )

        self._transition(Phase.DONE)

    # ---- helpers ----

    def _transition(self, new_phase: Phase) -> None:
        if new_phase is self._phase:
            return
        old = self._phase
        self._phase = new_phase
        logger.info("phase: %s → %s", old.value, new_phase.value)
        if self._session_id is not None:
            self._store.set_phase(self._session_id, new_phase.value)
            self._store.record_event(
                self._session_id,
                kind="PHASE_CHANGE",
                payload={"from": old.value, "to": new_phase.value},
            )

    def _record_safety(self, decision: SafetyDecision) -> None:
        if self._session_id is not None:
            self._store.record_event(
                self._session_id,
                kind="SAFETY_" + decision.level.value.upper(),
                payload={
                    "reasons": decision.reasons,
                    "triggered": decision.triggered_signals,
                },
            )

    async def _safe_call(self, name: str, coro: Awaitable) -> None:
        """Best-effort: log + record but do not raise so the chain continues."""
        try:
            await coro
        except Exception as e:
            logger.warning("close-down step '%s' failed: %s", name, e)
            if self._session_id is not None:
                self._store.record_event(
                    self._session_id,
                    kind="CLOSEDOWN_STEP_FAILED",
                    payload={"step": name, "error": str(e)},
                )
