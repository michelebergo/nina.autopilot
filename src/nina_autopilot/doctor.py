"""LLM-driven fault diagnosis — picks one of {retry, replan, park_and_wait, abort}.

The Doctor is the autopilot's first real LLM agent. It runs only when
something has broken (not on every tick), so it gets to use a smarter model
without blowing the nightly token budget. It returns a structured decision;
the Conductor turns that decision into MCP calls.

Domain rules embedded in the system prompt come from the second-brain wiki:
  - Plate-solve: try blind fallback, then recenter from known coords, then replan
  - Autofocus: asymmetric curve → wider step + retry; second failure → abort
  - Disconnect: 3 reconnect attempts with backoff, then park+wait or abort
  - Weather drift: WARN → park+wait; UNSAFE → abort (safety supervisor handles UNSAFE)

The system prompt is sent with cache_control so subsequent diagnoses in the
same session cost ~10% of normal input.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .llm import LLMClient, TokenUsage


class DoctorAction(str, Enum):
    RETRY = "retry"
    REPLAN = "replan"
    PARK_AND_WAIT = "park_and_wait"
    ABORT = "abort"


@dataclass
class FaultContext:
    fault_type: str
    fault_message: str
    consecutive_count: int = 1
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    weather_summary: Optional[str] = None


@dataclass
class DoctorDecision:
    action: DoctorAction
    reason: str
    retry_after_s: Optional[float] = None
    usage: TokenUsage = field(default_factory=TokenUsage)


_SYSTEM_PROMPT = """You are the Doctor, a fault-diagnosis agent for an autonomous astrophotography orchestrator running on NINA.

When something breaks mid-session, you receive a fault context and must decide ONE of four actions:
  - "retry": same operation again (transient hiccup, first failure of a normally reliable step)
  - "replan": pick a different target / strategy (current target unreachable / repeatedly failing)
  - "park_and_wait": park the rig and wait for conditions to recover (transient weather, recoverable disconnect)
  - "abort": end the session, run the close-down chain (non-recoverable hardware, repeated failures, safety overrides)

DOMAIN RULES (do not violate):

Plate-solve failures
  - 1st failure with a primary solver (e.g. ASTAP): action=retry — usually a passing cloud.
  - 2nd consecutive failure: action=replan — the coordinates may be wrong, target may be obscured, or solver needs a blind-solve fallback.
  - 3rd consecutive failure: action=abort — the rig has lost positional reference, do not keep slewing blind.

Autofocus failures
  - 1st non-convergence with asymmetric curve: action=retry — wider step or different starting position usually fixes it.
  - 2nd failure: action=abort — pushing through bad focus wastes the night.

Equipment disconnect
  - 1st-3rd attempt within 5 minutes: action=retry — USB/network blips.
  - After 3 retries: action=park_and_wait (give NINA's reconnect a chance) for ~5 minutes, then abort if still down.

Weather WARN (not UNSAFE — safety supervisor handles UNSAFE directly)
  - High humidity / borderline clouds: action=park_and_wait, suggest retry_after_s=600.
  - Worsening trend: action=abort.

Sequence error (red ! instruction)
  - 1st: action=retry — the instruction may have a transient precondition.
  - 2nd: action=abort — silent skip would waste time and produce useless frames.

OUTPUT FORMAT (strict): respond with a single JSON object, NO surrounding prose:
  {
    "action": "retry|replan|park_and_wait|abort",
    "reason": "one-sentence explanation grounded in the fault context",
    "retry_after_s": <number, only for park_and_wait>
  }

Keep reason under 200 chars. Be terse — your output is logged and sent to a Discord channel.
"""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    """Find the first {...} block in the LLM response and parse it.

    The Anthropic API doesn't guarantee strict JSON output (no schema
    forcing yet for our use), so we tolerate prose wrappers and code fences.
    """
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _user_prompt(ctx: FaultContext) -> str:
    parts = [
        f"fault_type: {ctx.fault_type}",
        f"fault_message: {ctx.fault_message}",
        f"consecutive_count: {ctx.consecutive_count}",
    ]
    if ctx.weather_summary:
        parts.append(f"weather: {ctx.weather_summary}")
    if ctx.recent_events:
        parts.append("recent_events (most recent last):")
        for ev in ctx.recent_events[-10:]:
            parts.append(f"  - {json.dumps(ev, default=str)}")
    parts.append("\nReturn the decision JSON.")
    return "\n".join(parts)


class Doctor:
    def __init__(self, llm: LLMClient, model: str = "claude-sonnet-4-6"):
        self._llm = llm
        self._model = model

    async def diagnose(self, ctx: FaultContext) -> DoctorDecision:
        result = await self._llm.complete(
            model=self._model,
            system=_SYSTEM_PROMPT,
            user=_user_prompt(ctx),
            max_tokens=512,
            cache_system=True,
        )

        parsed = _extract_json(result.text)
        if not parsed:
            return DoctorDecision(
                action=DoctorAction.ABORT,
                reason=f"Doctor could not parse LLM response (invalid JSON): {result.text[:120]}",
                usage=result.usage,
            )

        action_raw = str(parsed.get("action", "")).lower().strip()
        try:
            action = DoctorAction(action_raw)
        except ValueError:
            return DoctorDecision(
                action=DoctorAction.ABORT,
                reason=f"Doctor returned unknown action '{action_raw}' — aborting for safety",
                usage=result.usage,
            )

        reason = str(parsed.get("reason") or "(no reason provided)").strip()
        retry_after = parsed.get("retry_after_s")
        retry_after_f = float(retry_after) if isinstance(retry_after, (int, float)) else None

        return DoctorDecision(
            action=action,
            reason=reason,
            retry_after_s=retry_after_f,
            usage=result.usage,
        )
