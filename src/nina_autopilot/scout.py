"""World-state delta detector — Phase 4.1.

Scout compares the current SafetyReading to the previous one and produces a
compact human-readable summary the dashboard can display and Discord can
post. Scout is OBSERVATION only — it doesn't decide actions; the safety
supervisor (hard rules) and Doctor (LLM judgment) own decisions.

Phase 4.1 is rule-based; an LLM rewriter hook will be added later for nicer
phrasing on noteworthy deltas. Steady state cost: zero tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .safety import SafetyReading


class ScoutSeverity(str, Enum):
    OK = "ok"
    WARN = "warn"
    ALERT = "alert"


# Per-field noise floor: deltas below this are ignored.
_NOISE_FLOOR: dict[str, float] = {
    "cloud_cover_pct": 5.0,
    "wind_kmh": 2.0,
    "humidity_pct": 5.0,
    "dew_margin_c": 0.5,
    "cooler_delta_c": 0.5,
}

# Default thresholds that promote a change to WARN severity.
_DEFAULT_WARN: dict[str, float] = {
    "wind_kmh": 25.0,
    "cloud_cover_pct": 60.0,
    "humidity_pct": 90.0,
}


@dataclass
class ScoutSummary:
    text: str
    severity: ScoutSeverity = ScoutSeverity.OK
    signals_changed: dict[str, tuple] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "severity": self.severity.value,
            "signals_changed": {
                k: list(v) for k, v in self.signals_changed.items()
            },
        }


# Pretty-print field names for the human-facing summary.
_FIELD_LABEL: dict[str, str] = {
    "safety_is_safe": "safety monitor",
    "cloud_cover_pct": "cloud cover",
    "wind_kmh": "wind",
    "rain": "rain",
    "humidity_pct": "humidity",
    "dew_margin_c": "dew margin",
    "cooler_delta_c": "cooler delta",
    "mount_at_park": "mount park state",
    "dome_shutter_open": "dome shutter",
    "power_ok": "power",
}


_UNITS: dict[str, str] = {
    "cloud_cover_pct": "%",
    "wind_kmh": " km/h",
    "humidity_pct": "%",
    "dew_margin_c": "°C",
    "cooler_delta_c": "°C",
}


def _fmt(field: str, value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.1f}{_UNITS.get(field, '')}"
    return f"{value}{_UNITS.get(field, '')}"


def _is_noise(field: str, prev: Any, cur: Any) -> bool:
    """True if the change is below the noise floor for numeric fields."""
    floor = _NOISE_FLOOR.get(field)
    if floor is None:
        return False
    if not (isinstance(prev, (int, float)) and isinstance(cur, (int, float))):
        return False
    return abs(cur - prev) < floor


class Scout:
    def __init__(self, *, warn_thresholds: Optional[dict[str, float]] = None):
        self._previous: Optional[SafetyReading] = None
        self._warn_thresholds = {**_DEFAULT_WARN, **(warn_thresholds or {})}

    def observe(self, current: SafetyReading) -> ScoutSummary:
        previous = self._previous
        self._previous = current

        # First call: describe what we know about the world right now.
        if previous is None:
            return self._initial_summary(current)

        changed: dict[str, tuple] = {}
        severity = ScoutSeverity.OK

        for field, label in _FIELD_LABEL.items():
            prev_val = getattr(previous, field)
            cur_val = getattr(current, field)
            if prev_val == cur_val:
                continue
            if _is_noise(field, prev_val, cur_val):
                continue
            changed[field] = (prev_val, cur_val)
            severity = self._promote(severity, self._severity_for_change(field, prev_val, cur_val))

        if not changed:
            return ScoutSummary(
                text="no significant changes since last observation",
                severity=ScoutSeverity.OK,
            )

        parts = []
        for field, (prev_val, cur_val) in changed.items():
            label = _FIELD_LABEL[field]
            parts.append(f"{label}: {_fmt(field, prev_val)} → {_fmt(field, cur_val)}")
        text = "; ".join(parts)
        # Discord content cap (matches the alerter's policy).
        if len(text) > 2000:
            text = text[:1999] + "…"

        return ScoutSummary(text=text, severity=severity, signals_changed=changed)

    # ---- internals ----

    def _initial_summary(self, reading: SafetyReading) -> ScoutSummary:
        present = []
        for field, label in _FIELD_LABEL.items():
            val = getattr(reading, field)
            if val is None:
                continue
            present.append(f"{label}={_fmt(field, val)}")
        if not present:
            return ScoutSummary(
                text="initial observation — no sensors reporting yet (unknown)",
                severity=ScoutSeverity.OK,
            )
        return ScoutSummary(
            text="initial observation: " + ", ".join(present),
            severity=ScoutSeverity.OK,
        )

    @staticmethod
    def _promote(current: ScoutSeverity, new: ScoutSeverity) -> ScoutSeverity:
        order = {ScoutSeverity.OK: 0, ScoutSeverity.WARN: 1, ScoutSeverity.ALERT: 2}
        return new if order[new] > order[current] else current

    def _severity_for_change(self, field: str, prev: Any, cur: Any) -> ScoutSeverity:
        # ALERT-level transitions
        if field == "safety_is_safe" and cur is False:
            return ScoutSeverity.ALERT
        if field == "rain" and cur is True:
            return ScoutSeverity.ALERT
        if field == "power_ok" and cur is False:
            return ScoutSeverity.ALERT

        # A previously-known signal disappearing → device/driver concern (WARN).
        if cur is None and prev is not None:
            return ScoutSeverity.WARN

        # Threshold-based WARN (numeric fields only).
        warn_threshold = self._warn_thresholds.get(field)
        if warn_threshold is not None and isinstance(cur, (int, float)):
            if cur >= warn_threshold:
                return ScoutSeverity.WARN

        return ScoutSeverity.OK
