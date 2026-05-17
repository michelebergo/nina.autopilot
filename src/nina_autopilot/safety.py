"""Hard-coded safety supervisor — pure rules, no LLM.

The wiki's `nina-safety-monitor.md` is explicit: safety reaction must be
hard-coded and distributed across consumers. The Conductor calls evaluate()
on every tick (5 s default); UNSAFE preempts everything else.

Design rules:
* Missing signals (None) DO NOT trigger anything. A rig without a Safety
  Monitor isn't automatically unsafe — only present signals get a vote.
* UNSAFE always wins; WARN only fires when no UNSAFE rule tripped.
* Every triggered rule contributes a reason string; the caller logs the
  full list so post-hoc diagnosis is possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SafetyLevel(str, Enum):
    SAFE = "safe"
    WARN = "warn"
    UNSAFE = "unsafe"


@dataclass
class SafetyReading:
    """Snapshot of all hardware signals at one moment. Any field can be None
    meaning "this signal is not available right now"."""

    safety_is_safe: Optional[bool] = None        # nina_get_safetymonitor_info
    cloud_cover_pct: Optional[float] = None      # nina_get_weather_info
    wind_kmh: Optional[float] = None             # nina_get_weather_info
    rain: Optional[bool] = None                  # nina_get_weather_info (rate>0)
    humidity_pct: Optional[float] = None         # nina_get_weather_info
    dew_margin_c: Optional[float] = None         # ambient_temp - dew_point
    cooler_delta_c: Optional[float] = None       # |target_temp - actual_temp|
    mount_at_park: Optional[bool] = None         # nina_get_mount_info
    dome_shutter_open: Optional[bool] = None     # nina_get_dome_info
    power_ok: Optional[bool] = None              # switch/UPS proxy


@dataclass
class SafetyThresholds:
    """Tunable thresholds — overridable via env in production."""
    wind_max_kmh: float = 30.0
    cloud_max_pct: float = 80.0
    dew_margin_min_c: float = 2.0
    humidity_warn_pct: float = 95.0
    cooler_delta_warn_c: float = 2.0
    cooler_delta_unsafe_c: float = 10.0


@dataclass
class SafetyDecision:
    level: SafetyLevel
    reasons: list[str] = field(default_factory=list)
    triggered_signals: dict[str, Any] = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        return self.level is SafetyLevel.SAFE

    @property
    def is_unsafe(self) -> bool:
        return self.level is SafetyLevel.UNSAFE


def evaluate(
    reading: SafetyReading,
    thresholds: Optional[SafetyThresholds] = None,
) -> SafetyDecision:
    """Pure function. Returns a SafetyDecision capturing every tripped rule."""
    th = thresholds or SafetyThresholds()
    unsafe: list[str] = []
    warn: list[str] = []
    triggered: dict[str, Any] = {}

    if reading.safety_is_safe is False:
        unsafe.append("Safety Monitor reports unsafe")
        triggered["safety_is_safe"] = False

    if reading.rain is True:
        unsafe.append("Rain detected")
        triggered["rain"] = True

    if reading.power_ok is False:
        unsafe.append("Power loss reported")
        triggered["power_ok"] = False

    if reading.wind_kmh is not None and reading.wind_kmh > th.wind_max_kmh:
        unsafe.append(f"Wind {reading.wind_kmh:.1f} km/h exceeds max {th.wind_max_kmh:.1f}")
        triggered["wind_kmh"] = reading.wind_kmh

    if reading.cloud_cover_pct is not None and reading.cloud_cover_pct > th.cloud_max_pct:
        unsafe.append(f"Cloud cover {reading.cloud_cover_pct:.0f}% exceeds max {th.cloud_max_pct:.0f}%")
        triggered["cloud_cover_pct"] = reading.cloud_cover_pct

    if reading.dew_margin_c is not None and reading.dew_margin_c < th.dew_margin_min_c:
        unsafe.append(f"Dew margin {reading.dew_margin_c:.1f}°C below min {th.dew_margin_min_c:.1f}°C")
        triggered["dew_margin_c"] = reading.dew_margin_c

    if reading.cooler_delta_c is not None:
        delta_abs = abs(reading.cooler_delta_c)
        if delta_abs > th.cooler_delta_unsafe_c:
            unsafe.append(f"Cooler off-target by {delta_abs:.1f}°C (>{th.cooler_delta_unsafe_c:.1f})")
            triggered["cooler_delta_c"] = reading.cooler_delta_c
        elif delta_abs > th.cooler_delta_warn_c:
            warn.append(f"Cooler off-target by {delta_abs:.1f}°C")
            triggered["cooler_delta_c"] = reading.cooler_delta_c

    if reading.humidity_pct is not None and reading.humidity_pct > th.humidity_warn_pct:
        warn.append(f"Humidity {reading.humidity_pct:.0f}% above warn threshold")
        triggered["humidity_pct"] = reading.humidity_pct

    if unsafe:
        return SafetyDecision(SafetyLevel.UNSAFE, reasons=unsafe + warn, triggered_signals=triggered)
    if warn:
        return SafetyDecision(SafetyLevel.WARN, reasons=warn, triggered_signals=triggered)
    return SafetyDecision(SafetyLevel.SAFE)
