"""Per-sub image-quality decisions — Phase 4.1.

Operator runs after every captured sub. It looks at HFR, star count, and
guide RMS and decides what should happen next:
  - ACCEPT          : keep the sub, no action
  - RESHOOT         : sub is junk, NINA's sequencer should retake it
  - REQUEST_AF      : focus is drifting, run autofocus before the next sub
  - REQUEST_DITHER  : guiding mediocre, suggest a dither

Phase 4.1 is rule-based (zero LLM tokens). Missing signals don't trigger
anything — same fail-safe as the safety supervisor. A rolling HFR baseline
catches focus drift without needing per-night calibration.

The Operator is stateful (keeps a rolling HFR window) but the state is tiny
and lives in-process; if the orchestrator restarts mid-night, the baseline
rebuilds within a handful of subs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class OperatorAction(str, Enum):
    ACCEPT = "accept"
    RESHOOT = "reshoot"
    REQUEST_AF = "request_af"
    REQUEST_DITHER = "request_dither"


@dataclass
class SubFrameStats:
    """Snapshot of a captured sub's quality metrics. Any field can be None
    meaning 'not measured for this sub' — Operator ignores it."""
    hfr: Optional[float] = None
    star_count: Optional[int] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    guide_rms_total: Optional[float] = None  # arcsec
    filter_name: Optional[str] = None
    exposure_s: Optional[float] = None


@dataclass
class OperatorDecision:
    action: OperatorAction
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)


class Operator:
    """Rule-based per-sub quality decider.

    Priority of rules (first hit wins):
      1. HFR > hfr_max                    → RESHOOT
      2. star_count < star_count_min      → RESHOOT
      3. guide_rms_total > guide_rms_reshoot → RESHOOT
      4. HFR drift from baseline > hfr_drift_threshold → REQUEST_AF
      5. guide_rms_total > guide_rms_dither → REQUEST_DITHER
      6. otherwise                         → ACCEPT
    """

    def __init__(
        self,
        *,
        hfr_max: float = 5.0,
        hfr_drift_threshold: float = 0.8,
        star_count_min: int = 100,
        guide_rms_reshoot: float = 2.0,
        guide_rms_dither: float = 1.0,
        baseline_window: int = 10,
        baseline_min: int = 3,
    ):
        self._hfr_max = hfr_max
        self._hfr_drift_threshold = hfr_drift_threshold
        self._star_count_min = star_count_min
        self._guide_rms_reshoot = guide_rms_reshoot
        self._guide_rms_dither = guide_rms_dither
        self._baseline_min = baseline_min
        self._hfr_history: deque[float] = deque(maxlen=baseline_window)

    def _baseline_hfr(self) -> Optional[float]:
        if len(self._hfr_history) < self._baseline_min:
            return None
        return sum(self._hfr_history) / len(self._hfr_history)

    def evaluate(self, s: SubFrameStats) -> OperatorDecision:
        metrics: dict[str, Any] = {}
        if s.hfr is not None:
            metrics["hfr"] = s.hfr
        if s.star_count is not None:
            metrics["star_count"] = s.star_count
        if s.guide_rms_total is not None:
            metrics["guide_rms_total"] = s.guide_rms_total

        # Rule 1 — HFR over hard max
        if s.hfr is not None and s.hfr > self._hfr_max:
            self._hfr_history.append(s.hfr)  # still update baseline
            return OperatorDecision(
                action=OperatorAction.RESHOOT,
                reason=f"HFR {s.hfr:.2f} exceeds max {self._hfr_max:.2f}",
                metrics=metrics,
            )

        # Rule 2 — star count below floor (clouds, transparency)
        if s.star_count is not None and s.star_count < self._star_count_min:
            if s.hfr is not None:
                self._hfr_history.append(s.hfr)
            return OperatorDecision(
                action=OperatorAction.RESHOOT,
                reason=f"Star count {s.star_count} below min {self._star_count_min}",
                metrics=metrics,
            )

        # Rule 3 — guide RMS catastrophic
        if s.guide_rms_total is not None and s.guide_rms_total > self._guide_rms_reshoot:
            if s.hfr is not None:
                self._hfr_history.append(s.hfr)
            return OperatorDecision(
                action=OperatorAction.RESHOOT,
                reason=f"Guide RMS {s.guide_rms_total:.2f}\" exceeds reshoot threshold",
                metrics=metrics,
            )

        # Rule 4 — HFR drift from rolling baseline (focus drift)
        if s.hfr is not None:
            baseline = self._baseline_hfr()
            self._hfr_history.append(s.hfr)
            if baseline is not None and (s.hfr - baseline) > self._hfr_drift_threshold:
                metrics["baseline_hfr"] = baseline
                return OperatorDecision(
                    action=OperatorAction.REQUEST_AF,
                    reason=(
                        f"HFR {s.hfr:.2f} above baseline {baseline:.2f} by "
                        f">{self._hfr_drift_threshold:.2f} — focus drift, request AF"
                    ),
                    metrics=metrics,
                )

        # Rule 5 — mild guide RMS bump → dither suggestion
        if s.guide_rms_total is not None and s.guide_rms_total > self._guide_rms_dither:
            return OperatorDecision(
                action=OperatorAction.REQUEST_DITHER,
                reason=f"Guide RMS {s.guide_rms_total:.2f}\" above dither threshold",
                metrics=metrics,
            )

        return OperatorDecision(
            action=OperatorAction.ACCEPT,
            reason="OK — within all thresholds",
            metrics=metrics,
        )
