"""Tests for operator.py — per-sub image-quality decisions.

The Operator runs after every sub completes. Inputs: HFR, star count, mean
ADU, guide RMS. Output: an OperatorDecision the Conductor (or a future
Operator-aware loop) can act on. Phase 4.1 is rule-based — zero LLM tokens
in steady state. An LLM second-opinion hook is added in a later turn.
"""

import pytest

from nina_autopilot.operator import (
    Operator,
    OperatorAction,
    OperatorDecision,
    SubFrameStats,
)


def stats(**kwargs) -> SubFrameStats:
    """Build a normal-looking sub with all fields populated; tests override."""
    defaults = dict(
        hfr=2.5,
        star_count=800,
        mean=2400.0,
        median=2350.0,
        guide_rms_total=0.6,
        filter_name="L",
        exposure_s=180.0,
    )
    defaults.update(kwargs)
    return SubFrameStats(**defaults)


class TestNominalAccept:
    def test_clean_sub_accepted(self):
        op = Operator()
        d = op.evaluate(stats())
        assert d.action is OperatorAction.ACCEPT
        assert "ok" in d.reason.lower() or "accepted" in d.reason.lower()

    def test_missing_stats_default_to_accept(self):
        """If no signals are present, don't reject — match safety.py semantics."""
        op = Operator()
        d = op.evaluate(SubFrameStats())
        assert d.action is OperatorAction.ACCEPT


class TestHfrReshoot:
    def test_high_hfr_reshoot(self):
        op = Operator()
        d = op.evaluate(stats(hfr=5.5))
        assert d.action is OperatorAction.RESHOOT
        assert "hfr" in d.reason.lower()

    def test_borderline_hfr_under_max_accepted(self):
        op = Operator(hfr_max=4.0)
        d = op.evaluate(stats(hfr=3.9))
        assert d.action is OperatorAction.ACCEPT

    def test_custom_hfr_max(self):
        op = Operator(hfr_max=3.0)
        d = op.evaluate(stats(hfr=3.5))
        assert d.action is OperatorAction.RESHOOT


class TestHfrDriftRequestsAF:
    def test_persistent_hfr_drift_triggers_af(self):
        """Rolling baseline drifts upward — REQUEST_AF before the next sub."""
        op = Operator(hfr_max=10.0, hfr_drift_threshold=0.8)
        # Seed a baseline at HFR ~2.2
        for _ in range(5):
            op.evaluate(stats(hfr=2.2))
        # New sub jumps above the drift threshold — but still under hfr_max
        d = op.evaluate(stats(hfr=3.2))
        assert d.action is OperatorAction.REQUEST_AF
        assert "af" in d.reason.lower() or "focus" in d.reason.lower()

    def test_drift_within_threshold_accepts(self):
        op = Operator(hfr_max=10.0, hfr_drift_threshold=1.0)
        for _ in range(5):
            op.evaluate(stats(hfr=2.2))
        d = op.evaluate(stats(hfr=2.5))
        assert d.action is OperatorAction.ACCEPT

    def test_baseline_needs_minimum_samples(self):
        """With fewer than the baseline-min samples, drift cannot fire."""
        op = Operator(hfr_max=10.0, hfr_drift_threshold=0.5, baseline_min=4)
        op.evaluate(stats(hfr=2.0))
        d = op.evaluate(stats(hfr=3.0))  # would fire but baseline isn't built yet
        assert d.action is OperatorAction.ACCEPT


class TestStarCountReshoot:
    def test_low_star_count_reshoot(self):
        op = Operator(star_count_min=50)
        d = op.evaluate(stats(star_count=20))
        assert d.action is OperatorAction.RESHOOT
        assert "star" in d.reason.lower()


class TestGuideRms:
    def test_high_guide_rms_reshoot(self):
        op = Operator(guide_rms_reshoot=1.5)
        d = op.evaluate(stats(guide_rms_total=2.5))
        assert d.action is OperatorAction.RESHOOT
        assert "guid" in d.reason.lower() or "rms" in d.reason.lower()

    def test_borderline_guide_rms_dither_request(self):
        """Mild guide-RMS bump → suggest a dither (often resolves bad guide star)."""
        op = Operator(guide_rms_dither=1.0, guide_rms_reshoot=2.0)
        d = op.evaluate(stats(guide_rms_total=1.2))
        assert d.action is OperatorAction.REQUEST_DITHER

    def test_normal_guide_rms_accept(self):
        op = Operator(guide_rms_reshoot=2.0)
        d = op.evaluate(stats(guide_rms_total=0.5))
        assert d.action is OperatorAction.ACCEPT


class TestPriority:
    def test_reshoot_dominates_request_af(self):
        """Both rules trip — RESHOOT is higher severity."""
        op = Operator(hfr_max=4.0, hfr_drift_threshold=0.5)
        for _ in range(5):
            op.evaluate(stats(hfr=2.0))
        d = op.evaluate(stats(hfr=4.5))  # over hfr_max AND >drift
        assert d.action is OperatorAction.RESHOOT

    def test_request_af_dominates_request_dither(self):
        """Focus drift + mild guide noise → REQUEST_AF wins (focus more critical)."""
        op = Operator(hfr_max=10.0, hfr_drift_threshold=0.5, guide_rms_dither=0.5)
        for _ in range(5):
            op.evaluate(stats(hfr=2.0, guide_rms_total=0.3))
        d = op.evaluate(stats(hfr=2.8, guide_rms_total=0.7))
        assert d.action is OperatorAction.REQUEST_AF


class TestDecisionMetadata:
    def test_decision_carries_observed_stats(self):
        """Logged decisions need the raw observation for forensics."""
        op = Operator()
        d = op.evaluate(stats(hfr=5.5, star_count=600))
        assert d.metrics["hfr"] == 5.5
        assert d.metrics["star_count"] == 600

    def test_baseline_metric_exposed_when_drift_fires(self):
        op = Operator(hfr_max=10.0, hfr_drift_threshold=0.5)
        for _ in range(5):
            op.evaluate(stats(hfr=2.0))
        d = op.evaluate(stats(hfr=3.0))
        assert "baseline_hfr" in d.metrics


class TestBaselineWindow:
    def test_baseline_uses_only_recent_samples(self):
        """Old samples fall out of the rolling window."""
        op = Operator(hfr_max=10.0, hfr_drift_threshold=0.4, baseline_window=5)
        # Old samples at 4.0 — should drop out
        for _ in range(5):
            op.evaluate(stats(hfr=4.0))
        # Newer samples at 2.0 should replace them within the window
        for _ in range(5):
            op.evaluate(stats(hfr=2.0))
        # Now a 2.3 should be inside the window's tolerance
        d = op.evaluate(stats(hfr=2.3))
        assert d.action is OperatorAction.ACCEPT
