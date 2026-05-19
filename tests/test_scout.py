"""Tests for scout.py — world-state delta detector.

Scout converts a pair of (previous, current) SafetyReading snapshots into a
compact human-readable summary the dashboard can show and Discord can post.
Phase 4.1 is rule-based; an LLM rewriter hook is added in a later turn for
nicer phrasing on noteworthy deltas.

Scout is OBSERVATION only — it does NOT decide actions. That stays with the
safety supervisor (hard rules) and the Doctor (LLM judgment).
"""

import pytest

from nina_autopilot.safety import SafetyReading
from nina_autopilot.scout import Scout, ScoutSeverity, ScoutSummary


class TestInitialReading:
    def test_first_call_describes_initial_state(self):
        s = Scout()
        summary = s.observe(SafetyReading(cloud_cover_pct=10.0, wind_kmh=5.0))
        assert isinstance(summary, ScoutSummary)
        # The initial summary lists what's known about the world
        assert "cloud" in summary.text.lower()
        assert "wind" in summary.text.lower()
        assert summary.severity is ScoutSeverity.OK

    def test_first_call_with_no_signals_says_so(self):
        s = Scout()
        summary = s.observe(SafetyReading())
        assert "no" in summary.text.lower() or "unknown" in summary.text.lower()


class TestNoChange:
    def test_identical_reading_returns_quiet_summary(self):
        s = Scout()
        s.observe(SafetyReading(cloud_cover_pct=20.0, wind_kmh=8.0))
        summary = s.observe(SafetyReading(cloud_cover_pct=20.0, wind_kmh=8.0))
        assert summary.signals_changed == {}
        assert summary.severity is ScoutSeverity.OK

    def test_within_noise_floor_counts_as_no_change(self):
        """A 0.5% cloud-cover jiggle is meaningless. Scout ignores it."""
        s = Scout()
        s.observe(SafetyReading(cloud_cover_pct=20.0, humidity_pct=60.0))
        summary = s.observe(SafetyReading(cloud_cover_pct=20.3, humidity_pct=60.1))
        assert summary.signals_changed == {}


class TestNoteworthyChange:
    def test_cloud_cover_rising(self):
        s = Scout()
        s.observe(SafetyReading(cloud_cover_pct=10.0))
        summary = s.observe(SafetyReading(cloud_cover_pct=55.0))
        assert "cloud" in summary.text.lower()
        assert "10" in summary.text and "55" in summary.text
        assert "cloud_cover_pct" in summary.signals_changed
        assert summary.signals_changed["cloud_cover_pct"] == (10.0, 55.0)

    def test_wind_jump_increases_severity(self):
        """A wind jump that crosses the warn threshold is severity WARN."""
        s = Scout(warn_thresholds={"wind_kmh": 20.0})
        s.observe(SafetyReading(wind_kmh=5.0))
        summary = s.observe(SafetyReading(wind_kmh=25.0))
        assert summary.severity is ScoutSeverity.WARN

    def test_safety_monitor_unsafe_is_alert(self):
        """SafetyMonitor flipping to unsafe → severity ALERT (highest)."""
        s = Scout()
        s.observe(SafetyReading(safety_is_safe=True))
        summary = s.observe(SafetyReading(safety_is_safe=False))
        assert summary.severity is ScoutSeverity.ALERT
        assert "unsafe" in summary.text.lower() or "safety" in summary.text.lower()

    def test_rain_appearing_is_alert(self):
        s = Scout()
        s.observe(SafetyReading(rain=False))
        summary = s.observe(SafetyReading(rain=True))
        assert summary.severity is ScoutSeverity.ALERT
        assert "rain" in summary.text.lower()


class TestSignalAppearedDisappeared:
    def test_signal_appearing_logged(self):
        """A previously-missing signal now reporting → note its arrival."""
        s = Scout()
        s.observe(SafetyReading())
        summary = s.observe(SafetyReading(humidity_pct=80.0))
        assert "humid" in summary.text.lower()
        assert "humidity_pct" in summary.signals_changed
        assert summary.signals_changed["humidity_pct"] == (None, 80.0)

    def test_signal_disappearing_flagged(self):
        """A signal that goes None when it had a value → driver/device may be down."""
        s = Scout()
        s.observe(SafetyReading(wind_kmh=12.0))
        summary = s.observe(SafetyReading())  # everything None now
        assert "wind" in summary.text.lower()
        # Disappearing readings warrant a WARN, not OK
        assert summary.severity in (ScoutSeverity.WARN, ScoutSeverity.ALERT)


class TestMultipleChanges:
    def test_multiple_signals_changed_all_listed(self):
        s = Scout()
        s.observe(SafetyReading(cloud_cover_pct=10.0, humidity_pct=50.0, wind_kmh=5.0))
        summary = s.observe(SafetyReading(cloud_cover_pct=70.0, humidity_pct=90.0, wind_kmh=25.0))
        changed = summary.signals_changed
        assert "cloud_cover_pct" in changed
        assert "humidity_pct" in changed
        assert "wind_kmh" in changed
        # Highest severity wins
        assert summary.severity in (ScoutSeverity.WARN, ScoutSeverity.ALERT)


class TestSummaryShape:
    def test_summary_text_fits_discord_msg(self):
        s = Scout()
        s.observe(SafetyReading(cloud_cover_pct=0.0))
        summary = s.observe(SafetyReading(
            cloud_cover_pct=80.0, wind_kmh=30.0, humidity_pct=95.0,
            safety_is_safe=False, rain=True, dew_margin_c=0.5,
        ))
        assert len(summary.text) <= 2000  # Discord limit

    def test_to_dict_serializable(self):
        s = Scout()
        s.observe(SafetyReading(cloud_cover_pct=10.0))
        summary = s.observe(SafetyReading(cloud_cover_pct=50.0))
        d = summary.to_dict()
        # Must be plain JSON-friendly types so the dashboard / event store can serialize.
        import json
        json.dumps(d)
        assert d["severity"] in {"ok", "warn", "alert"}
        assert isinstance(d["signals_changed"], dict)
