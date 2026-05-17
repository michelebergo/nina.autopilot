"""Tests for safety.py — pure-rules safety supervisor.

No LLM. No I/O. Given a snapshot of sensor signals, decide SAFE / WARN /
UNSAFE. The wiki is explicit (`nina-safety-monitor.md`) that the safety
reaction must be hard-coded, not LLM-judged.
"""

from nina_autopilot.safety import (
    SafetyDecision,
    SafetyLevel,
    SafetyReading,
    SafetyThresholds,
    evaluate,
)


def reading(**kwargs) -> SafetyReading:
    """Build a SafetyReading with all fields None unless overridden."""
    return SafetyReading(**kwargs)


class TestNoSignals:
    def test_all_none_is_safe(self):
        """A rig without sensors must not be assumed unsafe — that would block
        users who don't have a Safety Monitor wired up."""
        d = evaluate(reading())
        assert d.level is SafetyLevel.SAFE
        assert d.reasons == []


class TestSafetyMonitorBoolean:
    def test_is_safe_false_unsafe(self):
        d = evaluate(reading(safety_is_safe=False))
        assert d.level is SafetyLevel.UNSAFE
        assert any("safety monitor" in r.lower() for r in d.reasons)
        assert d.triggered_signals["safety_is_safe"] is False

    def test_is_safe_true_no_alarm(self):
        d = evaluate(reading(safety_is_safe=True))
        assert d.level is SafetyLevel.SAFE


class TestRain:
    def test_rain_unsafe(self):
        d = evaluate(reading(rain=True))
        assert d.level is SafetyLevel.UNSAFE
        assert any("rain" in r.lower() for r in d.reasons)


class TestPower:
    def test_power_ok_false_unsafe(self):
        """Lost power → can't keep the dome closed safely either. Abort now."""
        d = evaluate(reading(power_ok=False))
        assert d.level is SafetyLevel.UNSAFE
        assert any("power" in r.lower() for r in d.reasons)


class TestWind:
    def test_wind_above_threshold_unsafe(self):
        d = evaluate(reading(wind_kmh=45.0))
        assert d.level is SafetyLevel.UNSAFE
        assert any("wind" in r.lower() for r in d.reasons)

    def test_wind_below_threshold_safe(self):
        d = evaluate(reading(wind_kmh=5.0))
        assert d.level is SafetyLevel.SAFE

    def test_wind_custom_threshold(self):
        th = SafetyThresholds(wind_max_kmh=20.0)
        d = evaluate(reading(wind_kmh=25.0), thresholds=th)
        assert d.level is SafetyLevel.UNSAFE


class TestClouds:
    def test_cloud_above_threshold_unsafe(self):
        d = evaluate(reading(cloud_cover_pct=95.0))
        assert d.level is SafetyLevel.UNSAFE
        assert any("cloud" in r.lower() for r in d.reasons)

    def test_cloud_borderline_safe(self):
        d = evaluate(reading(cloud_cover_pct=50.0))
        assert d.level is SafetyLevel.SAFE


class TestDewMargin:
    def test_low_dew_margin_unsafe(self):
        """Margin < 2°C → mirror will frost. Same as a closed sky for us."""
        d = evaluate(reading(dew_margin_c=1.0))
        assert d.level is SafetyLevel.UNSAFE
        assert any("dew" in r.lower() for r in d.reasons)

    def test_dew_margin_zero_unsafe(self):
        d = evaluate(reading(dew_margin_c=0.0))
        assert d.level is SafetyLevel.UNSAFE

    def test_negative_dew_margin_unsafe(self):
        """Ambient already below dew point → guaranteed condensation."""
        d = evaluate(reading(dew_margin_c=-1.5))
        assert d.level is SafetyLevel.UNSAFE

    def test_healthy_dew_margin_safe(self):
        d = evaluate(reading(dew_margin_c=5.0))
        assert d.level is SafetyLevel.SAFE


class TestHumidityWarn:
    def test_high_humidity_warns_not_unsafe(self):
        """Humidity alone (without dew-margin info) is a warning, not abort."""
        d = evaluate(reading(humidity_pct=97.0))
        assert d.level is SafetyLevel.WARN
        assert any("humid" in r.lower() for r in d.reasons)

    def test_normal_humidity_safe(self):
        d = evaluate(reading(humidity_pct=60.0))
        assert d.level is SafetyLevel.SAFE


class TestCooler:
    def test_cooler_delta_warn(self):
        """Cam off-target by 2-10°C → warn, but keep imaging."""
        d = evaluate(reading(cooler_delta_c=3.0))
        assert d.level is SafetyLevel.WARN
        assert any("cooler" in r.lower() for r in d.reasons)

    def test_cooler_delta_unsafe(self):
        """Off by >10°C → cooler is failing, lose calibration consistency."""
        d = evaluate(reading(cooler_delta_c=15.0))
        assert d.level is SafetyLevel.UNSAFE

    def test_cooler_delta_negative_treated_as_magnitude(self):
        """Off in either direction matters. Pass absolute value."""
        d = evaluate(reading(cooler_delta_c=-12.0))
        assert d.level is SafetyLevel.UNSAFE


class TestMultiSignalPriority:
    def test_unsafe_dominates_warn(self):
        """High humidity (WARN) + rain (UNSAFE) → UNSAFE."""
        d = evaluate(reading(humidity_pct=98.0, rain=True))
        assert d.level is SafetyLevel.UNSAFE

    def test_reasons_list_collects_all_triggers(self):
        """All offending signals must be reported, not just the first."""
        d = evaluate(reading(rain=True, wind_kmh=60.0, cloud_cover_pct=99.0))
        assert d.level is SafetyLevel.UNSAFE
        # Three distinct rules tripped
        assert len(d.reasons) >= 3
        joined = " ".join(d.reasons).lower()
        assert "rain" in joined and "wind" in joined and "cloud" in joined

    def test_triggered_signals_captures_values(self):
        d = evaluate(reading(rain=True, wind_kmh=60.0))
        assert d.triggered_signals["rain"] is True
        assert d.triggered_signals["wind_kmh"] == 60.0


class TestDecisionAPI:
    def test_safe_predicate(self):
        assert evaluate(reading()).is_safe is True
        assert evaluate(reading(rain=True)).is_safe is False

    def test_unsafe_predicate(self):
        assert evaluate(reading(rain=True)).is_unsafe is True
        assert evaluate(reading()).is_unsafe is False
        # WARN is not unsafe
        assert evaluate(reading(humidity_pct=98.0)).is_unsafe is False
