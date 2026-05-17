"""Tests for doctor.py — LLM-driven fault diagnosis.

The Doctor receives a FaultContext describing what broke and returns a
structured DoctorDecision the Conductor can act on. Tests inject a fake LLM
client so we don't hit the real API.
"""

import json

import pytest

from nina_autopilot.doctor import (
    Doctor,
    DoctorAction,
    DoctorDecision,
    FaultContext,
)
from nina_autopilot.llm import LLMClient

# Reuse the FakeAnthropic from the llm tests
from tests.test_llm import FakeAnthropic


def make_doctor(canned_text: str, model: str = "claude-sonnet-4-6") -> tuple[Doctor, FakeAnthropic]:
    fake = FakeAnthropic()
    fake.messages.queue(canned_text)
    llm = LLMClient(client=fake)
    doctor = Doctor(llm=llm, model=model)
    return doctor, fake


def fault(**kwargs) -> FaultContext:
    defaults = dict(
        fault_type="plate_solve_failed",
        fault_message="Solver returned no match (ASTAP). RA/Dec unknown.",
        consecutive_count=1,
    )
    defaults.update(kwargs)
    return FaultContext(**defaults)


class TestDoctorParsesActions:
    async def test_retry_decision(self):
        doctor, _ = make_doctor(json.dumps({
            "action": "retry",
            "reason": "First failure, primary solver may have hiccupped",
        }))
        d = await doctor.diagnose(fault())
        assert d.action is DoctorAction.RETRY
        assert "primary solver" in d.reason

    async def test_replan_decision(self):
        doctor, _ = make_doctor(json.dumps({
            "action": "replan",
            "reason": "Two plate-solve attempts failed at this coordinate",
        }))
        d = await doctor.diagnose(fault(consecutive_count=2))
        assert d.action is DoctorAction.REPLAN

    async def test_park_and_wait_decision(self):
        doctor, _ = make_doctor(json.dumps({
            "action": "park_and_wait",
            "reason": "Cloud cover above threshold",
            "retry_after_s": 600,
        }))
        d = await doctor.diagnose(fault(fault_type="weather_warn"))
        assert d.action is DoctorAction.PARK_AND_WAIT
        assert d.retry_after_s == 600

    async def test_abort_decision(self):
        doctor, _ = make_doctor(json.dumps({
            "action": "abort",
            "reason": "Equipment disconnect could not be recovered",
        }))
        d = await doctor.diagnose(fault(fault_type="disconnect"))
        assert d.action is DoctorAction.ABORT


class TestDoctorContextPassedToLLM:
    async def test_fault_type_and_message_in_user_prompt(self):
        doctor, fake = make_doctor(json.dumps({"action": "retry", "reason": "x"}))
        await doctor.diagnose(fault(
            fault_type="autofocus_failed",
            fault_message="HFR curve did not converge after 9 steps",
        ))
        call = fake.messages.calls[0]
        user_content = call["messages"][0]["content"]
        assert "autofocus_failed" in user_content
        assert "HFR curve did not converge" in user_content

    async def test_consecutive_count_in_user_prompt(self):
        doctor, fake = make_doctor(json.dumps({"action": "abort", "reason": "x"}))
        await doctor.diagnose(fault(consecutive_count=3))
        user_content = fake.messages.calls[0]["messages"][0]["content"]
        assert "3" in user_content

    async def test_recent_events_included_if_provided(self):
        doctor, fake = make_doctor(json.dumps({"action": "retry", "reason": "x"}))
        await doctor.diagnose(fault(recent_events=[
            {"Event": "SLEW_START", "Time": "..."},
            {"Event": "PLATESOLVE_START", "Time": "..."},
            {"Event": "PLATESOLVE_FAILED", "Time": "..."},
        ]))
        user_content = fake.messages.calls[0]["messages"][0]["content"]
        assert "PLATESOLVE_FAILED" in user_content

    async def test_system_prompt_is_cached(self):
        """The Doctor's long domain-rules system prompt MUST be cached."""
        doctor, fake = make_doctor(json.dumps({"action": "retry", "reason": "x"}))
        await doctor.diagnose(fault())
        system = fake.messages.calls[0]["system"]
        assert isinstance(system, list)
        assert system[0].get("cache_control") == {"type": "ephemeral"}


class TestDoctorFailureModes:
    async def test_malformed_json_returns_abort(self):
        """LLM returns garbage → Doctor MUST fail safe = ABORT."""
        doctor, _ = make_doctor("this is not json")
        d = await doctor.diagnose(fault())
        assert d.action is DoctorAction.ABORT
        assert "parse" in d.reason.lower() or "invalid" in d.reason.lower()

    async def test_unknown_action_returns_abort(self):
        """LLM hallucinates an action we don't know about → fail safe = ABORT."""
        doctor, _ = make_doctor(json.dumps({
            "action": "summon_a_replacement_camera",
            "reason": "definitely a valid action",
        }))
        d = await doctor.diagnose(fault())
        assert d.action is DoctorAction.ABORT

    async def test_missing_reason_uses_placeholder(self):
        doctor, _ = make_doctor(json.dumps({"action": "retry"}))
        d = await doctor.diagnose(fault())
        assert d.action is DoctorAction.RETRY
        assert d.reason  # not empty


class TestDoctorJsonExtraction:
    """The LLM sometimes wraps JSON in prose / fenced blocks — we must cope."""

    async def test_json_inside_prose(self):
        doctor, _ = make_doctor(
            'Looking at this fault, my decision is: {"action": "retry", "reason": "first try"}. '
            'Hope that helps.'
        )
        d = await doctor.diagnose(fault())
        assert d.action is DoctorAction.RETRY

    async def test_json_inside_code_fence(self):
        doctor, _ = make_doctor(
            'My analysis:\n```json\n{"action": "abort", "reason": "non-recoverable"}\n```'
        )
        d = await doctor.diagnose(fault())
        assert d.action is DoctorAction.ABORT


class TestDoctorModelTier:
    async def test_uses_specified_model(self):
        doctor, fake = make_doctor(
            json.dumps({"action": "retry", "reason": "x"}),
            model="claude-opus-4-7",
        )
        await doctor.diagnose(fault())
        assert fake.messages.calls[0]["model"] == "claude-opus-4-7"

    async def test_records_token_usage(self):
        """The Doctor must propagate LLM usage so the budget circuit-breaker sees it."""
        fake = FakeAnthropic()
        fake.messages.queue(
            json.dumps({"action": "retry", "reason": "x"}),
            input_tokens=500, output_tokens=50,
        )
        llm = LLMClient(client=fake)
        doctor = Doctor(llm=llm, model="claude-sonnet-4-6")
        d = await doctor.diagnose(fault())
        assert d.usage.input_tokens == 500
        assert d.usage.output_tokens == 50
        # LLMClient's running total reflects the call
        assert llm.usage_total.input_tokens == 500
