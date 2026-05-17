"""Tests for the nightly LLM-spend circuit breaker.

The plan's locked decision: minimize tokens. Hard cap with two thresholds:
  - 80% of budget → demoted (warn; caller can downgrade model)
  - 100% of budget → halted (next .complete() raises BudgetExceeded)

This lives on LLMClient because every agent funnels through it; one place
to enforce the rule keeps the agents oblivious to budget mechanics.
"""

import json

import pytest

from nina_autopilot.llm import (
    BudgetExceeded,
    BudgetState,
    LLMClient,
)
from tests.test_llm import FakeAnthropic


class TestBudgetStateThresholds:
    async def test_state_normal_at_zero_spend(self):
        fake = FakeAnthropic()
        fake.messages.queue("ok", input_tokens=10, output_tokens=10)
        # Generous budget so we stay normal
        llm = LLMClient(client=fake, nightly_budget_usd=10.00)
        await llm.complete(model="claude-haiku-4-5", system="s", user="u")
        assert llm.budget_state is BudgetState.NORMAL

    async def test_state_demoted_above_warn_threshold(self):
        fake = FakeAnthropic()
        # 1M input + 500k output on Sonnet ≈ $3 + $7.5 = $10.5. Set budget to $13 → ~80% spent.
        fake.messages.queue("x", input_tokens=1_000_000, output_tokens=500_000)
        llm = LLMClient(client=fake, nightly_budget_usd=13.00, warn_at_pct=0.80)
        await llm.complete(model="claude-sonnet-4-6", system="s", user="u")
        assert llm.budget_state is BudgetState.DEMOTED

    async def test_state_halted_at_or_above_hard_cap(self):
        fake = FakeAnthropic()
        fake.messages.queue("x", input_tokens=1_000_000, output_tokens=500_000)
        llm = LLMClient(client=fake, nightly_budget_usd=5.00)  # well under cost
        await llm.complete(model="claude-sonnet-4-6", system="s", user="u")
        assert llm.budget_state is BudgetState.HALTED


class TestBudgetEnforcement:
    async def test_next_call_after_halt_raises(self):
        fake = FakeAnthropic()
        fake.messages.queue("x", input_tokens=1_000_000, output_tokens=500_000)
        # Second call should never reach the fake — circuit breaker blocks it.
        llm = LLMClient(client=fake, nightly_budget_usd=1.00)
        await llm.complete(model="claude-sonnet-4-6", system="s", user="u")
        assert llm.budget_state is BudgetState.HALTED

        with pytest.raises(BudgetExceeded):
            await llm.complete(model="claude-sonnet-4-6", system="s", user="u")
        # Fake recorded exactly one call (the second was blocked before send)
        assert len(fake.messages.calls) == 1

    async def test_no_budget_means_no_enforcement(self):
        """Backwards compat: omitting the budget leaves the breaker disabled."""
        fake = FakeAnthropic()
        for _ in range(3):
            fake.messages.queue("x", input_tokens=10_000_000)
        llm = LLMClient(client=fake)  # no budget
        for _ in range(3):
            await llm.complete(model="claude-opus-4-7", system="s", user="u")
        assert llm.budget_state is BudgetState.NORMAL

    async def test_budget_remaining_helper(self):
        fake = FakeAnthropic()
        fake.messages.queue("x", input_tokens=100_000)  # $0.10 on Haiku input
        llm = LLMClient(client=fake, nightly_budget_usd=5.00)
        await llm.complete(model="claude-haiku-4-5", system="s", user="u")
        remaining = llm.budget_remaining_usd()
        assert 4.0 < remaining < 5.0


class TestBudgetSnapshot:
    async def test_snapshot_includes_state_and_cost(self):
        fake = FakeAnthropic()
        fake.messages.queue("x", input_tokens=10_000, output_tokens=2_000)
        llm = LLMClient(client=fake, nightly_budget_usd=2.00)
        await llm.complete(model="claude-haiku-4-5", system="s", user="u")
        snap = llm.budget_snapshot()
        assert snap["budget_usd"] == 2.00
        assert snap["state"] in {"normal", "demoted", "halted"}
        assert snap["spent_usd"] > 0
        assert snap["remaining_usd"] >= 0
