"""Minimal Anthropic API wrapper for the autopilot's agents.

Three jobs:
  1. Centralize prompt-caching configuration (cache_control on system prompt).
  2. Track cumulative token usage for the nightly budget circuit breaker.
  3. Allow dependency injection so tests don't hit the real API.

This is deliberately NOT a tool-use loop. Phase 3 agents (Planner, Doctor)
do single-turn LLM calls returning structured JSON; we'll add a real
multi-turn AgentRunner the day we need one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class BudgetState(str, Enum):
    NORMAL = "normal"
    DEMOTED = "demoted"  # warn threshold tripped — caller should downgrade model
    HALTED = "halted"    # hard cap tripped — further .complete() will raise


class BudgetExceeded(RuntimeError):
    """Raised when LLMClient.complete() would push spend past nightly_budget_usd."""


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens


@dataclass
class LLMResult:
    text: str
    usage: TokenUsage
    stop_reason: str
    model: str


# Rough $/Mtok rates (input, output). Tuned for budget warnings, not billing.
# Cache-write = 1.25× input, cache-read = 0.1× input (Anthropic standard tiers).
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "haiku":  (1.00,  5.00),
    "sonnet": (3.00, 15.00),
    "opus":  (15.00, 75.00),
}


def _pricing_for(model: str) -> tuple[float, float]:
    m = model.lower()
    for key, rate in _MODEL_PRICING.items():
        if key in m:
            return rate
    return _MODEL_PRICING["sonnet"]  # safe default


class LLMClient:
    """Thin async wrapper around `anthropic.AsyncAnthropic().messages.create`."""

    def __init__(
        self,
        *,
        client: Any = None,
        api_key: Optional[str] = None,
        nightly_budget_usd: Optional[float] = None,
        warn_at_pct: float = 0.80,
    ):
        if client is not None:
            self._client = client
        else:
            # Lazy import so tests don't require ANTHROPIC_API_KEY in env.
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()
        self.usage_total = TokenUsage()
        self._cost_total_usd: float = 0.0
        self._nightly_budget_usd = nightly_budget_usd
        self._warn_at_pct = warn_at_pct

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> LLMResult:
        # Pre-flight budget check — refuse to send if we're already halted.
        if self.budget_state is BudgetState.HALTED:
            raise BudgetExceeded(
                f"Nightly budget ${self._nightly_budget_usd:.2f} exceeded "
                f"(spent ${self._cost_total_usd:.2f}) — refusing further LLM calls."
            )
        if cache_system:
            system_arg: Any = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
        else:
            system_arg = system

        response = await self._client.messages.create(
            model=model,
            system=system_arg,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )

        # Concatenate any text blocks in the response.
        text_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        text = "".join(text_parts)

        u = response.usage
        usage = TokenUsage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        )
        self.usage_total.add(usage)

        in_rate, out_rate = _pricing_for(model)
        cost = (
            usage.input_tokens * in_rate
            + usage.cache_creation_input_tokens * in_rate * 1.25
            + usage.cache_read_input_tokens * in_rate * 0.1
            + usage.output_tokens * out_rate
        ) / 1_000_000.0
        self._cost_total_usd += cost

        return LLMResult(
            text=text,
            usage=usage,
            stop_reason=response.stop_reason,
            model=model,
        )

    def cost_estimate_usd(self) -> float:
        return self._cost_total_usd

    @property
    def budget_state(self) -> BudgetState:
        if self._nightly_budget_usd is None:
            return BudgetState.NORMAL
        if self._cost_total_usd >= self._nightly_budget_usd:
            return BudgetState.HALTED
        if self._cost_total_usd >= self._warn_at_pct * self._nightly_budget_usd:
            return BudgetState.DEMOTED
        return BudgetState.NORMAL

    def budget_remaining_usd(self) -> float:
        if self._nightly_budget_usd is None:
            return float("inf")
        return max(self._nightly_budget_usd - self._cost_total_usd, 0.0)

    def budget_snapshot(self) -> dict[str, Any]:
        # `remaining_usd` is None when no cap is set — `Infinity` is not valid
        # JSON and breaks strict deserializers (e.g. C# Newtonsoft).
        remaining: Any = (
            None if self._nightly_budget_usd is None
            else max(self._nightly_budget_usd - self._cost_total_usd, 0.0)
        )
        return {
            "budget_usd": self._nightly_budget_usd,
            "spent_usd": self._cost_total_usd,
            "remaining_usd": remaining,
            "state": self.budget_state.value,
        }
