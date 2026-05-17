"""Tests for llm.py — minimal Anthropic API wrapper.

The wrapper exists for three reasons:
  1. Centralize prompt-caching configuration (cache_control on system prompt).
  2. Track cumulative token usage for the nightly budget circuit breaker.
  3. Inject a fake client in tests so we don't hit the real API.
"""

import pytest

from nina_autopilot.llm import LLMClient, LLMResult, TokenUsage


# ---------------------------------------------------------------------------
# A fake low-level client matching the Anthropic SDK surface we use.
# It records every call and returns whatever the test queued.
# ---------------------------------------------------------------------------

class FakeAnthropicMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.queued: list = []

    def queue(self, text: str, input_tokens=10, output_tokens=20,
              cache_creation=0, cache_read=0, stop_reason="end_turn"):
        self.queued.append({
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation": cache_creation,
            "cache_read": cache_read,
            "stop_reason": stop_reason,
        })

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.queued:
            raise RuntimeError("FakeAnthropicMessages: no queued response")
        spec = self.queued.pop(0)
        return _FakeMsgResponse(spec)


class _FakeMsgResponse:
    def __init__(self, spec):
        self.content = [_FakeTextBlock(spec["text"])]
        self.stop_reason = spec["stop_reason"]
        self.usage = _FakeUsage(spec)


class _FakeTextBlock:
    type = "text"
    def __init__(self, text): self.text = text


class _FakeUsage:
    def __init__(self, spec):
        self.input_tokens = spec["input_tokens"]
        self.output_tokens = spec["output_tokens"]
        self.cache_creation_input_tokens = spec["cache_creation"]
        self.cache_read_input_tokens = spec["cache_read"]


class FakeAnthropic:
    def __init__(self) -> None:
        self.messages = FakeAnthropicMessages()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLLMResult:
    async def test_returns_text_and_usage(self):
        fake = FakeAnthropic()
        fake.messages.queue("hello there", input_tokens=15, output_tokens=4)
        llm = LLMClient(client=fake)

        result = await llm.complete(
            model="claude-haiku-4-5-20251001",
            system="be brief",
            user="say hi",
        )

        assert isinstance(result, LLMResult)
        assert result.text == "hello there"
        assert result.usage.input_tokens == 15
        assert result.usage.output_tokens == 4


class TestPromptCaching:
    async def test_system_passed_as_cacheable_block(self):
        """The wrapper must wrap `system` in a list with cache_control so the
        Anthropic SDK actually caches it (string form does not cache)."""
        fake = FakeAnthropic()
        fake.messages.queue("ok")
        llm = LLMClient(client=fake)

        await llm.complete(
            model="claude-haiku-4-5-20251001",
            system="long system prompt full of domain rules...",
            user="hi",
            cache_system=True,
        )

        call = fake.messages.calls[0]
        system = call["system"]
        assert isinstance(system, list), "system must be a list of content blocks for cache_control"
        assert system[0]["type"] == "text"
        assert system[0]["text"].startswith("long system prompt")
        assert system[0].get("cache_control") == {"type": "ephemeral"}

    async def test_cache_off_passes_plain_string(self):
        fake = FakeAnthropic()
        fake.messages.queue("ok")
        llm = LLMClient(client=fake)

        await llm.complete(
            model="claude-haiku-4-5-20251001",
            system="tiny prompt",
            user="hi",
            cache_system=False,
        )
        assert fake.messages.calls[0]["system"] == "tiny prompt"


class TestCumulativeUsage:
    async def test_usage_accumulates_across_calls(self):
        """The nightly budget circuit breaker needs running totals."""
        fake = FakeAnthropic()
        fake.messages.queue("one", input_tokens=10, output_tokens=5)
        fake.messages.queue("two", input_tokens=20, output_tokens=8,
                            cache_read=100)
        llm = LLMClient(client=fake)

        await llm.complete(model="m", system="s", user="u")
        await llm.complete(model="m", system="s", user="u")

        assert llm.usage_total.input_tokens == 30
        assert llm.usage_total.output_tokens == 13
        assert llm.usage_total.cache_read_input_tokens == 100

    async def test_cost_estimate_haiku(self):
        """Total cost is approximate but must be a non-zero float when called."""
        fake = FakeAnthropic()
        fake.messages.queue("x", input_tokens=1000, output_tokens=500)
        llm = LLMClient(client=fake)
        await llm.complete(model="claude-haiku-4-5-20251001", system="s", user="u")
        # We don't pin a specific number — just that it computes and is positive.
        assert llm.cost_estimate_usd() > 0


class TestUserMessageShape:
    async def test_user_passed_as_messages(self):
        fake = FakeAnthropic()
        fake.messages.queue("ok")
        llm = LLMClient(client=fake)
        await llm.complete(model="m", system="s", user="hello")
        call = fake.messages.calls[0]
        assert call["messages"] == [{"role": "user", "content": "hello"}]


class TestModelAndMaxTokens:
    async def test_model_threaded_through(self):
        fake = FakeAnthropic()
        fake.messages.queue("ok")
        llm = LLMClient(client=fake)
        await llm.complete(model="claude-sonnet-4-6", system="s", user="u")
        assert fake.messages.calls[0]["model"] == "claude-sonnet-4-6"

    async def test_max_tokens_default_and_override(self):
        fake = FakeAnthropic()
        fake.messages.queue("ok")
        fake.messages.queue("ok")
        llm = LLMClient(client=fake)
        await llm.complete(model="m", system="s", user="u")
        await llm.complete(model="m", system="s", user="u", max_tokens=8192)
        # First call default (we'll define 4096), second is overridden
        assert fake.messages.calls[0]["max_tokens"] == 4096
        assert fake.messages.calls[1]["max_tokens"] == 8192
