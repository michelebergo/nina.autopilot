"""OpenAI-compatible LLM backend for the wiki agent.

Covers any server speaking the /chat/completions dialect: Ollama and LM Studio
locally, OpenAI, Mistral, and Gemini's OpenAI-compat endpoint in the cloud.
Exposes the same .complete() surface as llm.LLMClient so wiki_ingest can take
either client unchanged.

Two lessons from the NINA plugins are baked in:
  - thinking-capable local models (Gemma 4, Qwen 3.x, DeepSeek) reason at length
    by default: requests send reasoning_effort "none" unless disabled;
  - some runs put the actual answer in a `reasoning` field with empty `content`,
    or inline <think> blocks: both are recovered/stripped.

No cost tracking: pricing varies wildly across compat servers (and is zero for
local ones), so cost_estimate_usd() reports 0 and the nightly budget breaker
does not apply. Choose cloud models consciously.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import httpx

from .llm import LLMResult, TokenUsage


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OpenAICompatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        disable_thinking: bool = True,
        # Local models on modest GPUs can take minutes per long generation.
        timeout_s: float = 600.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        # Local servers ignore the token but some compat frontends require the header.
        self._api_key = api_key or os.getenv("LLMWIKI_API_KEY") or "local"
        self._disable_thinking = disable_thinking
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self.usage_total = TokenUsage()

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        cache_system: bool = True,  # accepted for interface parity; no-op here
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            # The wiki agent always wants strict JSON; constrained decoding is far
            # more reliable than asking a small model nicely. Servers that do not
            # support response_format get a retry without it below.
            "response_format": {"type": "json_object"},
        }
        if self._disable_thinking:
            payload["reasoning_effort"] = "none"

        response = await self._client.post(
            self._endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        if response.status_code == 400 and "response_format" in payload:
            payload.pop("response_format")
            response = await self._client.post(
                self._endpoint,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        response.raise_for_status()
        data = response.json()

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = _THINK_RE.sub("", message.get("content") or "").strip()
        if not text:
            # Thinking models sometimes leave content empty and answer in `reasoning`.
            for key in ("reasoning", "reasoning_content", "thinking"):
                candidate = message.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    text = candidate.strip()
                    break

        u = data.get("usage") or {}
        usage = TokenUsage(
            input_tokens=int(u.get("prompt_tokens") or 0),
            output_tokens=int(u.get("completion_tokens") or 0),
        )
        self.usage_total.add(usage)

        return LLMResult(
            text=text,
            usage=usage,
            stop_reason=str(choice.get("finish_reason") or ""),
            model=model,
        )

    def cost_estimate_usd(self) -> float:
        return 0.0
