"""Tests for llm_openai_compat.py — the /chat/completions wiki backend."""

import json

import httpx
import pytest

from nina_autopilot.llm_openai_compat import OpenAICompatClient


def make_client(handler) -> OpenAICompatClient:
    transport = httpx.MockTransport(handler)
    return OpenAICompatClient(
        base_url="http://fake:11434/v1",
        client=httpx.AsyncClient(transport=transport),
    )


class TestOpenAICompatClient:
    async def test_sends_thinking_off_and_parses_content(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": '{"read": []}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            })

        client = make_client(handler)
        result = await client.complete(model="gemma4:27b", system="sys", user="usr")

        assert captured["model"] == "gemma4:27b"
        assert captured["reasoning_effort"] == "none"
        assert captured["response_format"] == {"type": "json_object"}
        assert captured["messages"][0] == {"role": "system", "content": "sys"}
        assert result.text == '{"read": []}'
        assert result.usage.input_tokens == 10
        assert client.cost_estimate_usd() == 0.0

    async def test_retries_without_response_format_on_400(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append("response_format" in body)
            if "response_format" in body:
                return httpx.Response(400, json={"error": "unknown field response_format"})
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {},
            })

        client = make_client(handler)
        result = await client.complete(model="m", system="s", user="u")

        assert calls == [True, False]
        assert result.text == "{}"

    async def test_recovers_answer_from_reasoning_field_and_strips_think(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{
                    "message": {"content": "", "reasoning": '{"edits": []}'},
                    "finish_reason": "stop",
                }],
                "usage": {},
            })

        client = make_client(handler)
        result = await client.complete(model="qwen3:14b", system="s", user="u")
        assert result.text == '{"edits": []}'

        def handler2(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{
                    "message": {"content": "<think>long musing</think>{\"issues\": []}"},
                    "finish_reason": "stop",
                }],
                "usage": {},
            })

        client2 = make_client(handler2)
        result2 = await client2.complete(model="qwen3:14b", system="s", user="u")
        assert result2.text == '{"issues": []}'

    async def test_http_error_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        client = make_client(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await client.complete(model="m", system="s", user="u")
