"""Timeout policy at the real SDK/HTTPX request boundary, without sockets.

The provider's SDK constructor only receives an injected HTTPX MockTransport;
complete(), generated SDK create(), serialization and timeout selection are real.
"""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import perplexity
import pytest
from amplifier_core.message_models import ChatRequest, Message
import amplifier_module_provider_perplexity as module


@asynccontextmanager
async def provider_wire(monkeypatch, config, handler, coordinator=None):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    real_sdk = perplexity.AsyncPerplexity

    def sdk_with_transport(**kwargs):
        return real_sdk(http_client=http, **kwargs)

    # Inject transport only; preserve actual provider construction of SDK timeout.
    with monkeypatch.context() as patch:
        patch.setattr(module, "AsyncPerplexity", sdk_with_transport)
        provider = module.PerplexityProvider(
            api_key="fixture-only", config={"max_retries": 0, **config}, coordinator=coordinator
        )
        try:
            yield provider
        finally:
            await provider.close()
            await http.aclose()


def completion_request():
    return ChatRequest(
        model="r1-1776", messages=[Message(role="user", content="offline fixture")],
        max_output_tokens=17, temperature=0.25,
    )


def response():
    return httpx.Response(200, json={
        "id": "fixture", "model": "r1-1776", "created": 1,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def expected_timeout(value):
    return {"connect": 5.0, "pool": 5.0, "read": value, "write": value}


@pytest.mark.asyncio
@pytest.mark.parametrize("config,options,read_write", [
    ({}, {}, None),
    ({"timeout": None}, {}, None),
    ({"timeout": 120}, {}, 120),
    ({"timeout": 1.25}, {}, 1.25),
    ({}, {"timeout": None}, None),
    ({}, {"request_options": {"timeout": None}}, None),
    ({"timeout": 120}, {"timeout": None}, None),
    ({"timeout": 120}, {"request_options": {"timeout": None}}, None),
    ({"timeout": 120}, {"timeout": 30}, 30),
    ({"timeout": 120}, {"request_options": {"timeout": 2.5}}, 2.5),
    ({"timeout": 120}, {"request_options": {"timeout": 30}, "timeout": None}, None),
    ({"timeout": 120}, {"request_options": {"timeout": None}, "timeout": 2.5}, 2.5),
    ({}, {"timeout": 0}, 0),
], ids=["omitted", "config-none", "config-int", "config-float", "explicit-none",
        "mapped-none", "explicit-none-over-config", "mapped-none-over-config", "explicit-finite",
        "mapped-finite", "keyword-none-wins", "keyword-finite-wins", "zero"])
async def test_transport_timeout_preserves_connect_and_pool(monkeypatch, config, options, read_write):
    seen = []
    def handler(request):
        seen.append(request)
        return response()

    # Exact hidden selection must remain usable; do not gate inference on listing.
    async with provider_wire(monkeypatch, {"model_filters": {"include": []}, **config}, handler) as provider:
        result = await provider.complete(completion_request(), **options)
        assert result.content[0].text == "ok"
        assert len(seen) == 1
        body = json.loads(seen[0].content)
        assert body["model"] == "r1-1776" and body["max_tokens"] == 17
        assert body["temperature"] == 0.25
        assert "timeout" not in body and "request_options" not in body
        assert seen[0].headers["authorization"] == "Bearer fixture-only"
        actual = seen[0].extensions["timeout"]
        print(json.dumps({"config": config, "options": options, "actual": actual, "expected": expected_timeout(read_write)}, sort_keys=True))
        assert actual == expected_timeout(read_write)


@pytest.mark.asyncio
async def test_override_does_not_mutate_client_or_caller_options(monkeypatch):
    seen = []
    def handler(request):
        seen.append(dict(request.extensions["timeout"]))
        return response()
    mapped = {"timeout": None}
    async with provider_wire(monkeypatch, {"timeout": 120}, handler) as provider:
        await provider.complete(completion_request(), request_options=mapped)
        await provider.complete(completion_request(), timeout=3)
        await provider.complete(completion_request())
        assert mapped == {"timeout": None}
        assert provider.timeout == 120
        assert provider.client.timeout.as_dict() == expected_timeout(120)
        assert seen == [expected_timeout(None), expected_timeout(3), expected_timeout(120)]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, 12])
async def test_raw_capture_stays_json_serializable(monkeypatch, value):
    coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=AsyncMock()))
    async with provider_wire(monkeypatch, {"raw": True}, lambda _: response(), coordinator) as provider:
        await provider.complete(completion_request(), timeout=value)
        event = next(call.args[1] for call in coordinator.hooks.emit.call_args_list if call.args[0] == "llm:request")
        assert json.loads(json.dumps(event))["raw"]["timeout"] == value


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{}, {"timeout": None}, {"request_options": {"timeout": None}}, {"timeout": 60}],
                         ids=["default", "explicit-none", "mapped-none", "finite"])
async def test_cancellation_propagates_through_blocking_sdk_transport(monkeypatch, options):
    entered, blocked = asyncio.Event(), asyncio.Event()
    seen = []
    finalized = []
    async def handler(request):
        seen.append(request)
        entered.set()
        try:
            await blocked.wait()
        finally:
            finalized.append(True)
        return response()
    async with provider_wire(monkeypatch, {"max_retries": 2}, handler) as provider:
        task = asyncio.create_task(provider.complete(completion_request(), **options))
        await asyncio.wait_for(entered.wait(), 2)  # test bound, not provider policy
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finalized == [True] and len(seen) == 1
        assert not blocked.is_set()
        assert seen[0].extensions["timeout"] == expected_timeout(options.get("timeout"))
