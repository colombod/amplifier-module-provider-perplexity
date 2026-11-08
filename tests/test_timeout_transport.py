"""Timeout policy at the real SDK/HTTPX request boundary, without sockets.

The provider's SDK constructor only receives an injected HTTPX MockTransport;
complete(), generated SDK create(), serialization and timeout selection are real.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock

import httpx
import perplexity
import pytest
from amplifier_core import MockCoordinator, ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message, TextBlock

import amplifier_module_provider_perplexity as module


@asynccontextmanager
async def provider_wire(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    handler: Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
    coordinator: ModuleCoordinator | None = None,
) -> AsyncIterator[module.PerplexityProvider]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    real_sdk = perplexity.AsyncPerplexity

    def sdk_with_transport(**kwargs):
        return real_sdk(http_client=http, **kwargs)

    # Inject transport only; preserve actual provider construction of SDK timeout.
    with monkeypatch.context() as patch:
        patch.setattr(module, "AsyncPerplexity", sdk_with_transport)
        provider = module.PerplexityProvider(
            api_key="fixture-only",
            config={"max_retries": 0, **config},
            coordinator=coordinator,
        )
        try:
            yield provider
        finally:
            await provider.close()
            await http.aclose()


def completion_request() -> ChatRequest:
    return ChatRequest(
        model="r1-1776",
        messages=[Message(role="user", content="offline fixture")],
        max_output_tokens=17,
        temperature=0.25,
    )


def response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "fixture",
            "model": "r1-1776",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def expected_timeout(value):
    return {"connect": 5.0, "pool": 5.0, "read": value, "write": value}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config,options,read_write",
    [
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
        ({"timeout": 0}, {}, 0),
        ({"timeout": 120}, {"request_options": {"timeout": 0}}, 0),
    ],
    ids=[
        "omitted",
        "config-none",
        "config-int",
        "config-float",
        "explicit-none",
        "mapped-none",
        "explicit-none-over-config",
        "mapped-none-over-config",
        "explicit-finite",
        "mapped-finite",
        "keyword-none-wins",
        "keyword-finite-wins",
        "zero",
        "config-zero",
        "mapped-zero",
    ],
)
async def test_transport_timeout_preserves_connect_and_pool(
    monkeypatch, config, options, read_write
):
    seen = []

    def handler(request):
        seen.append(request)
        return response()

    # Exact hidden selection must remain usable; do not gate inference on listing.
    async with provider_wire(
        monkeypatch, {"model_filters": {"include": []}, **config}, handler
    ) as provider:
        result = await provider.complete(completion_request(), **options)
        block = result.content[0]
        assert isinstance(block, TextBlock)
        assert block.text == "ok"
        assert len(seen) == 1
        body = json.loads(seen[0].content)
        assert body["model"] == "r1-1776" and body["max_tokens"] == 17
        assert body["temperature"] == 0.25
        assert "timeout" not in body and "request_options" not in body
        assert seen[0].headers["authorization"] == "Bearer fixture-only"
        actual = seen[0].extensions["timeout"]
        print(
            json.dumps(
                {
                    "config": config,
                    "options": options,
                    "actual": actual,
                    "expected": expected_timeout(read_write),
                },
                sort_keys=True,
            )
        )
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
        timeout = provider.client.timeout
        assert isinstance(timeout, perplexity.Timeout)
        assert timeout.as_dict() == expected_timeout(120)
        assert seen == [
            expected_timeout(None),
            expected_timeout(3),
            expected_timeout(120),
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, 12, 0, 2.5])
async def test_raw_capture_stays_json_serializable(monkeypatch, value):
    coordinator = MockCoordinator()
    event_spy = AsyncMock(return_value={"action": "continue"})
    coordinator.hooks.register("llm:request", event_spy)
    coordinator.hooks.register("llm:response", event_spy)
    async with provider_wire(
        monkeypatch, {"raw": True}, lambda _: response(), coordinator
    ) as provider:
        await provider.complete(completion_request(), timeout=value)
        event = next(
            call.args[1]
            for call in event_spy.call_args_list
            if call.args[0] == "llm:request"
        )
        assert json.loads(json.dumps(event))["raw"]["timeout"] == value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route,raw",
    [("keyword", True), ("mapping", True), ("config", True), ("keyword", False)],
    ids=["direct-S-raw", "mapped-S-raw", "config-only-S-raw", "direct-S-no-raw"],
)
async def test_structured_raw_capture_preserves_wire_timeout(monkeypatch, route, raw):
    timeout = perplexity.Timeout(connect=1, read=2, write=3, pool=4)
    config: dict[str, Any] = {"raw": raw, "timeout": 120}
    options: dict[str, Any] = {}
    mapped = MappingProxyType({"timeout": timeout})
    if route == "config":
        config["timeout"] = timeout
    elif route == "keyword":
        options["timeout"] = timeout
    else:
        options["request_options"] = mapped
    seen = []
    coordinator = MockCoordinator()
    event_spy = AsyncMock(return_value={"action": "continue"})
    coordinator.hooks.register("llm:request", event_spy)
    coordinator.hooks.register("llm:response", event_spy)

    def handler(request):
        seen.append(request)
        return response()

    async with provider_wire(monkeypatch, config, handler, coordinator) as provider:
        result = await provider.complete(completion_request(), **options)
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "ok"
        assert len(seen) == 1
        assert seen[0].extensions["timeout"] == {
            "connect": 1,
            "read": 2,
            "write": 3,
            "pool": 4,
        }
        body = json.loads(seen[0].content)
        assert "timeout" not in body and "request_options" not in body
        assert timeout.as_dict() == {"connect": 1, "read": 2, "write": 3, "pool": 4}
        assert mapped["timeout"] is timeout and set(mapped) == {"timeout"}
        assert config["raw"] is raw
        assert set(config) == {"raw", "timeout"}
        if route == "config":
            assert config["timeout"] is timeout and options == {}
        elif route == "keyword":
            assert options["timeout"] is timeout and set(options) == {"timeout"}
            assert config["timeout"] == 120
        else:
            assert options["request_options"] is mapped
            assert set(options) == {"request_options"} and config["timeout"] == 120

        events = [call.args[1] for call in event_spy.call_args_list]
        event = next(
            call.args[1]
            for call in event_spy.call_args_list
            if call.args[0] == "llm:request"
        )
        serialized = json.dumps(events)
        assert "fixture-only" not in serialized
        assert "authorization" not in serialized.lower()
        if raw:
            assert "raw" in event
            if route == "config":
                assert "timeout" not in event["raw"]
            else:
                assert event["raw"]["timeout"] == {
                    "connect": 1,
                    "read": 2,
                    "write": 3,
                    "pool": 4,
                }
        else:
            assert "raw" not in event


async def _assert_completion_cancels(
    task: asyncio.Task[Any],
    entered: asyncio.Event,
    unblock: asyncio.Event,
    *,
    bound: float = 2.0,
) -> None:
    readiness = asyncio.create_task(entered.wait())
    try:
        await asyncio.wait(
            {readiness, task}, timeout=bound, return_when=asyncio.FIRST_COMPLETED
        )
        if task.done():
            if task.cancelled():
                raise AssertionError("completion cancelled before guard cancellation")
            task.result()
            raise AssertionError("completion finished before transport readiness")
        if not entered.is_set():
            raise AssertionError("completion transport readiness deadline exceeded")
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=bound)
        assert task in done, "completion cancellation deadline exceeded"
        with pytest.raises(asyncio.CancelledError):
            task.result()
        assert not unblock.is_set()
    finally:
        unblock.set()
        for owned in (readiness, task):
            if not owned.done():
                owned.cancel()
        done, pending = await asyncio.wait({readiness, task}, timeout=bound)
        for finished in done:
            if not finished.cancelled():
                finished.exception()
        assert not pending, "completion guard cleanup deadline exceeded"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [{}, {"timeout": None}, {"request_options": {"timeout": None}}, {"timeout": 60}],
    ids=["default", "explicit-none", "mapped-none", "finite"],
)
async def test_cancellation_propagates_through_blocking_sdk_transport(
    monkeypatch, options
):
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
        await _assert_completion_cancels(task, entered, blocked)
        assert finalized == [True] and len(seen) == 1
        assert seen[0].extensions["timeout"] == expected_timeout(options.get("timeout"))


@pytest.mark.asyncio
async def test_cancellation_guard_rejects_premature_cancellation(monkeypatch):
    entered, unblock, finalized = asyncio.Event(), asyncio.Event(), asyncio.Event()
    seen = []

    async def handler(request):
        seen.append(request)
        try:
            entered.set()
            raise asyncio.CancelledError
        finally:
            finalized.set()

    async with provider_wire(monkeypatch, {}, handler) as provider:
        task = asyncio.create_task(provider.complete(completion_request()))
        try:
            done, pending = await asyncio.wait({task}, timeout=2.0)
            assert not pending and task in done, (
                "premature completion observation deadline exceeded"
            )
            assert task.cancelled() and entered.is_set() and finalized.is_set()
            with pytest.raises(
                AssertionError, match="^completion cancelled before guard cancellation$"
            ):
                await _assert_completion_cancels(task, entered, unblock, bound=0.1)
            assert len(seen) == 1
        finally:
            unblock.set()
            task.cancel()
            done, pending = await asyncio.wait({task}, timeout=2.0)
            assert not pending, "fault completion cleanup deadline exceeded"
            for finished in done:
                if not finished.cancelled():
                    finished.result()


@pytest.mark.asyncio
async def test_cancellation_guard_rejects_swallowed_cancellation(monkeypatch):
    entered, unblock, finalized = asyncio.Event(), asyncio.Event(), asyncio.Event()
    seen = []
    release_timer: asyncio.TimerHandle | None = None

    async def handler(request):
        nonlocal release_timer
        seen.append(request)
        entered.set()
        try:
            try:
                await unblock.wait()
            except asyncio.CancelledError:
                # Owned safety release keeps the deliberately unbounded RED
                # guard finite while still exceeding its 0.1-second deadline.
                release_timer = asyncio.get_running_loop().call_later(1.0, unblock.set)
                await unblock.wait()
                raise
        finally:
            finalized.set()
        return response()

    async with provider_wire(monkeypatch, {}, handler) as provider:
        task = asyncio.create_task(provider.complete(completion_request()))
        try:
            with pytest.raises(
                AssertionError, match="completion cancellation deadline exceeded"
            ):
                await _assert_completion_cancels(task, entered, unblock, bound=0.1)
            assert task.done() and finalized.is_set()
            assert len(seen) == 1
        finally:
            unblock.set()
            task.cancel()
            try:
                done, pending = await asyncio.wait({task}, timeout=2.0)
                assert not pending, "fault completion cleanup deadline exceeded"
                for finished in done:
                    if not finished.cancelled():
                        finished.result()
            finally:
                if release_timer is not None:
                    release_timer.cancel()


@pytest.mark.asyncio
async def test_cancellation_guard_cleans_up_after_missing_readiness(monkeypatch):
    entered, actual_entry = asyncio.Event(), asyncio.Event()
    unblock, finalized = asyncio.Event(), asyncio.Event()
    seen = []

    async def handler(request):
        seen.append(request)
        actual_entry.set()  # Deliberately do not signal the guard's entered event.
        try:
            await unblock.wait()
        finally:
            finalized.set()
        return response()

    async with provider_wire(monkeypatch, {}, handler) as provider:
        task = asyncio.create_task(provider.complete(completion_request()))
        try:
            await asyncio.wait_for(actual_entry.wait(), 2.0)
            with pytest.raises(
                AssertionError,
                match="^completion transport readiness deadline exceeded$",
            ):
                await _assert_completion_cancels(task, entered, unblock, bound=0.1)
            # Observe the guard's cleanup, not the outer safety net below.
            assert task.done(), "completion remains pending after readiness failure"
            assert finalized.is_set()
            assert len(seen) == 1
        finally:
            unblock.set()
            task.cancel()
            done, pending = await asyncio.wait({task}, timeout=2.0)
            assert not pending, "fault completion cleanup deadline exceeded"
            for finished in done:
                if not finished.cancelled():
                    finished.result()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["config", "keyword", "mapping"])
@pytest.mark.parametrize(
    "phases,expected",
    [
        (
            {"connect": 1, "read": 2, "write": 3, "pool": 4},
            {"connect": 1, "read": 2, "write": 3, "pool": 4},
        ),
        (
            {"connect": None, "read": None, "write": None, "pool": None},
            {"connect": None, "read": None, "write": None, "pool": None},
        ),
        (
            {"connect": None, "read": 7, "write": None, "pool": 0},
            {"connect": None, "read": 7, "write": None, "pool": 0},
        ),
        (
            {"connect": 0, "read": 0, "write": 0, "pool": 0},
            {"connect": 0, "read": 0, "write": 0, "pool": 0},
        ),
    ],
    ids=["S", "N", "M", "Z"],
)
async def test_structured_timeout_preserves_explicit_phases(
    monkeypatch, route, phases, expected
):
    timeout = perplexity.Timeout(**phases)
    config: dict[str, Any] = {"timeout": 120}
    options: dict[str, Any] = {}
    if route == "config":
        config["timeout"] = timeout
    elif route == "keyword":
        options["timeout"] = timeout
    else:
        options["request_options"] = {"timeout": timeout}
    seen = []

    def handler(request):
        seen.append(request)
        return response()

    async with provider_wire(monkeypatch, config, handler) as provider:
        try:
            result = await provider.complete(completion_request(), **options)
            assert isinstance(result.content[0], TextBlock)
            assert result.content[0].text == "ok"
            assert len(seen) == 1
            assert seen[0].extensions["timeout"] == expected
            body = json.loads(seen[0].content)
            assert "timeout" not in body and "request_options" not in body
        finally:
            assert timeout.as_dict() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,expected",
    [
        (
            "keyword-N-over-mapping-M",
            {"connect": None, "read": None, "write": None, "pool": None},
        ),
        (
            "keyword-none-over-mapping-M",
            {"connect": 5.0, "read": None, "write": None, "pool": 5.0},
        ),
        (
            "keyword-zero-over-mapping-M",
            {"connect": 5.0, "read": 0, "write": 0, "pool": 5.0},
        ),
        (
            "mapping-none-over-config-S",
            {"connect": 5.0, "read": None, "write": None, "pool": 5.0},
        ),
        (
            "empty-mapping-inherits-config-S",
            {"connect": 1, "read": 2, "write": 3, "pool": 4},
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
async def test_structured_timeout_precedence(monkeypatch, case, expected):
    configured = perplexity.Timeout(connect=1, read=2, write=3, pool=4)
    mapped = perplexity.Timeout(connect=None, read=7, write=None, pool=0)
    direct = perplexity.Timeout(connect=None, read=None, write=None, pool=None)
    options: dict[str, Any] = {"request_options": {"timeout": mapped}}
    if case == "keyword-N-over-mapping-M":
        options["timeout"] = direct
    elif case == "keyword-none-over-mapping-M":
        options["timeout"] = None
    elif case == "keyword-zero-over-mapping-M":
        options["timeout"] = 0
    elif case == "mapping-none-over-config-S":
        options["request_options"] = {"timeout": None}
    else:
        options["request_options"] = {}
    seen = []

    def handler(request):
        seen.append(request)
        return response()

    async with provider_wire(monkeypatch, {"timeout": configured}, handler) as provider:
        result = await provider.complete(completion_request(), **options)
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "ok"
        assert len(seen) == 1
        assert seen[0].extensions["timeout"] == expected
        body = json.loads(seen[0].content)
        assert "timeout" not in body and "request_options" not in body
        assert configured.as_dict() == {"connect": 1, "read": 2, "write": 3, "pool": 4}
        assert mapped.as_dict() == {
            "connect": None,
            "read": 7,
            "write": None,
            "pool": 0,
        }
        assert direct.as_dict() == {
            "connect": None,
            "read": None,
            "write": None,
            "pool": None,
        }


@pytest.mark.asyncio
async def test_structured_timeout_does_not_mutate_inputs_or_cached_client(monkeypatch):
    configured = perplexity.Timeout(connect=1, read=2, write=3, pool=4)
    mapped_timeout = perplexity.Timeout(connect=None, read=7, write=None, pool=0)
    direct = perplexity.Timeout(connect=None, read=None, write=None, pool=None)
    config = {"timeout": configured}
    mapping = MappingProxyType({"timeout": mapped_timeout})
    seen = []

    def handler(request):
        seen.append(request)
        return response()

    async with provider_wire(monkeypatch, config, handler) as provider:
        client = provider.client
        client_timeout = client.timeout
        assert isinstance(client_timeout, perplexity.Timeout)
        calls = [
            (
                {"request_options": mapping},
                {"connect": None, "read": 7, "write": None, "pool": 0},
            ),
            (
                {"timeout": direct},
                {"connect": None, "read": None, "write": None, "pool": None},
            ),
            (
                {"timeout": None},
                {"connect": 5.0, "read": None, "write": None, "pool": 5.0},
            ),
            (
                {"timeout": 0},
                {"connect": 5.0, "read": 0, "write": 0, "pool": 5.0},
            ),
            ({}, {"connect": 1, "read": 2, "write": 3, "pool": 4}),
        ]
        for count, (options, expected) in enumerate(calls, start=1):
            result = await provider.complete(completion_request(), **options)
            assert isinstance(result.content[0], TextBlock)
            assert result.content[0].text == "ok"
            assert len(seen) == count
            assert seen[-1].extensions["timeout"] == expected
            body = json.loads(seen[-1].content)
            assert "timeout" not in body and "request_options" not in body
            assert provider.client is client and client.timeout is client_timeout
            assert provider.timeout is configured
            assert provider.config["timeout"] is configured
            assert config["timeout"] is configured
            assert mapping["timeout"] is mapped_timeout
            assert set(config) == {"timeout"} and set(mapping) == {"timeout"}
            assert configured.as_dict() == {
                "connect": 1,
                "read": 2,
                "write": 3,
                "pool": 4,
            }
            assert client_timeout.as_dict() == {
                "connect": 1,
                "read": 2,
                "write": 3,
                "pool": 4,
            }
            assert mapped_timeout.as_dict() == {
                "connect": None,
                "read": 7,
                "write": None,
                "pool": 0,
            }
            assert direct.as_dict() == {
                "connect": None,
                "read": None,
                "write": None,
                "pool": None,
            }
