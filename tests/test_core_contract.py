"""Actual Core admission plus separate typed native-wire qualification.

The inherited checks never exercise complete(). The mounted fixture binds the
actual registry object and awaits cleanup; separate completions retain budgets,
usage, hooks and SDK parser coverage.
"""

import json
from collections.abc import AsyncIterator
from importlib.metadata import entry_points
from inspect import isawaitable
from pathlib import Path
from typing import Any

import httpx
import perplexity
import pytest
import pytest_asyncio
from amplifier_core import (
    ChatRequest,
    ChatResponse,
    HookRegistry,
    HookResult,
    Message,
    MockCoordinator,
    TextBlock,
)
from amplifier_core.validation.behavioral import ProviderBehaviorTests
from amplifier_core.validation.structural import ProviderStructuralTests
from perplexity.generated.api import ResponseCreateResponse
from pydantic import ValidationError

import amplifier_module_provider_perplexity as module
from amplifier_module_provider_perplexity import PerplexityProvider


@pytest.fixture(autouse=True)
def synthetic_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Also credential the structural validator's own independent real mount."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "fixture-only")


@pytest.fixture
def module_path(record_property) -> Path:
    path = Path(module.__file__).resolve().parent
    assert path.is_dir() and (path / "__init__.py").is_file()
    record_property("provider_module_path", str(path))
    return path


@pytest.fixture
def coordinator() -> MockCoordinator:
    coordinator = MockCoordinator()
    assert isinstance(coordinator.hooks, HookRegistry)
    return coordinator


@pytest.fixture
def native_payload() -> dict[str, object]:
    payload = {
        "id": "test-response",
        "object": "response",
        "created_at": 1,
        "model": "vendor/returned-model",
        "status": "completed",
        "error": None,
        "output": [
            {
                "id": "fixture-message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "partial", "annotations": []}
                ],
            }
        ],
        "usage": {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40},
    }
    ResponseCreateResponse.model_validate(payload, strict=True)
    return payload


@pytest_asyncio.fixture
async def native_wire(
    monkeypatch: pytest.MonkeyPatch, native_payload: dict[str, object], record_property
) -> AsyncIterator[list[httpx.Request]]:
    seen: list[httpx.Request] = []
    clients: list[perplexity.AsyncPerplexity] = []
    real_sdk = perplexity.AsyncPerplexity

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST" and request.url.path == "/v1/responses"
        assert request.headers["authorization"] == "Bearer fixture-only"
        seen.append(request)
        return httpx.Response(200, json=native_payload)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), trust_env=False
    ) as http:

        def sdk_with_transport(**kwargs):
            client = real_sdk(http_client=http, **kwargs)
            clients.append(client)
            return client

        # Patch the provider's imported alias BEFORE mount or lazy client access.
        monkeypatch.setattr(module, "AsyncPerplexity", sdk_with_transport)
        try:
            yield seen
        finally:
            # provider_module's real cleanup must run before this fixture exits.
            assert all(client.is_closed() for client in clients)
            if clients:
                assert http.is_closed
            record_property("sdk_created_and_closed", len(clients))
    assert http.is_closed


@pytest_asyncio.fixture
async def provider_module(
    coordinator: MockCoordinator, native_wire: list[httpx.Request], record_property
) -> AsyncIterator[PerplexityProvider]:
    cleanup = await module.mount(coordinator, {"raw": True, "max_retries": 0})
    assert callable(cleanup)
    provider = coordinator.get("providers")["perplexity"]
    try:
        assert isinstance(provider, PerplexityProvider)
        assert provider.coordinator is coordinator
        assert provider._client is None and native_wire == []
        yield provider
    finally:
        client = provider._client
        result = cleanup()
        assert isawaitable(result)
        await result
        assert provider._client is None
        if client is not None:
            assert client.is_closed()
        record_property("registered_provider_identity", True)
        record_property("actual_cleanup_awaited", True)


class TestPerplexityStructural(ProviderStructuralTests):
    """Inherit the real exported Core structural test, without body overrides."""


class TestPerplexityBehavior(ProviderBehaviorTests):
    """Inherit all five real exported Core behavior tests, without overrides."""


@pytest.mark.asyncio
async def test_typed_request_output_bound_and_hooks(
    provider_module: PerplexityProvider,
    coordinator: MockCoordinator,
    native_wire: list[httpx.Request],
    native_payload: dict[str, object],
):
    # Known SDK enum drift: strict rejects incomplete; unchanged real HTTP-200
    # parsing must retain the literal status and the extra truncation reason.
    native_payload["status"] = "incomplete"
    native_payload["incomplete_details"] = {"reason": "max_output_tokens"}
    with pytest.raises(ValidationError) as caught:
        ResponseCreateResponse.model_validate(native_payload, strict=True)
    assert [error["loc"] for error in caught.value.errors()] == [("status",)]
    events: list[tuple[str, dict[str, Any]]] = []

    async def observe(event: str, data: dict[str, Any]) -> HookResult:
        events.append((event, data))
        return HookResult()

    coordinator.hooks.register("llm:request", observe)
    coordinator.hooks.register("llm:response", observe)
    result = await provider_module.complete(
        ChatRequest(
            model="sonar-pro",
            max_output_tokens=100,
            temperature=0,
            messages=[Message(role="user", content="Question")],
        )
    )
    assert len(native_wire) == 1
    body = json.loads(native_wire[0].content)
    assert body["max_output_tokens"] == 100
    assert body["temperature"] == 0
    assert body["effort"] == "fast" and "model" not in body
    assert body["input"] == [{"type": "message", "role": "user", "content": "Question"}]
    assert "messages" not in body and "max_tokens" not in body
    assert type(result) is ChatResponse
    assert result.content == [TextBlock(text="partial")]
    usage = result.usage
    metadata = result.metadata
    assert usage is not None
    assert metadata is not None
    assert result.finish_reason == "length" and usage.total_tokens == 40
    assert usage.input_tokens == 30 and usage.output_tokens == 10
    assert metadata["perplexity:response_id"] == "test-response"
    assert metadata["perplexity:model"] == "vendor/returned-model"
    assert metadata["perplexity:requested_selector"] == "sonar-pro"
    assert metadata["perplexity:incomplete_details"] == {"reason": "max_output_tokens"}
    assert [event for event, _ in events] == ["llm:request", "llm:response"]
    event = events[0][1]
    assert event["raw"]["input"] == body["input"]
    assert event["raw"]["max_output_tokens"] == 100
    assert event["raw"] == body
    assert "api_key" not in event["raw"]
    serialized = json.dumps(events)
    assert (
        "fixture-only" not in serialized and "authorization" not in serialized.lower()
    )
    assert events[1][1]["usage"] == {"input_tokens": 30, "output_tokens": 10}
    assert events[1][1]["model"] == "vendor/returned-model"
    assert events[1][1]["metadata"] == {"api": "agent", "status": "incomplete"}
    assert json.loads(result.model_dump_json())["usage"]["total_tokens"] == 40
    timeout = provider_module.client.timeout
    assert isinstance(timeout, perplexity.Timeout)
    assert timeout.read is None and timeout.connect == 5
    assert timeout.write is None and timeout.pool == 5
    assert native_wire[0].extensions["timeout"] == timeout.as_dict()
    assert "timeout" not in body


@pytest.mark.asyncio
async def test_entrypoint_mounts_actual_perplexity(
    provider_module: PerplexityProvider, coordinator: MockCoordinator
):
    entrypoint = list(
        entry_points(group="amplifier.modules", name="provider-perplexity")
    )
    assert len(entrypoint) == 1 and entrypoint[0].load() is module.mount
    assert coordinator.get("providers")["perplexity"] is provider_module
    assert provider_module.name == "perplexity"


@pytest.mark.asyncio
async def test_cleanup_closes_real_client(
    provider_module: PerplexityProvider, native_wire: list[httpx.Request]
):
    # Fixture teardown observes the actual constructed SDK/HTTPX closure and
    # cleared cache after awaited mount cleanup, not a non-None cleanup proxy.
    result = await provider_module.complete(
        ChatRequest(messages=[Message(role="user", content="offline completed")])
    )
    assert len(native_wire) == 1
    assert result.finish_reason == "stop"
    assert result.content == [TextBlock(text="partial")]
    assert provider_module._client is not None
    assert not provider_module._client.is_closed()
