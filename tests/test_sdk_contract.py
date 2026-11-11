"""Native SDK wire contracts, strict fixtures and explicitly labeled drift/faults."""

import inspect
import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Literal

import httpx
import perplexity
import pytest
from amplifier_core import HookRegistry, HookResult, MockCoordinator, ModuleCoordinator
from amplifier_core.llm_errors import LLMError
from amplifier_core.message_models import ChatRequest, Message, TextBlock
from perplexity.generated.api import ResponseCreateResponse
from pydantic import ValidationError

import amplifier_module_provider_perplexity as module
from amplifier_module_provider_perplexity._agent_api import (
    convert_agent_response,
    resolve_route,
)


@asynccontextmanager
async def provider_wire(
    monkeypatch: pytest.MonkeyPatch,
    config: Mapping[str, object],
    handler: Callable[[httpx.Request], httpx.Response],
    coordinator: ModuleCoordinator | None = None,
    *,
    expected_max_tokens: int = 17,
) -> AsyncIterator[module.PerplexityProvider]:
    wire_budgets: list[object] = []

    def checked_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.method == "POST"
        assert request.url.path == "/v1/responses"
        assert request.headers["authorization"] == "Bearer fixture-only"
        assert body["model"] == "r1-1776"
        wire_budgets.append(body["max_output_tokens"])
        assert body["temperature"] == 0.25
        assert body["input"] == [
            {"type": "message", "role": "user", "content": "offline fixture"}
        ]
        assert body["stream"] is False and body["background"] is False
        assert body["store"] is False and body["max_steps"] == 1
        assert "effort" not in body and "messages" not in body
        assert "request_options" not in body
        return handler(request)

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(checked_handler), trust_env=False
    )
    real_sdk = perplexity.AsyncPerplexity
    clients: list[perplexity.AsyncPerplexity] = []
    provider: module.PerplexityProvider | None = None

    # Any is confined to this external constructor's keyword boundary.
    def sdk_with_transport(**kwargs: Any) -> perplexity.AsyncPerplexity:
        client = real_sdk(http_client=http, **kwargs)
        clients.append(client)
        return client

    try:
        with monkeypatch.context() as patch:
            patch.setattr(module, "AsyncPerplexity", sdk_with_transport)
            provider = module.PerplexityProvider(
                api_key="fixture-only",
                config={**config, "max_retries": 0},
                coordinator=coordinator,
            )
            yield provider
            # Assert outside SDK dispatch so a budget mismatch is not wrapped as
            # APIConnectionError by the SDK's transport-error handler.
            for budget in wire_budgets:
                assert budget == expected_max_tokens
    finally:
        try:
            if provider is not None:
                await provider.close()
        finally:
            try:
                for client in clients:
                    if not client.is_closed():
                        await client.close()
            finally:
                await http.aclose()
        assert all(client.is_closed() for client in clients)
        assert http.is_closed


def completion_request() -> ChatRequest:
    return ChatRequest(
        model="r1-1776",
        messages=[Message(role="user", content="offline fixture")],
        max_output_tokens=17,
        temperature=0.25,
    )


def response_payload(
    content: object = "Alpha Beta.",
    *,
    finish_reason: Literal["stop", "length"] | None = "stop",
    extra: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Native response fixture; explicit drift must fail strict SDK validation."""
    if isinstance(content, str):
        parts: object = [{"type": "output_text", "text": content, "annotations": []}]
    else:
        parts = content
    payload: dict[str, Any] = {
        "id": "fixture-response",
        "model": "returned-model",
        "created_at": 1,
        "object": "response",
        "status": "completed" if finish_reason == "stop" else "incomplete",
        "error": None,
        "output": [
            {
                "id": "fixture-message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": parts,
            }
        ],
        "usage": {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
        },
        **(extra or {}),
    }
    if finish_reason == "length":
        payload["incomplete_details"] = {"reason": "max_output_tokens"}
    try:
        ResponseCreateResponse.model_validate(payload, strict=True)
    except ValidationError:
        # Current schema excludes incomplete envelopes and malformed/future content.
        # Such cases explicitly exercise unchanged SDK permissive HTTP decoding.
        assert finish_reason != "stop" or not isinstance(content, str)
    return payload


def assert_strict_drift(payload: Mapping[str, object]) -> None:
    """Explicit drift evidence, separate from strict valid-wire qualification."""
    with pytest.raises(ValidationError):
        ResponseCreateResponse.model_validate(payload, strict=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config,core_budget,options,expected",
    [
        (
            {"max_tokens": 23},
            17,
            {"request_options": {"max_tokens": 19}, "max_tokens": 0},
            0,
        ),
        ({"max_tokens": 23}, 17, {"request_options": {"max_tokens": 0}}, 0),
        ({"max_tokens": 23}, 0, {}, 0),
        (
            {"max_tokens": 23},
            17,
            {"request_options": {"max_tokens": 0}, "max_tokens": None},
            17,
        ),
        ({"max_tokens": 23}, 17, {"request_options": {"max_tokens": None}}, 17),
        ({"max_tokens": 23}, None, {"max_tokens": None}, 23),
        ({"max_tokens": 23}, None, {}, 23),
        ({}, None, {}, 4096),
        ({"max_tokens": 0}, None, {}, 0),
    ],
    ids=[
        "direct-zero-over-all",
        "mapped-zero-over-core-config",
        "core-zero-over-config",
        "direct-none-shadows-mapped-zero",
        "mapped-none-falls-back-core",
        "direct-none-core-none-config",
        "omitted-core-none-config",
        "omitted-core-none-default",
        "omitted-core-none-config-zero",
    ],
)
async def test_budget_selection_preserves_zero_at_sdk_wire(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, int],
    core_budget: int | None,
    options: dict[str, object],
    expected: int,
) -> None:
    payload = response_payload()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    request = ChatRequest(
        model="r1-1776",
        messages=[Message(role="user", content="offline fixture")],
        max_output_tokens=core_budget,
        temperature=0.25,
    )
    async with provider_wire(
        monkeypatch, config, handler, expected_max_tokens=expected
    ) as provider:
        if expected == 0:
            # R07: retain precedence vectors, reject zero locally, never fallback.
            with pytest.raises(ValueError, match="max_tokens"):
                await provider.complete(request, **options)
            assert seen == [] and provider._client is None
            return
        result = await provider.complete(request, **options)
    assert len(seen) == 1
    assert json.loads(seen[0].content)["max_output_tokens"] == expected
    block = result.content[0]
    assert isinstance(block, TextBlock) and block.text == "Alpha Beta."


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["config", "mapping", "keyword"])
@pytest.mark.parametrize(
    "option,value,expected",
    [
        (
            "search_domain_filter",
            ["example.com", "-excluded.example"],
            {"search_domain_filter": ["example.com", "-excluded.example"]},
        ),
        ("search_recency_filter", "week", {"search_recency_filter": "week"}),
        (
            "search_context_size",
            "high",
            {"web_search_options": {"search_context_size": "high"}},
        ),
    ],
    ids=["domain-top-level", "recency-top-level", "context-only-nested"],
)
async def test_search_option_wire_placement(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    option: str,
    value: object,
    expected: dict[str, object],
) -> None:
    seen: list[dict[str, object]] = []
    payload = response_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    config = {option: value} if route == "config" else {}
    options = (
        {"request_options": MappingProxyType({option: value})}
        if route == "mapping"
        else {option: value}
        if route == "keyword"
        else {}
    )
    async with provider_wire(monkeypatch, config, handler) as provider:
        if option in ("search_context_size", "search_domain_filter"):
            # R07/R08: context removed; mixed inclusion/exclusion domains reject.
            with pytest.raises(ValueError, match=option):
                await provider.complete(completion_request(), **options)
            assert seen == [] and provider._client is None
            return
        await provider.complete(completion_request(), **options)
    assert len(seen) == 1
    body = seen[0]
    assert body["tools"] == [{"type": "web_search", "filters": expected}]
    assert "web_search_options" not in body
    if option != "search_domain_filter":
        assert "search_domain_filter" not in body
    if option != "search_recency_filter":
        assert "search_recency_filter" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["hour", "day", "month", "year"])
async def test_other_supported_recencies_reach_top_level(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=response_payload())

    async with provider_wire(monkeypatch, {}, handler) as provider:
        await provider.complete(completion_request(), search_recency_filter=value)
    assert seen[0]["tools"] == [
        {"type": "web_search", "filters": {"search_recency_filter": value}}
    ]
    assert "search_recency_filter" not in seen[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["inherit", "mapped", "direct"])
@pytest.mark.parametrize(
    "option,configured,mapped,direct,wire_key",
    [
        (
            "search_domain_filter",
            ["config.example"],
            ["map.example"],
            ["kw.example"],
            "search_domain_filter",
        ),
        ("search_recency_filter", "year", "month", "day", "search_recency_filter"),
        ("search_context_size", "high", "medium", "low", "web_search_options"),
    ],
    ids=["domain", "recency", "context"],
)
async def test_search_option_presence_precedence(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    option: str,
    configured: object,
    mapped: object,
    direct: object,
    wire_key: str,
) -> None:
    config = {option: configured}
    mapping = {} if case == "inherit" else {option: mapped}
    options: dict[str, object] = {"request_options": MappingProxyType(mapping)}
    if case == "direct":
        options[option] = direct
    before = deepcopy(
        (config, mapping, {k: v for k, v in options.items() if k != "request_options"})
    )
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=response_payload())

    async with provider_wire(monkeypatch, config, handler) as provider:
        if option == "search_context_size":
            rejection_before = deepcopy(
                (
                    provider.config,
                    config,
                    mapping,
                    {k: v for k, v in options.items() if k != "request_options"},
                )
            )
            with pytest.raises(ValueError, match=option):
                await provider.complete(completion_request(), **options)
            assert seen == [] and provider._client is None
            assert rejection_before == (
                provider.config,
                config,
                mapping,
                {k: v for k, v in options.items() if k != "request_options"},
            )
            return
        await provider.complete(completion_request(), **options)
        assert provider.config[option] == configured
    assert before == (
        config,
        mapping,
        {k: v for k, v in options.items() if k != "request_options"},
    )
    wanted = {"inherit": configured, "mapped": mapped, "direct": direct}[case]
    assert wire_key not in seen[0]
    assert seen[0]["tools"] == [{"type": "web_search", "filters": {option: wanted}}]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["config", "mapping", "keyword"])
@pytest.mark.parametrize(
    "option,clear,configured,wire_key",
    [
        ("search_domain_filter", None, ["config.example"], "search_domain_filter"),
        ("search_domain_filter", [], ["config.example"], "search_domain_filter"),
        ("search_recency_filter", None, "week", "search_recency_filter"),
        ("search_recency_filter", "none", "week", "search_recency_filter"),
        ("search_context_size", None, "high", "web_search_options"),
    ],
    ids=["domain-null", "domain-empty", "recency-null", "recency-none", "context-null"],
)
async def test_explicit_clear_omits_sdk_field(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    option: str,
    clear: object,
    configured: object,
    wire_key: str,
) -> None:
    config = {option: clear if route == "config" else configured}
    options: dict[str, object] = {}
    if route == "mapping":
        options["request_options"] = MappingProxyType({option: clear})
    elif route == "keyword":
        options.update({"request_options": {option: configured}, option: clear})
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=response_payload())

    async with provider_wire(monkeypatch, config, handler) as provider:
        await provider.complete(completion_request(), **options)
    assert wire_key not in seen[0]
    assert seen[0]["tools"] == [{"type": "web_search"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["config", "mapping", "keyword"])
async def test_domain_list_is_copied_before_request_hook(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    domains = ["original.example"]
    coordinator = MockCoordinator()

    async def mutate_caller_list(_event: str, _data: dict[str, object]) -> HookResult:
        domains.append("hook-added.example")
        return HookResult()

    coordinator.hooks.register("llm:request", mutate_caller_list)
    config = {"search_domain_filter": domains} if route == "config" else {}
    options = (
        {"request_options": MappingProxyType({"search_domain_filter": domains})}
        if route == "mapping"
        else {"search_domain_filter": domains}
        if route == "keyword"
        else {}
    )
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=response_payload())

    async with provider_wire(monkeypatch, config, handler, coordinator) as provider:
        await provider.complete(completion_request(), **options)
    assert domains == ["original.example", "hook-added.example"]
    assert "hook-added.example" not in json.dumps(seen[0]), "caller list alias leaked"
    assert seen[0]["tools"] == [
        {
            "type": "web_search",
            "filters": {"search_domain_filter": ["original.example"]},
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("option", ["return_images", "return_related_questions"])
@pytest.mark.parametrize(
    "config_value,mapping,direct,wanted",
    [
        (False, {}, {}, False),
        (True, {}, {}, True),
        (False, {"value": True}, {}, True),
        (False, {"value": False}, {"value": True}, True),
        (True, {"value": False}, {}, False),
        (True, {"value": True}, {"value": False}, False),
    ],
    ids=[
        "default-false",
        "config-true",
        "mapped-true",
        "direct-true",
        "mapped-false-over-config",
        "direct-false-over-both",
    ],
)
async def test_boolean_options_forward_explicit_false(
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    config_value: bool,
    mapping: dict[str, bool],
    direct: dict[str, bool],
    wanted: bool,
) -> None:
    config = {option: config_value} if config_value else {}
    options: dict[str, object] = {
        "request_options": MappingProxyType(
            {option: mapping["value"]} if mapping else {}
        )
    }
    if direct:
        options[option] = direct["value"]
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=response_payload())

    async with provider_wire(monkeypatch, config, handler) as provider:
        if wanted:
            # R08: true is unsupported, not silently ignored or forwarded.
            with pytest.raises(ValueError, match=option):
                await provider.complete(completion_request(), **options)
            assert seen == [] and provider._client is None
            return
        await provider.complete(completion_request(), **options)
    assert option not in seen[0]
    other = "return_related_questions" if option == "return_images" else "return_images"
    assert other not in seen[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["config", "mapping", "keyword"])
@pytest.mark.parametrize(
    "option,value",
    [
        ("search_domain_filter", "private-domain-canary"),
        ("search_domain_filter", ("private-tuple-canary",)),
        ("search_domain_filter", ["example.com", 7]),
        ("search_recency_filter", "private-recency-canary"),
        ("search_recency_filter", ""),
        ("search_context_size", "private-context-canary"),
        ("search_context_size", ""),
        ("return_images", None),
        ("return_images", "private-image-canary"),
        ("return_images", 1),
        ("return_related_questions", None),
        ("return_related_questions", "private-related-canary"),
        ("return_related_questions", 0),
    ],
    ids=[
        "domain-string",
        "domain-tuple",
        "domain-element",
        "recency-enum",
        "recency-empty",
        "context-enum",
        "context-empty",
        "images-null",
        "images-string",
        "images-int",
        "related-null",
        "related-string",
        "related-int",
    ],
)
async def test_invalid_option_is_rejected_before_wire_without_echo(
    monkeypatch: pytest.MonkeyPatch, route: str, option: str, value: object
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=response_payload())

    config = {option: value} if route == "config" else {}
    options = (
        {"request_options": MappingProxyType({option: value})}
        if route == "mapping"
        else {option: value}
        if route == "keyword"
        else {}
    )
    try:
        # Enclose construction too: either eager or per-call validation is valid.
        with pytest.raises(ValueError) as caught:
            async with provider_wire(monkeypatch, config, handler) as provider:
                await provider.complete(completion_request(), **options)
        assert option in str(caught.value)
        assert "canary" not in str(caught.value)
        assert repr(value) not in str(caught.value)
    finally:
        assert seen == [], "invalid adapter option reached the SDK wire"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_citations", [False, True], ids=["plain", "citations"])
@pytest.mark.parametrize(
    "content,wanted",
    [
        ("Alpha Beta.", "Alpha Beta."),
        ("", ""),
        (None, ""),
        ([], ""),
        (
            [
                {"type": "output_text", "text": "Alpha "},
                {"type": "output_text", "text": "Beta."},
            ],
            "Alpha Beta.",
        ),
        (
            [
                {"type": "output_text", "text": "Alpha "},
                {
                    "type": "image_url",
                    "image_url": "https://example.com/image.png",
                    "text": "DO_NOT_RENDER_IMAGE",
                },
                {
                    "type": "file_url",
                    "file_url": "https://example.com/file.txt",
                    "text": "DO_NOT_RENDER_FILE",
                },
                {
                    "type": "pdf_url",
                    "pdf_url": "https://example.com/file.pdf",
                    "text": "DO_NOT_RENDER_PDF",
                },
                {
                    "type": "video_url",
                    "video_url": "https://example.com/video.mp4",
                    "text": "DO_NOT_RENDER_VIDEO",
                },
                {"type": "output_text", "text": "Beta."},
            ],
            "Alpha Beta.",
        ),
        (
            [
                {
                    "type": "image_url",
                    "image_url": "https://example.com/image.png",
                    "text": "DO_NOT_RENDER_IMAGE",
                }
            ],
            "",
        ),
    ],
    ids=[
        "string",
        "empty-string",
        "null",
        "empty-list",
        "text-fragments",
        "mixed-media-decoys",
        "image-only-decoy",
    ],
)
async def test_sdk_content_normalizes_before_citations(
    monkeypatch: pytest.MonkeyPatch,
    content: object,
    wanted: str,
    with_citations: bool,
) -> None:
    # R10: citation URLs are native annotations, never an appended Sources list.
    payload = response_payload(content)
    if with_citations:
        payload["output"].insert(
            0,
            {
                "id": "cited",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com/second",
                            },
                            {
                                "type": "url_citation",
                                "url": "https://example.com/first",
                            },
                        ],
                    }
                ],
            },
        )
    if content is None or (
        isinstance(content, list)
        and any(item.get("type") != "output_text" for item in content)
    ):
        assert_strict_drift(payload)
    else:
        ResponseCreateResponse.model_validate(payload, strict=True)
    async with provider_wire(
        monkeypatch, {}, lambda _: httpx.Response(200, json=payload)
    ) as provider:
        if content is None:
            # R10: malformed assistant content now fails closed rather than empty.
            with pytest.raises(LLMError, match="invalid response"):
                await provider.complete(completion_request())
            return
        result = await provider.complete(completion_request())
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == wanted
    assert result.metadata is not None
    if with_citations:
        assert result.metadata["perplexity:citations"] == [
            "https://example.com/second",
            "https://example.com/first",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_citations", [False, True], ids=["plain", "citations"])
@pytest.mark.parametrize(
    "chunks,wanted",
    [
        (
            [
                {"type": "output_text", "text": "Alpha "},
                {"type": "output_text", "text": "Beta."},
            ],
            "Alpha Beta.",
        ),
        (
            [
                {"type": "output_text", "text": "Alpha "},
                {
                    "type": "image_url",
                    "image_url": "https://example.com/image.png",
                    "text": "DO_NOT_RENDER_IMAGE",
                },
                {"type": "output_text", "text": "Beta."},
            ],
            "Alpha Beta.",
        ),
        (
            [
                {"type": "output_text", "text": 7},
                {"type": "file_url", "text": "DO_NOT_RENDER_FILE"},
                {"type": "future_media", "text": "DO_NOT_RENDER_FUTURE"},
                {"content": "DO_NOT_STRINGIFY"},
                {"type": "output_text", "text": "ok"},
            ],
            "ok",
        ),
    ],
    ids=["text-fragments", "media-decoy", "ignore-nontext-and-nonstrings"],
)
async def test_dict_chunk_converter_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[dict[str, object]],
    wanted: str,
    with_citations: bool,
) -> None:
    # R10 converter-only post-dump fault, NOT strict/wire qualification.
    payload = response_payload()
    payload["output"][0]["content"] = deepcopy(chunks)
    if with_citations:
        payload["output"][0]["content"].append(
            {
                "type": "output_text",
                "text": "",
                "annotations": [
                    {"type": "url_citation", "url": "https://example.com/source"}
                ],
            }
        )
    response = ResponseCreateResponse.model_validate(response_payload(), strict=True)
    async with provider_wire(
        monkeypatch, {}, lambda _: pytest.fail("converter-only test reached wire")
    ):
        monkeypatch.setattr(response, "model_dump", lambda **_kwargs: deepcopy(payload))
        result = convert_agent_response(response, resolve_route("r1-1776"))
    block = result.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == wanted
    assert result.metadata is not None
    if with_citations:
        assert result.metadata["perplexity:citations"] == ["https://example.com/source"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [None, "stop", "length"])
async def test_finish_reason_preserves_sdk_value(
    monkeypatch: pytest.MonkeyPatch, reason: Literal["stop", "length"] | None
) -> None:
    payload = response_payload(finish_reason=reason)
    if reason != "stop":
        assert_strict_drift(payload)
    async with provider_wire(
        monkeypatch, {}, lambda _: httpx.Response(200, json=payload)
    ) as provider:
        result = await provider.complete(completion_request())
    assert result.finish_reason == reason


@pytest.mark.asyncio
async def test_empty_choices_do_not_invent_a_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = response_payload(extra={"output": []})
    async with provider_wire(
        monkeypatch, {}, lambda _: httpx.Response(200, json=payload)
    ) as provider:
        result = await provider.complete(completion_request())
    block = result.content[0]
    assert isinstance(block, TextBlock) and block.text == ""
    assert result.finish_reason == "stop"  # R10: completed native envelope.


@pytest.mark.asyncio
async def test_future_finish_reason_converter_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # R10: unknown native status fails closed; converter-only future-boundary test.
    validated = ResponseCreateResponse.model_validate(response_payload(), strict=True)
    response = validated.model_copy(update={"status": "future_reason"})
    async with provider_wire(
        monkeypatch, {}, lambda _: pytest.fail("converter-only test reached wire")
    ):
        with pytest.raises(LLMError, match="invalid response"):
            convert_agent_response(response, resolve_route("r1-1776"))


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["present", "null", "absent"])
async def test_usage_optional_on_real_sdk_response(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    payload = response_payload()
    if state == "null":
        payload["usage"] = None
    elif state == "absent":
        del payload["usage"]
    ResponseCreateResponse.model_validate(payload, strict=True)
    async with provider_wire(
        monkeypatch, {}, lambda _: httpx.Response(200, json=payload)
    ) as provider:
        result = await provider.complete(completion_request())
    if state == "present":
        assert result.usage is not None
        assert result.usage.input_tokens == 5
        assert result.usage.output_tokens == 3
        assert result.usage.total_tokens == 8
    else:
        assert result.usage is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields,wanted",
    [
        ({}, {}),
        (
            {
                "citations": None,
                "search_results": None,
                "images": None,
                "related_questions": None,
            },
            {},
        ),
        (
            {
                "citations": [],
                "search_results": [],
                "images": [],
                "related_questions": [],
            },
            {
                "perplexity:citations": [],
                "perplexity:search_results": [],
                "perplexity:images": [],
                "perplexity:related_questions": [],
            },
        ),
        (
            {
                "citations": ["https://example.com/cited"],
                "search_results": [
                    {
                        "title": "Returned title",
                        "url": "https://example.com/found",
                        "date": None,
                        "snippet": "Returned snippet",
                        "extra_rank": 2,
                    }
                ],
                "images": [{"image_url": "https://example.com/image.png", "width": 12}],
                "related_questions": ["Returned question?"],
                "unselected_extra": "not metadata",
            },
            {
                "perplexity:citations": ["https://example.com/cited"],
                "perplexity:search_results": [
                    {
                        "title": "Returned title",
                        "url": "https://example.com/found",
                        "snippet": "Returned snippet",
                        "extra_rank": 2,
                    }
                ],
                "perplexity:images": [
                    {"image_url": "https://example.com/image.png", "width": 12}
                ],
                "perplexity:related_questions": ["Returned question?"],
            },
        ),
        (
            {
                "search_results": [
                    {
                        "title": "Only search",
                        "url": "https://example.com/found",
                        "snippet": "",
                    }
                ]
            },
            {
                "perplexity:search_results": [
                    {
                        "title": "Only search",
                        "url": "https://example.com/found",
                        "snippet": "",
                    }
                ]
            },
        ),
    ],
    ids=[
        "absent",
        "null",
        "explicit-empty",
        "full-selected-json",
        "search-results-only",
    ],
)
async def test_selected_response_metadata_is_namespaced_json(
    monkeypatch: pytest.MonkeyPatch,
    fields: dict[str, object],
    wanted: dict[str, object],
) -> None:
    # R10: retain original absence/null/empty/full/search-only vectors, using
    # native output envelopes. Legacy image/related extras are not native data.
    payload = response_payload()
    # Retired top-level extras may still be returned, but never become native
    # source metadata or text. Their original null/empty/populated vectors remain.
    payload.update(
        {
            key: fields[key]
            for key in ("images", "related_questions", "unselected_extra")
            if key in fields
        }
    )
    output = payload["output"]
    citations = fields.get("citations")
    assert citations is None or isinstance(citations, list)
    native_results = fields.get("search_results")
    assert native_results is None or isinstance(native_results, list)
    if citations is not None:
        output[0]["content"][0]["annotations"] = [
            {"type": "url_citation", "url": url} for url in citations
        ]
    if native_results is not None:
        output.insert(
            0,
            {
                "type": "search_results",
                "results": [{"id": 7, **result} for result in native_results],
            },
        )
    ResponseCreateResponse.model_validate(payload, strict=True)
    async with provider_wire(
        monkeypatch,
        {},
        lambda _: httpx.Response(200, json=payload),
    ) as provider:
        result = await provider.complete(completion_request())
    assert all(isinstance(block, TextBlock) for block in result.content)
    block = result.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "Alpha Beta."
    expected = {
        "perplexity:response_id": "fixture-response",
        "perplexity:model": "returned-model",
        "perplexity:requested_selector": "r1-1776",
        "perplexity:status": "completed",
        "perplexity:usage_cost_status": "missing",
        "perplexity:annotations": [
            {
                "output_index": 1 if native_results is not None else 0,
                "content_index": 0,
                "annotations": [
                    {"type": "url_citation", "url": url} for url in (citations or [])
                ],
            }
        ],
        "perplexity:citations": fields.get("citations") or [],
    }
    if native_results is not None:
        results = [
            {
                key: value
                for key, value in {"id": 7, **item}.items()
                if key != "extra_rank"
            }
            for item in native_results
        ]
        expected["perplexity:search_results"] = results
        expected["perplexity:search_result_items"] = [
            {
                "type": "search_results",
                "output_index": 0,
                "results": results,
            }
        ]
    # R10 replaces retired image/related metadata with omission. Native result
    # IDs/null dates survive, but arbitrary extra_rank is not canonical.
    assert result.metadata is not None
    for key in ("perplexity:images", "perplexity:related_questions"):
        assert key not in result.metadata
    assert result.metadata == expected
    assert json.loads(json.dumps(result.metadata)) == expected


@pytest.mark.asyncio
async def test_empty_response_model_falls_back_to_requested_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = response_payload(extra={"model": ""})
    async with provider_wire(
        monkeypatch, {}, lambda _: httpx.Response(200, json=payload)
    ) as provider:
        result = await provider.complete(completion_request())
    assert result.metadata is not None
    # R10: never invent actual model.
    assert result.metadata["perplexity:model"] is None
    assert result.metadata["perplexity:requested_selector"] == "r1-1776"


def logging_coordinator(
    events: list[tuple[str, dict[str, object]]],
) -> MockCoordinator:
    coordinator = MockCoordinator()

    async def observe(event: str, data: dict[str, object]) -> HookResult:
        events.append((event, deepcopy(data)))
        return HookResult()

    for event in ("llm:request", "llm:response", "provider:error"):
        coordinator.hooks.register(event, observe)
    return coordinator


@pytest.mark.asyncio
async def test_mount_explicit_key_wins_over_conflicting_environment(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    record_property: Callable[[str, object], None],
) -> None:
    """Catch credential precedence changes on the actual registered provider."""
    explicit = "synthetic-explicit-mount-only"
    environment = "synthetic-environment-mount-only"
    monkeypatch.setenv("PERPLEXITY_API_KEY", environment)
    coordinator = MockCoordinator()
    requests: list[httpx.Request] = []
    clients: list[perplexity.AsyncPerplexity] = []
    real_sdk = perplexity.AsyncPerplexity
    payload = response_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        authorized = request.headers["authorization"] == f"Bearer {explicit}"
        assert authorized
        assert request.method == "POST" and request.url.path == "/v1/responses"
        requests.append(request)
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), trust_env=False
    ) as http:

        def sdk_with_transport(**kwargs: Any) -> perplexity.AsyncPerplexity:
            client = real_sdk(http_client=http, **kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(module, "AsyncPerplexity", sdk_with_transport)
        cleanup = await module.mount(
            coordinator, {"api_key": explicit, "max_retries": 0}
        )
        assert callable(cleanup)
        try:
            provider = coordinator.get("providers")["perplexity"]
            assert isinstance(provider, module.PerplexityProvider)
            assert provider.coordinator is coordinator
            assert provider._client is None
            result = await provider.complete(completion_request())
            assert result.content == [TextBlock(text="Alpha Beta.")]
            assert len(requests) == 1 and len(clients) == 1
            assert provider._client is clients[0]
            assert clients[0].max_retries == 0
        finally:
            try:
                cleanup_result = cleanup()
                assert inspect.isawaitable(cleanup_result)
                await cleanup_result
                assert all(client.is_closed() for client in clients)
            finally:
                for client in clients:
                    if not client.is_closed():
                        await client.close()
        assert provider._client is None
        assert all(client.is_closed() for client in clients)
    assert http.is_closed
    assert all(
        explicit not in record.getMessage() and environment not in record.getMessage()
        for record in caplog.records
    )
    record_property("explicit_mount_key_wins", True)
    record_property("actual_registered_provider", True)
    record_property("actual_cleanup_awaited", True)
    record_property("wire_requests", len(requests))


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["deny", "ask_user"])
async def test_noncontinue_hook_results_remain_observational(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
    action: Literal["deny", "ask_user"],
) -> None:
    """Catch accidental approval/cancellation gates in complete's observations."""
    coordinator = MockCoordinator()
    assert isinstance(coordinator.hooks, HookRegistry)
    events: list[str] = []
    requests: list[httpx.Request] = []
    payload = response_payload()

    async def observe(event: str, _data: dict[str, object]) -> HookResult:
        events.append(event)
        assert len(requests) == (0 if event == "llm:request" else 1)
        return HookResult(action=action, reason="Synthetic observation only")

    for event in ("llm:request", "llm:response", "provider:error"):
        coordinator.hooks.register(event, observe)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=payload)

    async with provider_wire(monkeypatch, {}, handler, coordinator) as provider:
        result = await provider.complete(completion_request())
    assert len(requests) == 1
    assert events == ["llm:request", "llm:response"]
    assert result.content == [TextBlock(text="Alpha Beta.")]
    record_property("actual_core_hook_registry", True)
    record_property("noncontinue_action", action)
    record_property("wire_requests", len(requests))


def observe_native_sdk_errors(
    monkeypatch: pytest.MonkeyPatch,
    provider: module.PerplexityProvider,
    errors: list[perplexity.APIError],
) -> None:
    original_create = provider.client.responses.create

    async def transparent_create(**kwargs: Any) -> object:
        # Observational wrapper only: real SDK serialization, dispatch, and errors.
        try:
            return await original_create(**kwargs)
        except perplexity.APIError as error:
            errors.append(error)
            raise

    monkeypatch.setattr(provider.client.responses, "create", transparent_create)


def assert_provider_error_surfaces_safe(
    caplog: pytest.LogCaptureFixture,
    events: list[tuple[str, dict[str, object]]],
    marker: str,
    error_name: str | None,
    level: int,
) -> None:
    # Third-party SDK DEBUG logs are deliberately outside this provider's scope.
    records = [record for record in caplog.records if record.name == module.__name__]
    assert [name for name, _ in events] == ["llm:request"]
    assert marker not in json.dumps(events)
    failures = [record for record in records if record.levelno >= logging.WARNING]
    assert len(failures) == 1
    assert failures[0].levelno == level
    assert all(
        record.exc_info is None and record.stack_info is None for record in records
    )
    assert marker not in "\n".join(record.getMessage() for record in records)
    if error_name is not None:
        assert error_name in failures[0].getMessage()


@pytest.mark.asyncio
@pytest.mark.parametrize("debug", [False, True], ids=["debug-off", "debug-on"])
@pytest.mark.parametrize("raw", [False, True], ids=["raw-off", "raw-on"])
@pytest.mark.parametrize(
    "status,error_class",
    [
        (400, perplexity.BadRequestError),
        (401, perplexity.AuthenticationError),
        (403, perplexity.PermissionDeniedError),
        (404, perplexity.NotFoundError),
        (422, perplexity.UnprocessableEntityError),
        (429, perplexity.RateLimitError),
        (500, perplexity.InternalServerError),
    ],
    ids=["400", "401", "403", "404", "422", "429", "500"],
)
async def test_logging_status_body_stays_out_of_provider_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    record_property: Callable[[str, object], None],
    status: int,
    error_class: type[perplexity.APIStatusError],
    raw: bool,
    debug: bool,
) -> None:
    marker = "BODY_ONLY_SENSITIVE_MARKER"
    payload = {"error": {"message": marker, "type": "synthetic_error"}}
    requests: list[httpx.Request] = []
    events: list[tuple[str, dict[str, object]]] = []
    native_errors: list[perplexity.APIError] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert marker not in request.content.decode("utf-8")
        requests.append(request)
        return httpx.Response(status, json=payload)

    with caplog.at_level(logging.DEBUG, logger=module.__name__):
        async with provider_wire(
            monkeypatch,
            {"raw": raw, "debug": debug},
            handler,
            logging_coordinator(events),
        ) as provider:
            assert provider.client.max_retries == 0
            observe_native_sdk_errors(monkeypatch, provider, native_errors)
            with pytest.raises(error_class) as caught:
                await provider.complete(completion_request())
    assert len(requests) == 1 and len(native_errors) == 1
    assert caught.value is native_errors[0]
    assert type(caught.value) is error_class
    assert caught.value.status_code == status
    assert caught.value.body == payload
    assert caught.value.__cause__ is native_errors[0].__cause__
    assert caught.value.response.status_code == status
    assert caught.value.response.json() == payload
    assert marker in str(caught.value) and marker in json.dumps(caught.value.body)
    assert events[0][1].get("raw") is not None if raw else "raw" not in events[0][1]
    record_property("native_identity_verified", True)
    record_property("wire_requests", len(requests))
    record_property("event_marker_absent", marker not in json.dumps(events))
    assert_provider_error_surfaces_safe(
        caplog,
        events,
        marker,
        error_class.__name__,
        logging.WARNING if status == 429 else logging.ERROR,
    )
    record_property("provider_log_marker_absent", True)
    failure = next(
        record
        for record in caplog.records
        if record.name == module.__name__ and record.levelno >= logging.WARNING
    )
    assert str(status) in failure.getMessage()
    assert failure.getMessage() == (
        f"Perplexity rate limit: {error_class.__name__} status={status}"
        if status == 429
        else f"Perplexity API error: {error_class.__name__} status={status}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport_class,sdk_class",
    [
        (httpx.ConnectError, perplexity.APIConnectionError),
        (httpx.ReadTimeout, perplexity.APITimeoutError),
    ],
    ids=["connect-error", "read-timeout"],
)
async def test_logging_real_transport_cause_preserves_native_identity(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    record_property: Callable[[str, object], None],
    transport_class: type[httpx.TransportError],
    sdk_class: type[perplexity.APIConnectionError],
) -> None:
    marker = "TRANSPORT_CAUSE_SENSITIVE_MARKER"
    causes: list[httpx.TransportError] = []
    native_errors: list[perplexity.APIError] = []
    events: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        cause = transport_class(marker, request=request)
        causes.append(cause)
        raise cause

    with caplog.at_level(logging.DEBUG, logger=module.__name__):
        async with provider_wire(
            monkeypatch,
            {"raw": True, "debug": True},
            handler,
            logging_coordinator(events),
        ) as provider:
            observe_native_sdk_errors(monkeypatch, provider, native_errors)
            with pytest.raises(sdk_class) as caught:
                await provider.complete(completion_request())
    assert len(causes) == 1 and len(native_errors) == 1
    assert type(caught.value) is sdk_class
    assert caught.value is native_errors[0]
    assert caught.value.__cause__ is causes[0]
    assert marker in str(caught.value.__cause__)
    record_property("native_identity_verified", True)
    record_property("cause_identity_verified", True)
    record_property("wire_requests", len(causes))
    record_property("event_marker_absent", marker not in json.dumps(events))
    assert_provider_error_surfaces_safe(caplog, events, marker, None, logging.ERROR)
    record_property("provider_log_marker_absent", True)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["connection", "timeout"])
async def test_logging_injected_native_message_is_not_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    record_property: Callable[[str, object], None],
    kind: str,
) -> None:
    # Deliberate SDK-boundary fault injection, NOT a real HTTPX wire scenario.
    marker = "INJECTED_NATIVE_MESSAGE_SENSITIVE_MARKER"
    request = httpx.Request("POST", "https://example.invalid/v1/responses")
    error = (
        perplexity.APIConnectionError(message=marker, request=request)
        if kind == "connection"
        else perplexity.APITimeoutError(request)
    )
    if kind == "timeout":
        error.message = marker
        error.args = (marker,)
    events: list[tuple[str, dict[str, object]]] = []
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_payload())

    async def injected_create(**_kwargs: Any) -> object:
        raise error

    with caplog.at_level(logging.DEBUG, logger=module.__name__):
        async with provider_wire(
            monkeypatch,
            {"raw": True, "debug": True},
            handler,
            logging_coordinator(events),
        ) as provider:
            monkeypatch.setattr(provider.client.responses, "create", injected_create)
            with pytest.raises(type(error)) as caught:
                await provider.complete(completion_request())
    assert requests == []
    assert caught.value is error
    assert caught.value.message == marker and marker in str(caught.value)
    record_property("native_identity_verified", True)
    record_property("wire_requests", 0)
    record_property("event_marker_absent", marker not in json.dumps(events))
    assert_provider_error_surfaces_safe(
        caplog, events, marker, type(error).__name__, logging.ERROR
    )
    record_property("provider_log_marker_absent", True)
