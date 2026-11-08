"""Static recommendations, immutable overrides, and exact native SDK inference."""

import importlib
import json

import perplexity
import pytest
from amplifier_core import ModelInfo
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_perplexity import PerplexityProvider


@pytest.mark.asyncio
async def test_default_unfiltered_overrides_and_static_origin():
    provider = PerplexityProvider()
    assert [m.id for m in await provider.list_models()] == [
        "sonar",
        "sonar-pro",
        "sonar-reasoning-pro",
        "sonar-deep-research",
    ]
    assert len(await provider.list_models(filters={})) == 6
    assert [
        m.id for m in await provider.list_models(filters={"include": ["R1-*"]})
    ] == ["r1-1776"]
    assert await provider.list_models(filters={"include": []}) == []
    assert (
        await provider.list_models(filters={"include": ["*"], "exclude": ["*"]}) == []
    )
    assert all(isinstance(m, ModelInfo) for m in await provider.list_models())
    assert provider._client is None
    assert provider.get_model_list_info()["source"] == "static"


@pytest.mark.asyncio
async def test_config_and_returned_metadata_are_not_live_aliases():
    config = {"model_filters": {"include": ["R1-*"]}}
    provider = PerplexityProvider(config=config)
    config["model_filters"]["include"].clear()
    info = provider.get_model_list_info()
    info["default_filters"]["include"].clear()
    info["filters_schema"]["properties"].clear()
    assert [m.id for m in await provider.list_models(filters=None)] == ["r1-1776"]
    assert len(await provider.list_models(filters={})) == 6
    assert [
        m.id for m in await provider.list_models(filters={"include": ["SONAR"]})
    ] == ["sonar"]
    first = await provider.list_models()
    first[0].capabilities.clear()
    assert "offline" in (await provider.list_models())[0].capabilities
    assert provider.get_model_list_info()["filters_schema"]["properties"]


@pytest.mark.parametrize(
    "filters", [None, [], {"include": "*"}, {"hide_dated": "false"}, {"unexpected": 1}]
)
def test_invalid_config(filters):
    with pytest.raises((TypeError, ValueError)):
        PerplexityProvider(config={"model_filters": filters})


@pytest.mark.asyncio
async def test_explicit_hidden_model_reaches_sdk_with_exact_options():
    transport_module = (
        "httpx2"
        if any(
            cls.__module__.startswith("httpx2")
            for cls in perplexity.DefaultAsyncHttpxClient.__mro__
        )
        else "httpx"
    )
    http = importlib.import_module(transport_module)
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        return http.Response(
            200,
            json={
                "id": "test",
                "model": "r1-1776",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    provider = PerplexityProvider(
        api_key="fixture-only", config={"model_filters": {"include": []}}
    )
    provider._client = perplexity.AsyncPerplexity(
        api_key="fixture-only",
        max_retries=0,
        http_client=http.AsyncClient(transport=http.MockTransport(handle)),
    )
    try:
        assert await provider.list_models() == []
        await provider.complete(
            ChatRequest(
                messages=[Message(role="user", content="test")],
                model="r1-1776",
                max_output_tokens=17,
                temperature=0.25,
            )
        )
        assert seen[0]["model"] == "r1-1776"
        assert seen[0]["max_tokens"] == 17
        assert seen[0]["temperature"] == 0.25
    finally:
        await provider.close()
