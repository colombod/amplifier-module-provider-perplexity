"""Static recommendations, immutable overrides, and exact native SDK inference."""

import json

import httpx
import perplexity
import pytest
from amplifier_core import ModelInfo
from amplifier_core.message_models import ChatRequest, Message, TextBlock
from perplexity.generated.api import ResponseCreateResponse

import amplifier_module_provider_perplexity as module
from amplifier_module_provider_perplexity import PerplexityProvider
from amplifier_module_provider_perplexity._model_listing import selector_schema


@pytest.mark.asyncio
async def test_default_unfiltered_overrides_and_static_origin():
    provider = PerplexityProvider()
    assert [m.id for m in await provider.list_models()] == [
        "agent-fast",
        "agent-low",
        "agent-medium",
        "agent-high",
    ]
    assert [m.id for m in await provider.list_models(filters={})] == [
        "agent-fast",
        "agent-low",
        "agent-medium",
        "agent-high",
        "sonar",
        "sonar-pro",
        "sonar-reasoning-pro",
        "sonar-deep-research",
    ]
    assert [
        m.id for m in await provider.list_models(filters={"include": ["SONAR-PRO"]})
    ] == ["sonar-pro"]
    assert await provider.list_models(filters={"include": []}) == []
    assert (
        await provider.list_models(filters={"include": ["*"], "exclude": ["*"]}) == []
    )
    assert all(isinstance(m, ModelInfo) for m in await provider.list_models())
    assert provider._client is None
    assert provider.get_model_list_info() == {
        "version": 1,
        "source": "static",
        "filters_keyword": "filters",
        "filters_schema": selector_schema(),
        "default_filters": {
            "include": ["agent-fast", "agent-low", "agent-medium", "agent-high"],
            "exclude": [],
            "hide_dated": False,
        },
        "unfiltered": {},
        "override": "replace",
    }
    assert [m.display_name for m in await provider.list_models(filters={})] == [
        "Agent Fast",
        "Agent Low",
        "Agent Medium",
        "Agent High",
        "Sonar (Agent Fast alias)",
        "Sonar Pro (Agent Fast alias)",
        "Sonar Reasoning Pro (Agent Low alias)",
        "Sonar Deep Research (Agent High alias)",
    ]
    assert provider._client is None


@pytest.mark.asyncio
async def test_config_and_returned_metadata_are_not_live_aliases():
    config = {"model_filters": {"include": ["SONAR-PRO"]}}
    provider = PerplexityProvider(config=config)
    config["model_filters"]["include"].clear()
    info = provider.get_model_list_info()
    info["default_filters"]["include"].clear()
    info["filters_schema"]["properties"].clear()
    assert [m.id for m in await provider.list_models(filters=None)] == ["sonar-pro"]
    assert len(await provider.list_models(filters={})) == 8
    assert [
        m.id for m in await provider.list_models(filters={"include": ["SONAR"]})
    ] == ["sonar"]
    first = await provider.list_models()
    first[0].capabilities.clear()
    first[0].defaults["max_tokens"] = 1
    first[0].id = "mutated"
    fresh = (await provider.list_models())[0]
    assert fresh.id == "sonar-pro"
    assert fresh.capabilities == ["web_search", "citations"]
    assert fresh.defaults == {}
    assert provider.get_model_list_info()["filters_schema"]["properties"]
    assert provider._client is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_id,context_window,max_output_tokens,capabilities",
    [
        ("sonar", 0, 0, ["web_search", "citations"]),
        ("sonar-pro", 0, 0, ["web_search", "citations"]),
        ("sonar-reasoning-pro", 0, 0, ["web_search", "citations"]),
        ("sonar-deep-research", 0, 0, ["web_search", "citations"]),
        ("agent-fast", 0, 0, ["web_search", "citations"]),
        ("agent-low", 0, 0, ["web_search", "citations"]),
        ("agent-medium", 0, 0, ["web_search", "citations"]),
        ("agent-high", 0, 0, ["web_search", "citations"]),
    ],
    ids=[
        "sonar",
        "sonar-pro",
        "sonar-reasoning-pro",
        "sonar-deep-research",
        "agent-fast",
        "agent-low",
        "agent-medium",
        "agent-high",
    ],
)
async def test_retained_model_limits_and_adapter_capabilities(
    model_id: str,
    context_window: int,
    max_output_tokens: int,
    capabilities: list[str],
) -> None:
    provider = PerplexityProvider()
    models = await provider.list_models(filters={})
    model = next(model for model in models if model.id == model_id)
    assert isinstance(model, ModelInfo)
    assert model.context_window == context_window
    assert model.max_output_tokens == max_output_tokens
    assert model.capabilities == capabilities
    assert model.defaults == {} and model.pricing is None
    assert ModelInfo.model_validate_json(model.model_dump_json()) == model
    assert provider._client is None


@pytest.mark.parametrize(
    "filters", [None, [], {"include": "*"}, {"hide_dated": "false"}, {"unexpected": 1}]
)
def test_invalid_config(filters):
    with pytest.raises((TypeError, ValueError)):
        PerplexityProvider(config={"model_filters": filters})


@pytest.mark.asyncio
async def test_explicit_hidden_model_reaches_sdk_with_exact_options(monkeypatch):
    # Offline explicit unlisted ID, not a claim of availability or entitlement.
    payload = {
        "id": "test",
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
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    ResponseCreateResponse.model_validate(payload, strict=True)
    seen: list[httpx.Request] = []
    clients: list[perplexity.AsyncPerplexity] = []
    real_sdk = perplexity.AsyncPerplexity

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), trust_env=False
    ) as http:

        def sdk_with_transport(**kwargs):
            client = real_sdk(http_client=http, **kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(module, "AsyncPerplexity", sdk_with_transport)
        provider = PerplexityProvider(
            api_key="fixture-only",
            config={"model_filters": {"include": []}, "max_retries": 0},
        )
        try:
            assert await provider.list_models() == []
            assert clients == [] and provider._client is None
            result = await provider.complete(
                ChatRequest(
                    messages=[Message(role="user", content="test")],
                    model="vendor/Hidden-ID",
                    max_output_tokens=17,
                    temperature=0.25,
                )
            )
            assert len(seen) == len(clients) == 1
            assert seen[0].method == "POST" and seen[0].url.path == "/v1/responses"
            body = json.loads(seen[0].content)
            assert body["model"] == "vendor/Hidden-ID" and "effort" not in body
            assert body["max_output_tokens"] == 17
            assert body["temperature"] == 0.25
            assert body["input"] == [
                {"type": "message", "role": "user", "content": "test"}
            ]
            assert "max_tokens" not in body and "messages" not in body
            assert result.content == [TextBlock(text="ok")]
            assert result.metadata is not None
            assert result.metadata["perplexity:model"] == "vendor/returned-model"
            assert (
                result.metadata["perplexity:requested_selector"] == "vendor/Hidden-ID"
            )
        finally:
            await provider.close()
        assert provider._client is None
        assert all(client.is_closed() for client in clients)
        assert http.is_closed


@pytest.mark.asyncio
async def test_native_filter_replacement_globs_and_instance_isolation():
    aliases = PerplexityProvider(config={"model_filters": {"include": ["SoNaR*"]}})
    canonical = PerplexityProvider(
        config={"model_filters": {"include": ["AGENT-?"], "exclude": ["*HIGH"]}}
    )
    assert [m.id for m in await aliases.list_models()] == [
        "sonar",
        "sonar-pro",
        "sonar-reasoning-pro",
        "sonar-deep-research",
    ]
    assert await canonical.list_models() == []
    assert [
        m.id for m in await aliases.list_models(filters={"exclude": ["SONAR*"]})
    ] == ["agent-fast", "agent-low", "agent-medium", "agent-high"]
    assert [
        m.id
        for m in await canonical.list_models(
            filters={"include": ["AGENT-[fl]*", "SONAR-?RO"], "exclude": ["*LOW"]}
        )
    ] == ["agent-fast", "sonar-pro"]
    assert len(await aliases.list_models(filters={})) == 8
    assert await aliases.list_models(filters={"include": []}) == []
    assert await aliases.list_models(filters={"exclude": ["*"]}) == []
    assert len(await aliases.list_models(filters=None)) == 4
    assert aliases._client is canonical._client is None
