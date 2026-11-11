"""Unit tests for PerplexityProvider.

Tests cover:
- Message conversion for all roles
- Response parsing with citations
- Provider info and model listing
- Error handling
- SDK client lifecycle
"""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from decimal import Decimal
from unittest.mock import Mock

import httpx
import perplexity
import pytest
from amplifier_core import HookResult, MockCoordinator, ModuleCoordinator
from amplifier_core.llm_errors import LLMError
from amplifier_core.message_models import ChatResponse, Message, TextBlock

from amplifier_module_provider_perplexity import PerplexityProvider
from amplifier_module_provider_perplexity._agent_api import (
    convert_agent_response,
    convert_messages,
    resolve_route,
)


class TestMessageConversion:
    """Test message conversion from Amplifier format to Perplexity format."""

    def setup_method(self) -> None:
        """Set up test provider instance."""
        self.provider = PerplexityProvider(api_key="test-key")

    def test_convert_system_message(self) -> None:
        """R07: system role survives at its original input position."""
        messages = [Message(role="system", content="You are a helpful assistant.")]
        assert convert_messages(messages)[0].model_dump(exclude_unset=True) == {
            "type": "message",
            "role": "system",
            "content": "You are a helpful assistant.",
        }

    def test_convert_multiple_system_messages(self) -> None:
        """R07: multiple system messages must not be combined."""

        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="system", content="Be concise."),
        ]
        converted = convert_messages(messages)
        assert [item.model_dump(exclude_unset=True) for item in converted] == [
            {"type": "message", "role": "system", "content": "You are helpful."},
            {"type": "message", "role": "system", "content": "Be concise."},
        ]

    def test_convert_user_message(self) -> None:
        """User messages should pass through directly."""

        messages = [Message(role="user", content="Hello, how are you?")]
        assert convert_messages(messages)[0].model_dump(exclude_unset=True) == {
            "type": "message",
            "role": "user",
            "content": "Hello, how are you?",
        }

    def test_convert_assistant_message(self) -> None:
        """Assistant messages should pass through directly."""

        messages = [Message(role="assistant", content="I'm doing well, thank you!")]
        assert convert_messages(messages)[0].model_dump(exclude_unset=True) == {
            "type": "message",
            "role": "assistant",
            "content": "I'm doing well, thank you!",
        }

    def test_convert_developer_message(self) -> None:
        """R07: developer messages retain role without invented context tags."""

        messages = [Message(role="developer", content="This is context information.")]
        assert convert_messages(messages)[0].model_dump(exclude_unset=True) == {
            "type": "message",
            "role": "developer",
            "content": "This is context information.",
        }

    def test_convert_tool_message_skipped(self) -> None:
        """R07: unsupported tool history is rejected, never silently skipped."""

        messages = [Message(role="tool", content="Tool result")]
        with pytest.raises(ValueError, match="role"):
            convert_messages(messages)

    def test_convert_mixed_messages(self) -> None:
        """Mixed message types should be handled correctly."""

        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]
        assert [
            item.model_dump(exclude_unset=True) for item in convert_messages(messages)
        ] == [
            {"type": "message", "role": "system", "content": "Be helpful."},
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "message", "role": "assistant", "content": "Hi there!"},
            {"type": "message", "role": "user", "content": "How are you?"},
        ]


class TestTextExtraction:
    """Test text content extraction from various formats."""

    def setup_method(self) -> None:
        """Set up test provider instance."""
        self.provider = PerplexityProvider(api_key="test-key")

    def test_extract_string_content(self) -> None:
        """String content should be returned as-is."""
        result = convert_messages([Message(role="user", content="Hello world")])
        assert result[0].model_dump(exclude_unset=True)["content"] == "Hello world"

    def test_extract_dict_text_blocks(self) -> None:
        """R07: Core validates dict text, arbitrary attribute fallbacks reject."""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        result = convert_messages(
            [Message.model_validate({"role": "user", "content": content})]
        )
        assert result[0].model_dump(exclude_unset=True)["content"] == "Hello\nWorld"

        class TextDict(dict[str, str]):
            text = "Attribute text"

        for kind in ("other", "text"):
            malformed = Message.model_construct(
                role="user", content=[TextDict(type=kind, text="Mapping text")]
            )
            with pytest.raises(ValueError, match="content"):
                convert_messages([malformed])

    def test_extract_textblock_objects(self) -> None:
        """TextBlock objects should be extracted."""

        content = [
            TextBlock(type="text", text="Hello"),
            TextBlock(type="text", text="World"),
        ]
        result = convert_messages([Message(role="user", content=[*content])])
        assert result[0].model_dump(exclude_unset=True)["content"] == "Hello\nWorld"

    def test_extract_empty_list(self) -> None:
        """Empty list should return empty string."""
        result = convert_messages([Message(role="user", content=[])])
        assert result[0].model_dump(exclude_unset=True)["content"] == ""


class TestSDKResponseConversion:
    """Test conversion of SDK responses to Amplifier format."""

    def setup_method(self) -> None:
        """Set up test provider instance."""
        self.provider = PerplexityProvider(api_key="test-key")

    def _create_mock_response(
        self,
        content: str = "Test response",
        model: str = "sonar",
        prompt_tokens: int = 10,
        completion_tokens: int = 20,
        finish_reason: str = "stop",
        citations: list | None = None,
        response_id: str = "test-id",
    ):
        """R10: genuine native SDK model; incomplete envelope is known enum drift."""
        from perplexity.generated.api import ResponseCreateResponse
        from pydantic import ValidationError

        payload = {
            "id": response_id,
            "object": "response",
            "created_at": 1,
            "model": model,
            "status": "completed",
            "error": None,
            "output": [
                {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": content,
                            "annotations": [
                                {"type": "url_citation", "url": url}
                                for url in (citations or [])
                            ],
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        validated = ResponseCreateResponse.model_validate(payload, strict=True)
        if finish_reason == "length":
            payload.update(
                status="incomplete", incomplete_details={"reason": "max_output_tokens"}
            )
            with pytest.raises(ValidationError):
                ResponseCreateResponse.model_validate(payload, strict=True)
            # Converter-only drift; native HTTP decode is independently covered.
            return validated.model_copy(
                update={
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                }
            )
        return validated

    def test_convert_basic_response(self) -> None:
        """Basic response should be converted correctly."""
        mock_response = self._create_mock_response(
            content="The capital of France is Paris.",
            model="sonar",
            prompt_tokens=10,
            completion_tokens=20,
        )

        response = convert_agent_response(mock_response, resolve_route("sonar"))

        assert response.metadata is not None
        assert response.metadata["perplexity:model"] == "sonar"
        block = response.content[0]
        assert isinstance(block, TextBlock)
        assert "The capital of France is Paris." in block.text
        assert response.usage is not None
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20

    def test_convert_response_with_citations(self) -> None:
        """Response with citations should include them in content."""
        mock_response = self._create_mock_response(
            content="Paris is the capital.",
            citations=["https://example.com/paris", "https://wiki.org/france"],
        )

        response = convert_agent_response(mock_response, resolve_route("sonar"))

        block = response.content[0]
        assert isinstance(block, TextBlock)
        assert "Paris is the capital." in block.text
        assert block.text == "Paris is the capital."
        assert response.metadata is not None
        assert response.metadata["perplexity:citations"] == [
            "https://example.com/paris",
            "https://wiki.org/france",
        ]

    def test_convert_response_stop_reason_length(self) -> None:
        """Length finish reason should map to max_tokens."""
        mock_response = self._create_mock_response(finish_reason="length")

        response = convert_agent_response(mock_response, resolve_route("sonar"))

        assert response.finish_reason == "length"

    def test_convert_response_stop_reason_stop(self) -> None:
        """Stop finish reason should map to end_turn."""
        mock_response = self._create_mock_response(finish_reason="stop")

        response = convert_agent_response(mock_response, resolve_route("sonar"))

        assert response.finish_reason == "stop"

    def test_convert_empty_choices(self) -> None:
        """R10: completed empty native output has no invented answer or usage."""
        from perplexity.generated.api import ResponseCreateResponse

        payload = TestAgentResponseContract._payload()
        payload.update(output=[], usage=None)
        native = ResponseCreateResponse.model_validate(payload, strict=True)
        response = convert_agent_response(native, resolve_route("sonar"))

        block = response.content[0]
        assert isinstance(block, TextBlock)
        assert block.text == ""
        assert response.finish_reason == "stop"
        assert response.usage is None


class TestProviderInfo:
    """Test provider metadata and configuration."""

    def setup_method(self) -> None:
        """Set up test provider instance."""
        self.provider = PerplexityProvider(api_key="test-key")

    def test_get_info_returns_valid_provider_info(self) -> None:
        """get_info() should return valid ProviderInfo."""
        info = self.provider.get_info()

        assert info.id == "perplexity"
        assert info.display_name == "Perplexity AI"
        assert "PERPLEXITY_API_KEY" in info.credential_env_vars
        assert info.capabilities == ["citations", "web_search"]

    def test_get_info_has_config_fields(self) -> None:
        """get_info() should include configuration fields."""
        info = self.provider.get_info()

        field_ids = [f.id for f in info.config_fields]
        assert field_ids == [
            "api_key",
            "search_recency_filter",
            "search_type",
        ]
        assert info.defaults == {
            "default_model": "agent-fast",
            "max_tokens": 4096,
            "max_steps": 1,
            "timeout": None,
            "max_retries": 2,
        }

    @pytest.mark.parametrize("option", ["return_images", "return_related_questions"])
    def test_response_option_config_fields_are_optional_booleans(
        self, option: str
    ) -> None:
        info = self.provider.get_info()
        fields = {field.id: field for field in info.config_fields}
        # R08: no boolean CLI prompts for unsupported native response options.
        assert option not in fields
        assert fields["search_type"].choices == ["inherit", "fast", "web"]
        assert fields["search_type"].default == "inherit"

    def test_api_key_field_is_secret(self) -> None:
        """API key config field should be marked as secret."""
        info = self.provider.get_info()

        api_key_field = next(f for f in info.config_fields if f.id == "api_key")
        assert api_key_field.field_type == "secret"


class TestModelListing:
    """Test model listing functionality."""

    def setup_method(self) -> None:
        """Set up test provider instance."""
        self.provider = PerplexityProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_list_models_returns_expected_models(self) -> None:
        """No-argument listing recommends the four canonical Agent selectors."""
        models = await self.provider.list_models()
        model_ids = [m.id for m in models]

        assert model_ids == ["agent-fast", "agent-low", "agent-medium", "agent-high"]

    @pytest.mark.asyncio
    async def test_sonar_model_capabilities(self) -> None:
        """Sonar model should have correct capabilities."""
        models = await self.provider.list_models(filters={})
        sonar = next(m for m in models if m.id == "sonar")

        assert sonar.context_window == 0
        assert sonar.max_output_tokens == 0
        assert sonar.defaults == {} and sonar.pricing is None
        assert "web_search" in sonar.capabilities
        assert "citations" in sonar.capabilities
        assert "streaming" not in sonar.capabilities

    @pytest.mark.asyncio
    async def test_sonar_pro_model_capabilities(self) -> None:
        """Sonar Pro model should have enhanced capabilities."""
        models = await self.provider.list_models(filters={})
        sonar_pro = next(m for m in models if m.id == "sonar-pro")

        assert sonar_pro.context_window == 0
        assert sonar_pro.max_output_tokens == 0
        assert sonar_pro.capabilities == ["web_search", "citations"]
        assert sonar_pro.display_name == "Sonar Pro (Agent Fast alias)"

    @pytest.mark.asyncio
    async def test_retired_models_are_not_in_unfiltered_catalog(self) -> None:
        """Retired recommendations are removed, not merely hidden by defaults."""
        models = await self.provider.list_models(filters={})
        model_ids = [model.id for model in models]
        assert "r1-1776" not in model_ids
        assert "sonar-reasoning" not in model_ids


class TestToolCalls:
    """Test tool call handling."""

    def setup_method(self) -> None:
        """Set up test provider instance."""
        self.provider = PerplexityProvider(api_key="test-key")

    def test_parse_tool_calls_returns_empty_list(self) -> None:
        """parse_tool_calls() should always return empty list."""

        result = self.provider.parse_tool_calls(ChatResponse(content=[]))
        assert result == []


class TestClientLifecycle:
    """Test SDK client initialization and cleanup."""

    def test_client_not_initialized_on_construction(self) -> None:
        """Client should not be initialized until accessed."""
        provider = PerplexityProvider(api_key="test-key")
        assert provider._client is None

    def test_client_requires_api_key(self) -> None:
        """Accessing client without API key should raise ValueError."""
        provider = PerplexityProvider(api_key=None)
        with pytest.raises(ValueError, match="api_key must be provided"):
            _ = provider.client

    @pytest.mark.asyncio
    async def test_client_lazy_initialization(self) -> None:
        """Client should be initialized on first access."""
        provider = PerplexityProvider(api_key="test-key")
        try:
            client = provider.client
            assert client is not None
            assert provider._client is not None
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_client_reused_on_subsequent_access(self) -> None:
        """Same client instance should be returned on subsequent access."""
        provider = PerplexityProvider(api_key="test-key")
        try:
            client1 = provider.client
            client2 = provider.client
            assert client1 is client2
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_close_method(self) -> None:
        """Close method should cleanup client."""
        provider = PerplexityProvider(api_key="test-key")
        _ = provider.client  # Initialize
        assert provider._client is not None

        await provider.close()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Context manager should cleanup on exit."""
        async with PerplexityProvider(api_key="test-key") as provider:
            _ = provider.client
            assert provider._client is not None

        assert provider._client is None


class TestConfiguration:
    """Test provider configuration."""

    def test_default_configuration(self) -> None:
        """Provider should use sensible defaults."""
        provider = PerplexityProvider(api_key="test-key")

        assert provider.default_model == "agent-fast"
        assert provider.max_tokens == 4096
        assert provider.temperature is None
        assert provider.timeout is None
        assert provider.max_retries == 2

    def test_custom_configuration(self) -> None:
        """Provider should accept custom configuration."""
        config = {
            "default_model": "sonar-pro",
            "max_tokens": 8192,
            "temperature": 0.5,
            "timeout": 120.0,
            "max_retries": 5,
            "search_recency_filter": "week",
            "search_context_size": "high",
        }
        provider = PerplexityProvider(api_key="test-key", config=config)

        assert provider.default_model == "sonar-pro"
        assert provider.max_tokens == 8192
        assert provider.temperature == 0.5
        assert provider.timeout == 120.0
        assert provider.max_retries == 5
        assert provider.search_recency_filter == "week"
        assert provider.search_context_size == "high"

    @pytest.mark.parametrize("option", ["debug", "raw"])
    def test_observation_flags_default_false(self, option: str) -> None:
        """Omitting observation opt-ins must never enable them."""
        provider = PerplexityProvider()
        assert getattr(provider, option) is False
        assert provider._client is None

    @pytest.mark.parametrize("option", ["debug", "raw"])
    @pytest.mark.parametrize("value", [False, True], ids=["off", "on"])
    def test_observation_flags_accept_actual_booleans(
        self, option: str, value: bool
    ) -> None:
        provider = PerplexityProvider(config={option: value})
        assert getattr(provider, option) is value
        assert provider._client is None

    @pytest.mark.parametrize("option", ["debug", "raw"])
    @pytest.mark.parametrize(
        "value",
        [
            None,
            0,
            1,
            "",
            "false",
            "true",
            "SENSITIVE_SYNTHETIC_FLAG_CANARY",
            [],
            {},
            (),
            {"sensitive_field": "SENSITIVE_SYNTHETIC_FLAG_CANARY"},
        ],
        ids=[
            "null",
            "zero",
            "one",
            "empty-string",
            "false-string",
            "true-string",
            "sensitive-string",
            "empty-list",
            "empty-map",
            "empty-tuple",
            "sensitive-map",
        ],
    )
    def test_observation_flags_reject_non_booleans_before_side_effects(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        option: str,
        value: object,
    ) -> None:
        """Reject coercion and sensitive values at construction, not dispatch."""
        import amplifier_module_provider_perplexity as module

        caplog.set_level("DEBUG", logger=module.__name__)
        coordinator = MockCoordinator()
        events: list[str] = []

        async def observe(event: str, _data: dict[str, object]) -> HookResult:
            events.append(event)
            return HookResult()

        for event in ("llm:request", "llm:response", "provider:error"):
            coordinator.hooks.register(event, observe)
        constructor = Mock(side_effect=AssertionError("Unexpected SDK construction"))
        monkeypatch.setattr(module, "AsyncPerplexity", constructor)
        with pytest.raises(ValueError) as caught:
            PerplexityProvider(config={option: value}, coordinator=coordinator)
        assert str(caught.value) == f"{option} must be a boolean"
        assert constructor.call_count == 0
        assert events == []
        assert not [
            record for record in caplog.records if record.name == module.__name__
        ]


class TestMount:
    """Test the mount() function."""

    @pytest.mark.asyncio
    async def test_mount_with_api_key_env(self, monkeypatch) -> None:
        """Mount should work with API key from environment."""
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key-from-env")

        from amplifier_module_provider_perplexity import mount

        coordinator = MockCoordinator()
        cleanup = await mount(coordinator, {})

        assert cleanup is not None
        try:
            # Verify provider was mounted
            assert len(coordinator.mount_history) == 1
            assert coordinator.mount_history[0]["mount_point"] == "providers"
            assert coordinator.mount_history[0]["name"] == "perplexity"
        finally:
            await cleanup()

    @pytest.mark.asyncio
    async def test_mount_without_api_key_returns_none(self, monkeypatch) -> None:
        """Mount should return None if no API key available."""
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

        from amplifier_module_provider_perplexity import mount

        coordinator = MockCoordinator()
        result = await mount(coordinator, {})

        assert result is None
        assert coordinator.mount_history == []


class TestAgentRequestContract:
    """Native request boundary, independent of the unchanged active provider."""

    @staticmethod
    def _require_helper() -> None:
        """Qualify public aliases before any helper runs, including the RED run."""
        from importlib.util import find_spec

        from perplexity.generated import api

        assert api.InputItem is api.InputItemInput
        assert api.ResponseCreateParams is api.ResponsesRequestInput
        assert api.ResponseCreateResponse is api.ResponsesResponseOutput
        assert (
            find_spec("amplifier_module_provider_perplexity._agent_api") is not None
        ), "T1 pure request helper is not implemented"

    @pytest.mark.parametrize(
        ("selector", "effort", "model"),
        [
            ("agent-fast", "fast", None),
            ("agent-low", "low", None),
            ("agent-medium", "medium", None),
            ("agent-high", "high", None),
            ("sonar", "fast", None),
            ("sonar-pro", "fast", None),
            ("sonar-reasoning-pro", "low", None),
            ("sonar-deep-research", "high", None),
            ("vendor/Hidden-ID", None, "vendor/Hidden-ID"),
            ("AGENT-FAST", None, "AGENT-FAST"),
        ],
    )
    def test_route_table_and_exact_id(
        self, selector: str, effort: str | None, model: str | None
    ) -> None:
        """Catch catalog-gating, case-folding, wrong aliases, or dual routing."""
        from dataclasses import FrozenInstanceError, fields

        self._require_helper()
        from amplifier_core.message_models import ChatRequest

        from amplifier_module_provider_perplexity._agent_api import (
            build_agent_request,
            request_body,
            resolve_route,
        )

        route = resolve_route(selector)
        assert (route.selector, route.effort, route.model) == (selector, effort, model)
        for field in fields(route):
            with pytest.raises(FrozenInstanceError):
                setattr(route, field.name, "changed")
        built, params = build_agent_request(
            ChatRequest(messages=[Message(role="user", content="hello")]),
            {"model_filters": {"include": []}},
            {"model": selector},
        )
        assert built == route
        body = request_body(params)
        if effort is not None:
            assert body["effort"] == effort
            assert "model" not in body
        else:
            assert body["model"] == model
            assert "effort" not in body
        for invalid in (
            "",
            " ",
            " agent-fast",
            "agent-fast\n",
            "*",
            "vendor/?",
            "[abc]",
        ):
            with pytest.raises(ValueError) as error:
                resolve_route(
                    invalid + "CANARY" if invalid in ("*", "vendor/?") else invalid
                )
            assert "CANARY" not in str(error.value)

    def test_input_roles_order_and_empty_text(self) -> None:
        """Catch system extraction, developer wrapping, reordering, or empty loss."""
        self._require_helper()
        from amplifier_core.message_models import ChatRequest

        from amplifier_module_provider_perplexity._agent_api import (
            build_agent_request,
            convert_messages,
            request_body,
        )

        messages = [
            Message(role="system", content=" \nS\u00e9cret\t"),
            Message(role="user", content=""),
            Message(role="developer", content=[]),
            Message(role="system", content="second"),
            Message.model_validate(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "one"},
                        {"type": "text", "text": ""},
                        {"type": "text", "text": "three"},
                    ],
                }
            ),
        ]
        converted = convert_messages(messages)
        assert all(not isinstance(item, dict) for item in converted)
        _, params = build_agent_request(ChatRequest(messages=messages), {}, {})
        assert request_body(params)["input"] == [
            {"type": "message", "role": "system", "content": " \nS\u00e9cret\t"},
            {"type": "message", "role": "user", "content": ""},
            {"type": "message", "role": "developer", "content": ""},
            {"type": "message", "role": "system", "content": "second"},
            {"type": "message", "role": "assistant", "content": "one\n\nthree"},
        ]
        assert messages[0].content == " \nS\u00e9cret\t"
        with pytest.raises(ValueError, match="messages"):
            convert_messages([])

    def test_unsupported_history_and_reasoning_reject(self) -> None:
        """Catch silently dropped media/tools, arbitrary .text, or masked reasoning."""
        from amplifier_core.message_models import ChatRequest

        self._require_helper()
        from amplifier_module_provider_perplexity._agent_api import (
            build_agent_request,
            convert_messages,
        )

        for role in ("tool", "function"):
            with pytest.raises(ValueError, match="role"):
                convert_messages(
                    [Message.model_validate({"role": role, "content": ""})]
                )
        for block in (
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.org/CANARY"},
            },
            {"type": "thinking", "thinking": "CANARY"},
            {"type": "tool_call", "id": "one", "name": "CANARY", "input": {}},
            {"type": "tool_result", "tool_call_id": "one", "output": "CANARY"},
            {"type": "reasoning", "content": [], "summary": ["CANARY"]},
        ):
            message = Message.model_validate(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "valid"}, block],
                }
            )
            with pytest.raises(ValueError) as error:
                convert_messages([message])
            assert "CANARY" not in str(error.value)

        class ArbitraryText:
            text = "CANARY"

        # model_construct deliberately tests a malformed host object, not Core coercion.
        malformed = Message.model_construct(role="user", content=[ArbitraryText()])
        with pytest.raises(ValueError, match="content"):
            convert_messages([malformed])

        for core in ("none", "low", "CANARY"):
            request = ChatRequest(
                messages=[Message(role="user", content="CANARY")], reasoning_effort=core
            )
            for options in ({}, {"reasoning_effort": None}):
                with pytest.raises(ValueError, match="reasoning_effort") as error:
                    build_agent_request(request, {}, options)
                assert "CANARY" not in str(error.value)
        request = ChatRequest(messages=[Message(role="user", content="CANARY")])
        for source in ({"reasoning_effort": "none"}, {"reasoning_effort": "CANARY"}):
            for config, options in ((source, {}), ({}, source)):
                with pytest.raises(ValueError, match="reasoning_effort"):
                    build_agent_request(request, config, options)
        _, cleared = build_agent_request(
            request, {"reasoning_effort": "old"}, {"reasoning_effort": None}
        )
        assert "reasoning" not in cleared

    def test_option_precedence_and_zero_budget(self) -> None:
        """Catch truthiness defaults, wrong Core precedence, and provider-side merging."""
        from amplifier_core.message_models import ChatRequest

        self._require_helper()
        from amplifier_module_provider_perplexity._agent_api import build_agent_request

        messages = [Message(role="user", content="hello")]
        core = ChatRequest(
            messages=messages,
            model="agent-low",
            max_output_tokens=23,
            temperature=0.4,
            top_p=0.5,
        )
        config: dict[str, object] = {
            "default_model": "agent-high",
            "max_tokens": 31,
            "temperature": 0.6,
            "top_p": 0.7,
        }
        mapped: dict[str, object] = {
            "model": "agent-medium",
            "max_tokens": 17,
            "temperature": 0.8,
            "top_p": 0.9,
        }
        direct: dict[str, object] = {
            "model": "vendor/Hidden-ID",
            "max_tokens": 11,
            "temperature": 0,
            "top_p": 0,
        }
        for options, want in (
            ({}, ("agent-low", 23, 0.4, 0.5)),
            (mapped, ("agent-medium", 17, 0.8, 0.9)),
            ({**mapped, **direct}, ("vendor/Hidden-ID", 11, 0.0, 0.0)),
            ({**mapped, **dict.fromkeys(direct)}, ("agent-low", 23, 0.4, 0.5)),
        ):
            route, params = build_agent_request(core, config, options)
            assert "temperature" in params and "top_p" in params
            assert (
                route.selector,
                params["max_output_tokens"],
                params["temperature"],
                params["top_p"],
            ) == want
        route, params = build_agent_request(ChatRequest(messages=messages), config, {})
        assert "temperature" in params and "top_p" in params
        assert (
            route.selector,
            params["max_output_tokens"],
            params["temperature"],
            params["top_p"],
        ) == ("agent-high", 31, 0.6, 0.7)
        route, params = build_agent_request(ChatRequest(messages=messages), {}, {})
        assert (route.selector, params["max_output_tokens"], params["max_steps"]) == (
            "agent-fast",
            4096,
            1,
        )
        assert "temperature" not in params and "top_p" not in params
        for value in (0, -1, True, 1.5, "10", None):
            # None falls through to a deliberately invalid configured zero budget.
            with pytest.raises(ValueError, match="max_tokens"):
                build_agent_request(
                    core.model_copy(update={"max_output_tokens": None}),
                    {"max_tokens": 0},
                    {"max_tokens": value},
                )
        for value in (0, -1, True):
            with pytest.raises(ValueError, match="max_tokens"):
                build_agent_request(
                    core.model_copy(update={"max_output_tokens": value}), {}, {}
                )

    def test_web_search_modes_and_filter_copy(self) -> None:
        """Catch Sonar-shaped fields, aliasing, ignored disabled search, or clear loss."""
        from copy import deepcopy

        from amplifier_core.message_models import ChatRequest

        self._require_helper()
        from amplifier_module_provider_perplexity._agent_api import (
            build_agent_request,
            request_body,
        )

        request = ChatRequest(messages=[Message(role="user", content="hello")])
        domains = ["example.org", "docs.example.org"]
        config: dict[str, object] = {
            "search_domain_filter": domains,
            "search_recency_filter": "week",
            "search_type": "fast",
            "search_max_tokens": 17,
            "search_max_tokens_per_page": 9,
        }
        before = deepcopy(config)
        _, params = build_agent_request(request, config, {})
        assert config == before
        assert request_body(params)["tools"] == [
            {
                "type": "web_search",
                "filters": {
                    "search_domain_filter": domains,
                    "search_recency_filter": "week",
                },
                "search_type": "fast",
                "max_tokens": 17,
                "max_tokens_per_page": 9,
            }
        ]
        domains.append("later.example.org")
        assert request_body(params)["tools"] == [
            {
                "type": "web_search",
                "filters": {
                    "search_domain_filter": ["example.org", "docs.example.org"],
                    "search_recency_filter": "week",
                },
                "search_type": "fast",
                "max_tokens": 17,
                "max_tokens_per_page": 9,
            }
        ]
        assert before["search_domain_filter"] == ["example.org", "docs.example.org"]
        for empty_domains in (None, []):
            for recency in (None, "none"):
                clears = dict.fromkeys(config)
                clears.update(
                    search_domain_filter=empty_domains, search_recency_filter=recency
                )
                _, cleared = build_agent_request(request, config, clears)
                assert request_body(cleared)["tools"] == [{"type": "web_search"}]
        _, inherited = build_agent_request(request, {"search_type": "inherit"}, {})
        assert request_body(inherited)["tools"] == [{"type": "web_search"}]
        for recency in ("hour", "day", "week", "month", "year"):
            _, selected = build_agent_request(
                request, {}, {"search_recency_filter": recency}
            )
            assert request_body(selected)["tools"] == [
                {"type": "web_search", "filters": {"search_recency_filter": recency}}
            ]
        _, excluded = build_agent_request(
            request, {}, {"search_domain_filter": ["-example.org"]}
        )
        assert request_body(excluded)["tools"] == [
            {
                "type": "web_search",
                "filters": {"search_domain_filter": ["-example.org"]},
            }
        ]
        for mode in ("web", "fast"):
            _, selected = build_agent_request(request, {}, {"search_type": mode})
            assert request_body(selected)["tools"] == [
                {"type": "web_search", "search_type": mode}
            ]
        _, disabled = build_agent_request(
            request, {}, {"model": "vendor/Hidden-ID", "web_search": False}
        )
        assert "tools" not in disabled
        for selector in (
            "agent-fast",
            "agent-low",
            "agent-medium",
            "agent-high",
            "sonar",
            "sonar-pro",
            "sonar-reasoning-pro",
            "sonar-deep-research",
        ):
            with pytest.raises(ValueError, match="web_search"):
                build_agent_request(
                    request, {}, {"model": selector, "web_search": False}
                )
        for key, value in config.items():
            with pytest.raises(ValueError, match="web_search"):
                build_agent_request(
                    request,
                    {},
                    {"model": "vendor/Hidden-ID", "web_search": False, key: value},
                )

    def test_native_params_and_raw_snapshot(self) -> None:
        """Catch mutable hook snapshots, timeout inheritance, or wire escape hatches."""
        import json
        from copy import deepcopy

        from amplifier_core.message_models import ChatRequest
        from perplexity import Timeout
        from perplexity.generated.api import ResponseCreateParams

        self._require_helper()
        from amplifier_module_provider_perplexity._agent_api import (
            build_agent_request,
            request_body,
        )

        request = ChatRequest(
            messages=[Message(role="user", content=[TextBlock(text="hello")])],
            timeout=777,
            conversation_id="host-only",
            metadata={"api_key": "CANARY"},
        )
        config: dict[str, object] = {"timeout": 9, "api_key": "CANARY", "raw": True}
        options: dict[str, object] = {
            "search_domain_filter": ["example.org"],
            "coordinator": object(),
            "cancellation": object(),
        }
        before = deepcopy(options["search_domain_filter"])
        _, params = build_agent_request(request, config, options)
        assert "timeout" not in params
        snapshot = request_body(params)
        assert snapshot == {
            "input": [{"type": "message", "role": "user", "content": "hello"}],
            "effort": "fast",
            "max_output_tokens": 4096,
            "max_steps": 1,
            "stream": False,
            "background": False,
            "store": False,
            "tools": [
                {
                    "type": "web_search",
                    "filters": {"search_domain_filter": ["example.org"]},
                }
            ],
        }
        native = ResponseCreateParams.model_validate(snapshot)
        assert (
            native.model_dump(mode="json", by_alias=True, exclude_unset=True)
            == snapshot
        )
        assert "CANARY" not in json.dumps(snapshot)
        expected = deepcopy(snapshot)
        # Mutate the raw hook view at every nested collection boundary.
        snapshot_input = snapshot["input"]
        assert isinstance(snapshot_input, list)
        snapshot_input[0]["content"] = "hook"
        snapshot_tools = snapshot["tools"]
        assert isinstance(snapshot_tools, list)
        snapshot_tools[0]["filters"]["search_domain_filter"].append("hook.example.org")
        snapshot_tools.append({"type": "image_search"})
        snapshot["max_output_tokens"] = 1
        assert request_body(params) == expected
        assert options["search_domain_filter"] == before
        assert config == {"timeout": 9, "api_key": "CANARY", "raw": True}
        assert request.messages[0].content == [TextBlock(text="hello")]
        for timeout in (
            None,
            0,
            3.5,
            Timeout(None, connect=0, read=0, write=None, pool=None),
        ):
            _, timed = build_agent_request(request, config, {"timeout": timeout})
            assert "timeout" in timed
            assert timed["timeout"] is timeout
            assert "timeout" not in request_body(timed)
        assert params["stream"] is params["background"] is params["store"] is False

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("max_steps", None),
            ("max_steps", 0),
            ("max_steps", 501),
            ("max_steps", True),
            ("max_steps", 1.5),
            ("max_steps", "2"),
            ("temperature", True),
            ("temperature", -0.01),
            ("temperature", 2.01),
            ("temperature", float("nan")),
            ("temperature", float("inf")),
            ("temperature", "1"),
            ("top_p", True),
            ("top_p", -0.01),
            ("top_p", 1.01),
            ("top_p", float("-inf")),
            ("top_p", float("nan")),
            ("web_search", None),
            ("web_search", 0),
            ("web_search", "false"),
            ("search_domain_filter", "CANARY"),
            ("search_domain_filter", [1]),
            ("search_domain_filter", [""]),
            ("search_domain_filter", [" "]),
            ("search_domain_filter", ["-"]),
            ("search_domain_filter", ["- "]),
            ("search_domain_filter", ["example.org", "-other.org"]),
            ("search_domain_filter", ["example.org"] * 21),
            ("search_recency_filter", "CANARY"),
            ("search_type", "inherit"),
            ("search_type", "CANARY"),
            ("search_max_tokens", 0),
            ("search_max_tokens", True),
            ("search_max_tokens", 1.5),
            ("search_max_tokens_per_page", -1),
            ("search_max_tokens_per_page", False),
            ("search_context_size", "medium"),
            ("return_images", True),
            ("return_images", 1),
            ("return_images", None),
            ("return_related_questions", True),
            ("return_related_questions", "false"),
            ("return_related_questions", None),
        ],
    )
    def test_invalid_effective_options_are_safe(self, key: str, value: object) -> None:
        """Catch permissive coercion and prompt/value leakage in validation errors."""
        from amplifier_core.message_models import ChatRequest

        self._require_helper()
        from amplifier_module_provider_perplexity._agent_api import build_agent_request

        request = ChatRequest(messages=[Message(role="user", content="CANARY")])
        with pytest.raises(ValueError) as error:
            build_agent_request(request, {}, {key: value})
        assert key in str(error.value)
        assert "CANARY" not in str(error.value)
        if key == "search_context_size":
            assert "search_max_tokens" in str(error.value)
        if key.startswith("return_") and value is True:
            assert "metadata" in str(error.value) or "unsupported" in str(error.value)

    def test_legacy_clears_and_boundaries(self) -> None:
        """Catch eager legacy validation and rejection of valid boundary values."""
        from amplifier_core.message_models import ChatRequest

        self._require_helper()
        from amplifier_module_provider_perplexity._agent_api import (
            build_agent_request,
            request_body,
        )

        request = ChatRequest(messages=[Message(role="user", content="hello")])
        config: dict[str, object] = {
            "return_images": True,
            "return_related_questions": True,
            "search_context_size": "high",
            "reasoning_effort": "low",
        }
        options: dict[str, object] = {
            "return_images": False,
            "return_related_questions": False,
            "search_context_size": None,
            "reasoning_effort": None,
        }
        _, cleared = build_agent_request(request, config, options)
        assert request_body(cleared)["tools"] == [{"type": "web_search"}]
        for max_steps, temperature, top_p in ((1, 0, 0), (500, 2, 1)):
            _, params = build_agent_request(
                request,
                {},
                {
                    "max_steps": max_steps,
                    "temperature": temperature,
                    "top_p": top_p,
                    "search_domain_filter": ["example.org"] * 20,
                    "search_max_tokens": 1,
                    "search_max_tokens_per_page": 1,
                },
            )
            assert "temperature" in params and "top_p" in params
            assert (params["max_steps"], params["temperature"], params["top_p"]) == (
                max_steps,
                temperature,
                top_p,
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("tools", [{"name": "custom", "parameters": {}}]),
            ("tool_choice", "auto"),
            ("stop", ["CANARY"]),
            ("response_format", {"type": "json"}),
            (
                "response_format",
                {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}},
            ),
            ("stream", True),
        ],
    )
    def test_unsupported_core_options(self, field: str, value: object) -> None:
        """Catch silent loss of explicit unsupported Core semantics."""
        from amplifier_core.message_models import ChatRequest

        self._require_helper()
        from amplifier_module_provider_perplexity._agent_api import build_agent_request

        request = ChatRequest.model_validate(
            {"messages": [{"role": "user", "content": "CANARY"}], field: value}
        )
        with pytest.raises(ValueError, match=field) as error:
            build_agent_request(request, {}, {})
        assert "CANARY" not in str(error.value)

    @pytest.mark.parametrize(
        "key",
        [
            "effort",
            "preset",
            "profile",
            "models",
            "previous_response_id",
            "instructions",
            "tools",
            "skills",
            "tool_choice",
            "background",
            "store",
            "extra_body",
            "extra_headers",
            "extra_query",
            "max_output_tokens",
        ],
    )
    def test_native_escape_hatches_reject(self, key: str) -> None:
        """Catch kwargs accidentally becoming an arbitrary SDK forwarding path."""
        from amplifier_core.message_models import ChatRequest

        self._require_helper()
        from amplifier_module_provider_perplexity._agent_api import build_agent_request

        request = ChatRequest(messages=[Message(role="user", content="CANARY")])
        for value in (None, False, "CANARY"):
            with pytest.raises(ValueError, match=key) as error:
                build_agent_request(request, {}, {key: value})
            assert "CANARY" not in str(error.value)
        if key == "max_output_tokens":
            assert "ChatRequest.max_output_tokens" in str(error.value)

    def test_inert_core_options_and_fixed_stream(self) -> None:
        """Catch forwarding of host-only fields or acceptance of streaming."""
        from amplifier_core.message_models import ChatRequest

        self._require_helper()
        from amplifier_module_provider_perplexity._agent_api import (
            build_agent_request,
            request_body,
        )

        for stream in (None, False):
            request = ChatRequest.model_validate(
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [],
                    "stop": [],
                    "stream": stream,
                    "response_format": {"type": "text"},
                }
            )
            _, params = build_agent_request(
                request, {}, {"stream": False, "unknown_host_state": "CANARY"}
            )
            assert "CANARY" not in str(request_body(params))
        with pytest.raises(ValueError, match="stream"):
            build_agent_request(request, {}, {"stream": True})


class TestAgentResponseContract:
    """Pure response contract; strict wire fixtures and labeled drift are separate."""

    @staticmethod
    def _require_converter() -> None:
        from perplexity.generated import api

        from amplifier_module_provider_perplexity import _agent_api

        assert api.ResponseCreateResponse is api.ResponsesResponseOutput
        assert "convert_agent_response" in vars(_agent_api), (
            "T2 pure response converter is not implemented"
        )

    @staticmethod
    def _message(
        content: list[dict[str, object]], *, status: str = "completed"
    ) -> dict[str, object]:
        return {
            "id": "fixture-message",
            "type": "message",
            "role": "assistant",
            "status": status,
            "content": content,
        }

    @classmethod
    def _payload(cls) -> dict[str, object]:
        return {
            "id": "fixture-response",
            "object": "response",
            "created_at": 1,
            "model": "vendor/returned-model",
            "status": "completed",
            "error": None,
            "output": [
                cls._message(
                    [{"type": "output_text", "text": "Alpha Beta.", "annotations": []}]
                )
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        }

    @classmethod
    def _convert(
        cls,
        payload: dict[str, object],
        *,
        boundary_fault: bool = False,
        selector: str = "sonar-pro",
    ) -> ChatResponse:
        from unittest.mock import patch

        from perplexity.generated.api import ResponseCreateResponse

        cls._require_converter()
        from amplifier_module_provider_perplexity._agent_api import (
            convert_agent_response,
            resolve_route,
        )

        if boundary_fault:
            # SDK model_construct itself coerces bool/numeric-string faults.
            # Inject the exact post-dump boundary on a genuine SDK instance instead.
            # This is ONLY converter fault injection, never strict/wire proof.
            response = ResponseCreateResponse.model_validate(
                cls._payload(), strict=True
            )
            with patch.object(response, "model_dump", return_value=payload) as dump:
                result = convert_agent_response(response, resolve_route(selector))
                dump.assert_called_once_with(
                    mode="json", exclude_unset=True, warnings=False
                )
                return result
        response = ResponseCreateResponse.model_validate(payload, strict=True)
        return convert_agent_response(response, resolve_route(selector))

    @classmethod
    async def _wire_convert(
        cls, payload: dict[str, object], *, schema_drift: bool = False
    ) -> ChatResponse:
        """Real released serializer/parser over local HTTPX, never mock create."""
        import json

        import httpx
        from perplexity import AsyncPerplexity
        from perplexity.generated.api import ResponseCreateResponse
        from pydantic import ValidationError

        cls._require_converter()
        from amplifier_module_provider_perplexity._agent_api import (
            convert_agent_response,
            resolve_route,
        )

        if schema_drift:
            with pytest.raises(ValidationError) as mismatch:
                ResponseCreateResponse.model_validate(payload, strict=True)
            # Controlled drift here is the envelope's status literal only.
            assert [error["loc"] for error in mismatch.value.errors()] == [("status",)]
            assert mismatch.value.errors()[0]["type"] == "literal_error"
        else:
            validated = ResponseCreateResponse.model_validate(payload, strict=True)
            assert type(validated) is ResponseCreateResponse
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            assert request.method == "POST"
            assert request.url.path == "/v1/responses"
            assert request.headers["authorization"] == "Bearer synthetic-t2-key"
            assert json.loads(request.content) == {
                "input": "fixture-input",
                "effort": "fast",
                "max_output_tokens": 16,
                "max_steps": 1,
                "stream": False,
                "background": False,
                "store": False,
            }
            return httpx.Response(200, json=payload)

        async with (
            httpx.AsyncClient(
                transport=httpx.MockTransport(handler), trust_env=False
            ) as http,
            AsyncPerplexity(
                api_key="synthetic-t2-key",
                base_url="https://fixture.invalid",
                http_client=http,
                max_retries=0,
            ) as client,
        ):
            native = await client.responses.create(
                input="fixture-input",
                effort="fast",
                max_output_tokens=16,
                max_steps=1,
                stream=False,
                background=False,
                store=False,
            )
            assert type(native) is ResponseCreateResponse
            decoded = native.model_dump(mode="json", exclude_unset=True, warnings=False)
            assert decoded["status"] == payload["status"]
            if "incomplete_details" in payload:
                assert decoded["incomplete_details"] == payload["incomplete_details"]
            result = convert_agent_response(native, resolve_route("sonar-pro"))
        assert len(requests) == 1
        assert http.is_closed
        assert client.is_closed()
        return result

    @staticmethod
    def _assert_safe_error(error: Exception, message: str) -> None:
        import traceback

        from amplifier_core.llm_errors import LLMError

        assert type(error) is LLMError
        assert str(error) == f"Perplexity Agent API {message}."
        assert error.provider == "perplexity"
        assert error.retryable is False
        assert error.status_code == 200
        assert error.model is None
        assert error.__cause__ is None
        assert error.__suppress_context__ is True
        assert not hasattr(error, "body")
        assert "CANARY" not in repr(error)
        assert "CANARY" not in "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    @pytest.mark.asyncio
    async def test_strict_native_wire_and_core_types(self) -> None:
        """Catch wrong SDK aliases/parser, choice adapters and fake Core output."""
        from typing import get_type_hints

        from amplifier_core.message_models import Usage
        from perplexity.generated.api import ResponseCreateResponse

        self._require_converter()
        from amplifier_module_provider_perplexity._agent_api import (
            AgentRoute,
            convert_agent_response,
        )

        result = await self._wire_convert(self._payload())
        assert get_type_hints(convert_agent_response) == {
            "response": ResponseCreateResponse,
            "route": AgentRoute,
            "return": ChatResponse,
        }
        assert type(result) is ChatResponse
        assert len(result.content) == 1 and type(result.content[0]) is TextBlock
        assert result.content[0].text == "Alpha Beta."
        assert result.finish_reason == "stop"
        assert type(result.usage) is Usage
        assert result.usage.input_tokens == 5
        assert result.usage.output_tokens == 3
        assert result.usage.total_tokens == 8
        assert result.usage.cost_usd is None
        assert not hasattr(result, "model")
        assert result.tool_calls is None
        assert result.metadata == {
            "perplexity:response_id": "fixture-response",
            "perplexity:model": "vendor/returned-model",
            "perplexity:requested_selector": "sonar-pro",
            "perplexity:resolved_effort": "fast",
            "perplexity:status": "completed",
            "perplexity:annotations": [
                {"output_index": 0, "content_index": 0, "annotations": []}
            ],
            "perplexity:citations": [],
            "perplexity:usage_cost_status": "missing",
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("details", "finish", "metadata"),
        [
            (
                {"reason": "max_output_tokens", "secret": "CANARY"},
                "length",
                {"reason": "max_output_tokens"},
            ),
            ({"reason": "future", "secret": "CANARY"}, None, {"reason": "future"}),
            ({}, None, None),
            (None, None, None),
            ({"reason": None}, None, None),
            ({"reason": 7}, None, None),
        ],
    )
    async def test_permissive_incomplete_enum_drift(
        self, details: object, finish: str | None, metadata: object
    ) -> None:
        """Catch truncation guesses; strict enum rejection precedes real decode."""
        payload = self._payload()
        payload.update(status="incomplete", incomplete_details=details)
        result = await self._wire_convert(payload, schema_drift=True)
        assert result.finish_reason == finish
        assert result.content == [TextBlock(text="Alpha Beta.")]
        assert result.metadata is not None
        if metadata is None:
            assert "perplexity:incomplete_details" not in result.metadata
        else:
            assert result.metadata["perplexity:incomplete_details"] == metadata
        assert result.metadata["perplexity:status"] == "incomplete"
        assert "CANARY" not in str(result.metadata)

    @pytest.mark.parametrize(
        ("status", "message"),
        [
            ("failed", "returned an error response"),
            ("cancelled", "response was cancelled"),
            ("queued", "returned an unsupported response state"),
            ("in_progress", "returned an unsupported response state"),
            ("requires_action", "returned an unsupported response state"),
        ],
    )
    def test_strict_error_states(self, status: str, message: str, caplog) -> None:
        """Catch treating terminal failure or unhandled server work as success."""
        from amplifier_core.llm_errors import LLMError

        payload = self._payload()
        payload["status"] = status
        with pytest.raises(LLMError) as error:
            self._convert(payload)
        self._assert_safe_error(error.value, message)
        assert caplog.records == []

    @pytest.mark.parametrize(
        "error_value", [{"code": "fixture", "message": "CANARY"}, {}, False, "", 0]
    )
    def test_error_overrides_completed_and_unknown(
        self, error_value: object, caplog
    ) -> None:
        """Even falsey non-null errors must preempt partial text and status checks."""
        from amplifier_core.llm_errors import LLMError

        for status in ("completed", "CANARY"):
            payload = self._payload()
            payload.update(error=error_value, status=status)
            with pytest.raises(LLMError) as error:
                self._convert(payload, boundary_fault=True)
            self._assert_safe_error(error.value, "returned an error response")
        assert caplog.records == []

    @pytest.mark.asyncio
    async def test_permissive_unknown_status_is_invalid(self) -> None:
        """Unknown runtime status must fail, not return partial success."""
        from amplifier_core.llm_errors import LLMError

        payload = self._payload()
        payload["status"] = "CANARY"
        with pytest.raises(LLMError) as error:
            await self._wire_convert(payload, schema_drift=True)
        self._assert_safe_error(error.value, "returned an invalid response")

    @pytest.mark.asyncio
    async def test_permissive_incomplete_without_details(self) -> None:
        payload = self._payload()
        payload["status"] = "incomplete"
        result = await self._wire_convert(payload, schema_drift=True)
        assert result.finish_reason is None
        assert result.metadata is not None
        assert "perplexity:incomplete_details" not in result.metadata

    @pytest.mark.asyncio
    async def test_strict_completed_error_is_not_an_answer(self, caplog) -> None:
        from amplifier_core.llm_errors import LLMError

        payload = self._payload()
        payload["error"] = {"code": "fixture", "message": "CANARY"}
        with pytest.raises(LLMError) as error:
            await self._wire_convert(payload)
        self._assert_safe_error(error.value, "returned an error response")
        assert caplog.records == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "total_tokens"])
    async def test_strict_sdk_negative_usage_rejected_by_converter(
        self, field: str
    ) -> None:
        """The current SDK schema admits negatives; Core adaptation must not."""
        from amplifier_core.llm_errors import LLMError

        payload = self._payload()
        payload["usage"] = {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
            field: -1,
        }
        with pytest.raises(LLMError) as error:
            await self._wire_convert(payload)
        self._assert_safe_error(error.value, "returned an invalid response")

    @pytest.mark.parametrize("dumped", [None, [], {2: "CANARY"}])
    def test_injected_non_object_dump_is_invalid(self, dumped: object) -> None:
        """Deliberate dump-boundary fault, not SDK wire conformance."""
        from unittest.mock import patch

        from amplifier_core.llm_errors import LLMError
        from perplexity.generated.api import ResponseCreateResponse

        self._require_converter()
        from amplifier_module_provider_perplexity._agent_api import (
            convert_agent_response,
            resolve_route,
        )

        response = ResponseCreateResponse.model_validate(self._payload(), strict=True)
        with (
            patch.object(response, "model_dump", return_value=dumped),
            pytest.raises(LLMError) as error,
        ):
            convert_agent_response(response, resolve_route("agent-fast"))
        self._assert_safe_error(error.value, "returned an invalid response")

    def test_injected_serialization_failure_suppresses_diagnostics(
        self, caplog
    ) -> None:
        """Unserializable future SDK data cannot leak serializer exception text."""
        from unittest.mock import patch

        from amplifier_core.llm_errors import LLMError
        from perplexity.generated.api import ResponseCreateResponse

        self._require_converter()
        from amplifier_module_provider_perplexity._agent_api import (
            convert_agent_response,
            resolve_route,
        )

        response = ResponseCreateResponse.model_validate(self._payload(), strict=True)
        with (
            patch.object(response, "model_dump", side_effect=ValueError("CANARY")),
            pytest.raises(LLMError) as error,
        ):
            convert_agent_response(response, resolve_route("agent-fast"))
        self._assert_safe_error(error.value, "returned an invalid response")
        assert caplog.records == []

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("status", None),
            ("status", "future"),
            ("status", 2),
            ("output", None),
            ("output", {}),
            ("output", [None]),
            ("output", [{}]),
            ("output", [{"type": 2}]),
            ("object", "CANARY"),
            ("id", 3),
            ("created_at", True),
            (
                "output",
                [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": None,
                    }
                ],
            ),
        ],
    )
    def test_constructed_malformed_envelopes(self, field: str, value: object) -> None:
        """Isolated converter faults, NOT strict or real-wire conformance."""
        from amplifier_core.llm_errors import LLMError

        payload = self._payload()
        payload[field] = value
        with pytest.raises(LLMError) as error:
            self._convert(payload, boundary_fault=True)
        self._assert_safe_error(error.value, "returned an invalid response")
        for missing in ("status", "id", "object", "created_at", "output"):
            missing_payload = self._payload()
            del missing_payload[missing]
            with pytest.raises(LLMError) as missing_error:
                self._convert(missing_payload, boundary_fault=True)
            self._assert_safe_error(missing_error.value, "returned an invalid response")

    @pytest.mark.parametrize(
        ("status", "message"),
        [
            ("failed", "returned an error response"),
            ("cancelled", "response was cancelled"),
            ("queued", "returned an unsupported response state"),
            ("in_progress", "returned an unsupported response state"),
            ("requires_action", "returned an unsupported response state"),
            ("future", "returned an invalid response"),
            (None, "returned an invalid response"),
        ],
    )
    def test_assistant_item_state_preempts_text(
        self, status: object, message: str
    ) -> None:
        """Completed envelope cannot launder unhandled/failed assistant work."""
        from amplifier_core.llm_errors import LLMError

        item = self._message([{"type": "output_text", "text": "CANARY"}])
        item["status"] = status
        payload = self._payload()
        payload["output"] = [item]
        with pytest.raises(LLMError) as error:
            self._convert(payload, boundary_fault=True)
        self._assert_safe_error(error.value, message)

    @pytest.mark.parametrize("kind", ["function_call", "mcp_approval_request"])
    def test_strict_client_tool_requests_are_unsupported(self, kind: str) -> None:
        """Native client actions are never silently executed or exposed as text."""
        from amplifier_core.llm_errors import LLMError

        item: dict[str, object] = {
            "id": "call",
            "type": kind,
            "name": "CANARY",
            "arguments": "{}",
        }
        if kind == "function_call":
            item.update(status="completed", call_id="call-id")
        else:
            item["server_label"] = "CANARY"
        payload = self._payload()
        payload["output"] = [item]
        with pytest.raises(LLMError) as error:
            self._convert(payload)
        self._assert_safe_error(error.value, "returned an unsupported response state")

    def test_constructed_mixed_content_ignores_decoys(self) -> None:
        """Future/media/reasoning .text is not an assistant output_text answer."""
        payload = self._payload()
        payload["output_text"] = "Alpha [7] Beta."
        payload["output"] = [
            {"type": "reasoning", "text": "CANARY"},
            self._message(
                [
                    {"type": "output_text", "text": "Alpha [7]", "annotations": []},
                    {"type": "image", "text": "CANARY"},
                    {"type": "output_text", "text": 12},
                    {"type": "output_text", "text": " "},
                ]
            ),
            {"type": "tool_trace", "text": "CANARY"},
            {
                **self._message([{"type": "output_text", "text": "CANARY"}]),
                "role": "user",
            },
            self._message([{"type": "output_text", "text": "Beta."}]),
        ]
        result = self._convert(payload, boundary_fault=True)
        assert result.content == [TextBlock(text="Alpha [7] Beta.")]
        block = result.content[0]
        assert isinstance(block, TextBlock)
        assert "Sources" not in block.text
        assert result.metadata is not None
        assert result.metadata["perplexity:citations"] == []
        assert result.metadata["perplexity:annotations"] == [
            {"output_index": 1, "content_index": 0, "annotations": []}
        ]

    @pytest.mark.parametrize(
        "output",
        [
            [],
            [{"type": "unknown", "item_name": "future", "payload": {"text": "CANARY"}}],
        ],
    )
    def test_strict_empty_answer_is_one_text_block(self, output: list[object]) -> None:
        payload = self._payload()
        payload["output"] = output
        result = self._convert(payload)
        assert result.content == [TextBlock(text="")]
        assert result.metadata is not None
        assert "perplexity:annotations" not in result.metadata
        assert "perplexity:citations" not in result.metadata

    @pytest.mark.asyncio
    async def test_strict_sources_annotations_and_whitelist(self) -> None:
        """Catch renumbering, citation invention, index shifting and extra leakage."""
        import json

        source7 = {
            "id": 7,
            "url": "https://example.org/retrieved",
            "title": "Seven",
            "snippet": "seven",
            "date": "2026-01-01",
            "last_updated": "2026-01-02",
            "source": "web",
        }
        source2 = {
            "id": 2,
            "url": "https://example.org/two",
            "title": "Two",
            "snippet": "two",
        }
        citation = {
            "type": "url_citation",
            "url": "https://example.org/cited",
            "title": "Cited",
            "start_index": 2,
            "end_index": 8,
        }
        other = {
            "type": "other",
            "url": "https://example.org/not-cited",
            "title": "Other",
            "start_index": 0,
            "end_index": 1,
        }
        payload = self._payload()
        payload.update(
            effort="fast-v2",
            images=["CANARY"],
            related_questions=["CANARY"],
            instructions="CANARY",
            metadata={"secret": "CANARY"},
            output=[
                {
                    "type": "search_results",
                    "queries": ["q"],
                    "results": [{**source7, "secret": "CANARY"}, source2],
                    "id": "search-a",
                    "call_id": "call-a",
                },
                self._message(
                    [
                        {
                            "type": "output_text",
                            "text": "Alpha [7]",
                            "annotations": [{**citation, "secret": "CANARY"}, other],
                        },
                        {"type": "output_text", "text": "[web:2]", "annotations": []},
                    ]
                ),
                {
                    "type": "search_results",
                    "queries": [],
                    "results": [source7],
                    "tool_call_id": "call-b",
                },
                self._message(
                    [{"type": "output_text", "text": "end", "annotations": [citation]}]
                ),
            ],
        )
        result = await self._wire_convert(payload)
        assert result.content == [TextBlock(text="Alpha [7][web:2]end")]
        assert result.metadata == {
            "perplexity:response_id": "fixture-response",
            "perplexity:model": "vendor/returned-model",
            "perplexity:requested_selector": "sonar-pro",
            "perplexity:resolved_effort": "fast",
            "perplexity:effort": "fast-v2",
            "perplexity:status": "completed",
            "perplexity:search_results": [source7, source2, source7],
            "perplexity:search_result_items": [
                {
                    "output_index": 0,
                    "type": "search_results",
                    "queries": ["q"],
                    "results": [source7, source2],
                    "id": "search-a",
                    "call_id": "call-a",
                },
                {
                    "output_index": 2,
                    "type": "search_results",
                    "queries": [],
                    "results": [source7],
                    "tool_call_id": "call-b",
                },
            ],
            "perplexity:annotations": [
                {
                    "output_index": 1,
                    "content_index": 0,
                    "annotations": [citation, other],
                },
                {"output_index": 1, "content_index": 1, "annotations": []},
                {"output_index": 3, "content_index": 0, "annotations": [citation]},
            ],
            "perplexity:citations": [
                "https://example.org/cited",
                "https://example.org/cited",
            ],
            "perplexity:usage_cost_status": "missing",
        }
        assert "CANARY" not in json.dumps(result.metadata, allow_nan=False)

        json.dumps(result.model_dump(mode="json"), allow_nan=False)

    @pytest.mark.parametrize("arrays", ["absent", None, []])
    def test_constructed_optional_arrays(self, arrays: object) -> None:
        """Keep explicit empties, but omit absent/null annotation/result arrays."""
        part: dict[str, object] = {"type": "output_text", "text": "Inline [7][web:2]"}
        search: dict[str, object] = {"type": "search_results"}
        if arrays != "absent":
            part["annotations"] = arrays
            search["results"] = arrays
            search["queries"] = arrays
        payload = self._payload()
        payload["output"] = [search, self._message([part])]
        result = self._convert(payload, boundary_fault=True)
        assert result.metadata is not None
        if arrays == []:
            assert result.metadata["perplexity:search_results"] == []
            assert result.metadata["perplexity:search_result_items"] == [
                {
                    "output_index": 0,
                    "type": "search_results",
                    "queries": [],
                    "results": [],
                }
            ]
            assert result.metadata["perplexity:annotations"] == [
                {"output_index": 1, "content_index": 0, "annotations": []}
            ]
            assert result.metadata["perplexity:citations"] == []
        else:
            for key in (
                "search_results",
                "search_result_items",
                "annotations",
                "citations",
            ):
                assert f"perplexity:{key}" not in result.metadata

    @pytest.mark.parametrize("model", [None, "", "  ", "vendor/Actual"])
    @pytest.mark.parametrize("selector", ["agent-low", "vendor/Hidden-ID"])
    def test_constructed_actual_model_never_falls_back(
        self, model: object, selector: str
    ) -> None:
        payload = self._payload()
        payload["model"] = model
        result = self._convert(payload, boundary_fault=True, selector=selector)
        assert result.metadata is not None
        assert result.metadata["perplexity:model"] == (
            "vendor/Actual" if model == "vendor/Actual" else None
        )
        assert result.metadata["perplexity:requested_selector"] == selector
        if selector == "agent-low":
            assert result.metadata["perplexity:resolved_effort"] == "low"
        else:
            assert "perplexity:resolved_effort" not in result.metadata
        assert "perplexity:effort" not in result.metadata

    @pytest.mark.parametrize("usage", ["absent", None])
    def test_strict_usage_absent_or_null(self, usage: object) -> None:
        payload = self._payload()
        if usage == "absent":
            del payload["usage"]
        else:
            payload["usage"] = usage
        result = self._convert(payload)
        assert result.usage is None
        assert result.metadata is not None
        assert result.metadata["perplexity:usage_cost_status"] == "missing"

    @pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "total_tokens"])
    @pytest.mark.parametrize("bad", [True, -1, 1.5, "3", None, "absent"])
    def test_constructed_invalid_required_usage(self, field: str, bad: object) -> None:
        """No SDK coercion hides converter-only malformed usage faults."""
        from amplifier_core.llm_errors import LLMError

        usage: dict[str, object] = {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
        }
        if bad == "absent":
            del usage[field]
        else:
            usage[field] = bad
        payload = self._payload()
        payload["usage"] = usage
        with pytest.raises(LLMError) as error:
            self._convert(payload, boundary_fault=True)
        self._assert_safe_error(error.value, "returned an invalid response")

    @pytest.mark.parametrize("usage", [False, [], "CANARY"])
    def test_constructed_usage_must_be_mapping(self, usage: object) -> None:
        from amplifier_core.llm_errors import LLMError

        payload = self._payload()
        payload["usage"] = usage
        with pytest.raises(LLMError) as error:
            self._convert(payload, boundary_fault=True)
        self._assert_safe_error(error.value, "returned an invalid response")

    @pytest.mark.parametrize("optional", [0, 4, None, True, -1, 1.5, "2"])
    def test_constructed_optional_counts_and_native_total(
        self, optional: object
    ) -> None:
        payload = self._payload()
        payload["usage"] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 17,
            "input_tokens_details": {
                "cache_read_input_tokens": optional,
                "cache_creation_input_tokens": optional,
            },
            "output_tokens_details": {"reasoning_tokens": optional},
        }
        result = self._convert(payload, boundary_fault=True)
        assert result.usage is not None
        assert (
            result.usage.input_tokens,
            result.usage.output_tokens,
            result.usage.total_tokens,
        ) == (0, 0, 17)
        want = optional if type(optional) is int and optional >= 0 else None
        assert result.usage.cache_read_tokens == want
        assert result.usage.cache_write_tokens == want
        assert result.usage.reasoning_tokens == want

    @pytest.mark.asyncio
    async def test_strict_usage_cost_decimal_and_extra_reasoning(self) -> None:
        """Use SDK-exposed decimal spelling, not float->Decimal binary artifacts."""
        from decimal import Decimal

        cost = {
            "currency": "USD",
            "input_cost": 0.01,
            "output_cost": 0.02,
            "total_cost": 0.047,
            "cache_creation_cost": 0.0,
            "cache_read_cost": 0.0,
            "tool_calls_cost": 0.017,
        }
        payload = self._payload()
        payload["usage"] = {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 80,
            "input_tokens_details": {
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 2,
            },
            "output_tokens_details": {"reasoning_tokens": 0},
            "cost": {**cost, "secret": "CANARY"},
        }
        result = await self._wire_convert(payload)
        assert result.usage is not None
        assert result.usage.total_tokens == 80
        assert result.usage.cache_read_tokens == 0
        assert result.usage.cache_write_tokens == 2
        assert result.usage.reasoning_tokens == 0
        assert type(result.usage.cost_usd) is Decimal
        assert result.usage.cost_usd == Decimal("0.047")
        assert result.usage.model_dump(mode="json")["cost_usd"] == "0.047"
        assert result.metadata is not None
        assert result.metadata["perplexity:usage_cost"] == cost
        assert result.metadata["perplexity:usage_cost_status"] == "known"

    @pytest.mark.parametrize(
        ("cost", "status", "value"),
        [
            (None, "missing", None),
            ("absent", "missing", None),
            ({}, "invalid", None),
            (False, "invalid", None),
            ({"currency": "USD", "total_cost": 0}, "known", "0"),
            ({"currency": "USD", "total_cost": 0.047}, "known", "0.047"),
            ({"currency": "USD", "total_cost": True}, "invalid", None),
            ({"currency": "USD", "total_cost": -1}, "invalid", None),
            ({"currency": "USD", "total_cost": float("nan")}, "invalid", None),
            ({"currency": "USD", "total_cost": float("inf")}, "invalid", None),
            ({"currency": "USD", "total_cost": float("-inf")}, "invalid", None),
            ({"currency": "USD", "total_cost": "0.047"}, "invalid", None),
            ({"currency": "USD", "total_cost": None}, "invalid", None),
            ({"currency": "EUR", "total_cost": 1}, "non_usd", None),
            ({"currency": "EUR", "total_cost": -1}, "invalid", None),
            ({"currency": "", "total_cost": 1}, "invalid", None),
            ({"total_cost": 1}, "invalid", None),
        ],
    )
    def test_constructed_cost_boundary_faults(
        self, cost: object, status: str, value: str | None
    ) -> None:
        """Malformed/partial cost is NOT strict SDK fixture evidence."""
        import json
        from decimal import Decimal

        payload = self._payload()
        usage: dict[str, object] = {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
        }
        if cost != "absent":
            usage["cost"] = cost
        payload["usage"] = usage
        result = self._convert(payload, boundary_fault=True)
        assert result.usage is not None
        assert result.usage.cost_usd == (Decimal(value) if value is not None else None)
        assert result.metadata is not None
        assert result.metadata["perplexity:usage_cost_status"] == status
        json.dumps(result.metadata, allow_nan=False)
        if cost in (None, "absent", False):
            assert "perplexity:usage_cost" not in result.metadata
        elif isinstance(cost, dict):
            kept = result.metadata["perplexity:usage_cost"]
            assert isinstance(kept, dict)
            if cost.get("total_cost") is True:
                assert kept == {"currency": "USD"}
            if status == "invalid" and cost.get("total_cost") not in (0, 1):
                assert "total_cost" not in kept

    def test_constructed_cost_allowlist_and_absent_optional_counts(self) -> None:
        import json
        from decimal import Decimal

        payload = self._payload()
        payload["usage"] = {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
            "cost": {
                "currency": "USD",
                "total_cost": 0,
                "input_cost": True,
                "output_cost": float("nan"),
                "cache_read_cost": -1,
                "cache_creation_cost": {"secret": "CANARY"},
                "tool_calls_cost": 0,
                "secret": "CANARY",
                "raw": {},
            },
        }
        result = self._convert(payload, boundary_fault=True)
        assert result.usage is not None
        assert result.usage.cost_usd == Decimal(0)
        assert result.usage.cache_read_tokens is None
        assert result.usage.cache_write_tokens is None
        assert result.usage.reasoning_tokens is None
        assert result.metadata is not None
        assert result.metadata["perplexity:usage_cost"] == {
            "currency": "USD",
            "total_cost": 0,
            "tool_calls_cost": 0,
        }
        assert "CANARY" not in json.dumps(result.metadata, allow_nan=False)


class TestActiveAgentWire:
    """Exercise complete through the real native SDK, not a mocked create."""

    @staticmethod
    @asynccontextmanager
    async def _wire(
        *,
        payload: dict[str, object] | None = None,
        config: dict[str, object] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ) -> AsyncIterator[tuple[PerplexityProvider, list[httpx.Request]]]:
        from perplexity.generated.api import ResponseCreateResponse
        from pydantic import ValidationError

        body = payload if payload is not None else TestAgentResponseContract._payload()
        if body.get("status") == "incomplete":
            with pytest.raises(ValidationError) as caught:
                ResponseCreateResponse.model_validate(body, strict=True)
            assert [error["loc"] for error in caught.value.errors()] == [("status",)]
        else:
            ResponseCreateResponse.model_validate(body, strict=True)
        seen: list[httpx.Request] = []
        caller_task = asyncio.current_task()

        def handler(request: httpx.Request) -> httpx.Response:
            assert asyncio.current_task() is caller_task
            assert request.method == "POST"
            assert request.url.path == "/v1/responses"
            assert request.headers["authorization"] == "Bearer synthetic-t3a-key"
            seen.append(request)
            return httpx.Response(200, json=body)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ) as http:
            client = perplexity.AsyncPerplexity(
                api_key="synthetic-t3a-key", http_client=http, max_retries=0
            )
            provider = PerplexityProvider(
                api_key="synthetic-t3a-key", config=config, coordinator=coordinator
            )
            provider._client = client
            try:
                yield provider, seen
            finally:
                await provider.close()
                assert client.is_closed()
        assert http.is_closed

    @pytest.mark.asyncio
    async def test_native_endpoint_and_payload(self) -> None:
        import json

        import httpx
        import perplexity
        from amplifier_core.message_models import ChatRequest
        from perplexity.generated.api import ResponseCreateResponse

        payload = TestAgentResponseContract._payload()
        ResponseCreateResponse.model_validate(payload, strict=True)
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=payload)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ) as http:
            provider = PerplexityProvider(api_key="synthetic-t3a-key")
            provider._client = perplexity.AsyncPerplexity(
                api_key="synthetic-t3a-key", http_client=http, max_retries=0
            )
            try:
                result = await provider.complete(
                    ChatRequest(messages=[Message(role="user", content="offline")])
                )
            finally:
                await provider.close()
                assert len(seen) == 1
                assert seen[0].method == "POST"
                assert seen[0].url.path == "/v1/responses"
            assert json.loads(seen[0].content) == {
                "input": [{"type": "message", "role": "user", "content": "offline"}],
                "effort": "fast",
                "max_output_tokens": 4096,
                "max_steps": 1,
                "stream": False,
                "background": False,
                "store": False,
                "tools": [{"type": "web_search"}],
            }
            assert type(result) is ChatResponse
            assert result.content == [TextBlock(text="Alpha Beta.")]
            assert result.usage is not None and result.usage.total_tokens == 8
        assert http.is_closed

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "selector,effort",
        [
            ("agent-fast", "fast"),
            ("agent-low", "low"),
            ("agent-medium", "medium"),
            ("agent-high", "high"),
            ("sonar", "fast"),
            ("sonar-pro", "fast"),
            ("sonar-reasoning-pro", "low"),
            ("sonar-deep-research", "high"),
            ("vendor/Hidden-ID", None),
            ("AGENT-FAST", None),
        ],
    )
    async def test_routes_are_independent_of_listing(
        self, selector: str, effort: str | None
    ) -> None:
        from amplifier_core.message_models import ChatRequest

        async with self._wire(config={"model_filters": {"include": []}}) as (
            provider,
            seen,
        ):
            assert await provider.list_models() == []
            result = await provider.complete(
                ChatRequest(messages=[Message(role="user", content="")], model=selector)
            )
        body = json.loads(seen[0].content)
        if effort is not None:
            assert body["effort"] == effort and "model" not in body
        else:
            assert body["model"] == selector and "effort" not in body
        assert result.metadata is not None
        assert result.metadata["perplexity:model"] == "vendor/returned-model"
        assert result.metadata["perplexity:requested_selector"] == selector
        assert result.metadata.get("perplexity:resolved_effort") == effort

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "selector",
        ["", " agent-fast", "agent-fast\n", "*CANARY", "vendor/?CANARY", "[CANARY]"],
    )
    async def test_invalid_selectors_precede_hooks_and_client(
        self, selector: str
    ) -> None:
        from amplifier_core.message_models import ChatRequest

        coordinator = MockCoordinator()
        events = []

        async def observe(event, data):
            events.append((event, data))
            return HookResult()

        coordinator.hooks.register("llm:request", observe)
        provider = PerplexityProvider(coordinator=coordinator)
        with pytest.raises(ValueError) as caught:
            await provider.complete(
                ChatRequest(
                    messages=[Message(role="user", content="CANARY")], model=selector
                )
            )
        assert "CANARY" not in str(caught.value)
        assert provider._client is None and events == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "options",
        [
            {"request_options": []},
            {"request_options": "CANARY"},
            {"max_tokens": 0},
            {"tools": None},
            {"extra_body": {}},
            {"stream": True},
            {"search_context_size": "high"},
            {"return_images": True},
            {"reasoning_effort": "low"},
        ],
    )
    async def test_preflight_options_precede_observation(
        self, options: dict[str, object]
    ) -> None:
        from amplifier_core.message_models import ChatRequest

        coordinator = MockCoordinator()
        events = []

        async def observe(event, data):
            events.append((event, data))
            return HookResult()

        coordinator.hooks.register("llm:request", observe)
        provider = PerplexityProvider(coordinator=coordinator)
        with pytest.raises(ValueError) as caught:
            await provider.complete(
                ChatRequest(messages=[Message(role="user", content="CANARY")]),
                **options,
            )
        assert "CANARY" not in str(caught.value)
        assert events == [] and provider._client is None

    @pytest.mark.asyncio
    async def test_native_events_usage_and_raw_are_independent(self) -> None:
        from types import MappingProxyType

        from amplifier_core.message_models import ChatRequest

        payload = TestAgentResponseContract._payload()
        payload["usage"] = {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 80,
            "input_tokens_details": {
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 2,
            },
            "output_tokens_details": {"reasoning_tokens": 1},
            "cost": {
                "currency": "USD",
                "input_cost": 0.01,
                "output_cost": 0.037,
                "total_cost": 0.047,
            },
        }
        coordinator = MockCoordinator()
        events = []
        domains = ["original.example"]
        request = ChatRequest(
            messages=[
                Message(role="system", content="S"),
                Message(role="developer", content="D"),
                Message(role="user", content=""),
                Message(role="assistant", content="A"),
            ]
        )
        options = {"search_domain_filter": domains, "timeout": 3, "max_tokens": 17}
        before = deepcopy(options)

        async def observe(event, data):
            assert len(seen) == (0 if event == "llm:request" else 1)
            events.append((event, deepcopy(data)))
            if event == "llm:request":
                data["raw"]["tools"][0]["filters"]["search_domain_filter"].append(
                    "EVIL"
                )
                data["raw"]["input"][0]["content"] = "EVIL"
                data["raw"]["timeout"] = -1
                data["metadata"]["effort"] = "high"
                domains.append("caller-added.example")
                request.messages[0].content = "caller-changed"
            else:
                data["usage"]["input_tokens"] = 999
            return HookResult()

        coordinator.hooks.register("llm:request", observe)
        coordinator.hooks.register("llm:response", observe)
        async with self._wire(
            payload=payload, config={"raw": True}, coordinator=coordinator
        ) as (provider, seen):
            result = await provider.complete(
                request,
                request_options=MappingProxyType(options),
                model="sonar-pro",
                temperature=0,
                top_p=0,
            )
        assert len(seen) == 1
        body = json.loads(seen[0].content)
        assert body["input"] == [
            {"type": "message", "role": "system", "content": "S"},
            {"type": "message", "role": "developer", "content": "D"},
            {"type": "message", "role": "user", "content": ""},
            {"type": "message", "role": "assistant", "content": "A"},
        ]
        assert body["tools"] == [
            {
                "type": "web_search",
                "filters": {"search_domain_filter": ["original.example"]},
            }
        ]
        assert body["temperature"] == 0 and body["top_p"] == 0
        assert body["effort"] == "fast" and "model" not in body
        assert seen[0].extensions["timeout"] == {
            "connect": 5,
            "pool": 5,
            "read": 3,
            "write": 3,
        }
        assert events[0][0] == "llm:request" and events[1][0] == "llm:response"
        first = events[0][1]
        assert first["model"] == "sonar-pro" and first["messages_count"] == 4
        assert first["metadata"] == {"api": "agent", "effort": "fast"}
        assert first["raw"] == {**body, "timeout": 3}
        assert isinstance(first["request_id"], str) and first["request_id"]
        second = events[1][1]
        # Real Core owns correlation; the provider must not substitute an ID.
        assert second["request_id"] == first["request_id"]
        assert second["model"] == "vendor/returned-model"
        assert second["requested_selector"] == "sonar-pro"
        assert second["metadata"] == {"api": "agent", "status": "completed"}
        assert second["usage"] == {
            "input_tokens": 5,
            "output_tokens": 3,
            "cache_read_tokens": 0,
            "cache_write_tokens": 2,
        }
        assert result.usage is not None
        assert result.usage.total_tokens == 80 and result.usage.reasoning_tokens == 1
        assert result.usage.cost_usd == Decimal("0.047")
        assert before == {
            "search_domain_filter": ["original.example"],
            "timeout": 3,
            "max_tokens": 17,
        }
        assert options["max_tokens"] == 17 and options["timeout"] == 3
        assert domains == ["original.example", "caller-added.example"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "timeout",
        [None, 0, 2.5, perplexity.Timeout(connect=0, read=None, write=7, pool=None)],
    )
    async def test_native_timeout_raw_before_normalization(
        self, timeout: float | perplexity.Timeout | None
    ) -> None:
        from amplifier_core.message_models import ChatRequest

        coordinator = MockCoordinator()
        events = []

        async def observe(event, data):
            events.append(deepcopy(data))
            return HookResult()

        coordinator.hooks.register("llm:request", observe)
        async with self._wire(
            config={"raw": True, "timeout": 120}, coordinator=coordinator
        ) as (provider, seen):
            await provider.complete(
                ChatRequest(messages=[Message(role="user", content="Q")]),
                request_options={"timeout": 100},
                timeout=timeout,
            )
            assert provider.timeout == 120
        expected = (
            timeout.as_dict()
            if isinstance(timeout, perplexity.Timeout)
            else {"connect": 5, "pool": 5, "read": timeout, "write": timeout}
        )
        assert seen[0].extensions["timeout"] == expected
        assert events[0]["raw"]["timeout"] == (
            timeout.as_dict() if isinstance(timeout, perplexity.Timeout) else timeout
        )
        assert "timeout" not in json.loads(seen[0].content)
        json.dumps(events)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "state", ["completed", "failed", "queued", "cancelled", "requires_action"]
    )
    async def test_native_state_events_and_no_automatic_continuation(
        self, state: str
    ) -> None:
        from amplifier_core.message_models import ChatRequest

        payload = TestAgentResponseContract._payload()
        payload.update(status=state, usage=None, model="")
        coordinator = MockCoordinator()
        events = []

        async def observe(event, data):
            events.append((event, deepcopy(data)))
            return HookResult()

        for event in ("llm:request", "llm:response", "provider:error"):
            coordinator.hooks.register(event, observe)
        async with self._wire(payload=payload, coordinator=coordinator) as (
            provider,
            seen,
        ):
            request = ChatRequest(messages=[Message(role="user", content="Q")])
            if state != "completed":
                with pytest.raises(LLMError) as caught:
                    await provider.complete(request)
                assert caught.value.retryable is False
                assert [name for name, _ in events] == ["llm:request"]
            else:
                result = await provider.complete(request)
                assert result.usage is None
                assert events[1][1]["model"] is None
                assert events[1][1]["usage"] == {"input_tokens": 0, "output_tokens": 0}
                assert "raw" not in events[0][1]
        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_catalog_v1_fresh_snapshots(self) -> None:
        config = {"model_filters": {"include": ["SONAR*"]}}
        provider = PerplexityProvider(config=config)
        config["model_filters"]["include"].clear()
        info = provider.get_model_list_info()
        assert info["version"] == 1 and info["source"] == "static"
        assert info["filters_keyword"] == "filters" and info["override"] == "replace"
        assert info["unfiltered"] == {}
        info["default_filters"]["include"].clear()
        info["filters_schema"]["properties"].clear()
        assert [m.id for m in await provider.list_models()] == [
            "sonar",
            "sonar-pro",
            "sonar-reasoning-pro",
            "sonar-deep-research",
        ]
        rows = await provider.list_models(filters={})
        assert [row.id for row in rows] == [
            "agent-fast",
            "agent-low",
            "agent-medium",
            "agent-high",
            "sonar",
            "sonar-pro",
            "sonar-reasoning-pro",
            "sonar-deep-research",
        ]
        assert [row.display_name for row in rows] == [
            "Agent Fast",
            "Agent Low",
            "Agent Medium",
            "Agent High",
            "Sonar (Agent Fast alias)",
            "Sonar Pro (Agent Fast alias)",
            "Sonar Reasoning Pro (Agent Low alias)",
            "Sonar Deep Research (Agent High alias)",
        ]
        for row in rows:
            assert row.context_window == 0 and row.max_output_tokens == 0
            assert row.defaults == {} and row.pricing is None
            assert row.capabilities == ["web_search", "citations"]
            row.capabilities.clear()
            row.defaults["mutated"] = True
        fresh = await provider.list_models(filters={})
        assert all(
            row.capabilities == ["web_search", "citations"] and row.defaults == {}
            for row in fresh
        )
        assert provider.get_model_list_info()["filters_schema"]["properties"]
        assert await provider.list_models(filters={"include": []}) == []
        assert await provider.list_models(filters={"exclude": ["*"]}) == []
        assert provider._client is None
