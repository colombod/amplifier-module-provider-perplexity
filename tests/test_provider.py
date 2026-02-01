"""Unit tests for PerplexityProvider.

Tests cover:
- Message conversion for all roles
- Response parsing with citations
- Provider info and model listing
- Error handling
"""

import pytest

from amplifier_module_provider_perplexity import PerplexityProvider


class TestMessageConversion:
    """Test message conversion from Amplifier format to Perplexity format."""

    def setup_method(self) -> None:
        """Set up test provider instance."""
        self.provider = PerplexityProvider(api_key="test-key")

    def test_convert_system_message(self) -> None:
        """System messages should be extracted into system prompt."""

        class MockMessage:
            role = "system"
            content = "You are a helpful assistant."

        messages = [MockMessage()]
        system_prompt, conversation = self.provider._convert_messages(messages)

        assert system_prompt == "You are a helpful assistant."
        assert conversation == []

    def test_convert_multiple_system_messages(self) -> None:
        """Multiple system messages should be combined."""

        class MockMessage1:
            role = "system"
            content = "You are helpful."

        class MockMessage2:
            role = "system"
            content = "Be concise."

        messages = [MockMessage1(), MockMessage2()]
        system_prompt, conversation = self.provider._convert_messages(messages)

        assert system_prompt == "You are helpful.\n\nBe concise."
        assert conversation == []

    def test_convert_user_message(self) -> None:
        """User messages should pass through directly."""

        class MockMessage:
            role = "user"
            content = "Hello, how are you?"

        messages = [MockMessage()]
        system_prompt, conversation = self.provider._convert_messages(messages)

        assert system_prompt is None
        assert len(conversation) == 1
        assert conversation[0] == {"role": "user", "content": "Hello, how are you?"}

    def test_convert_assistant_message(self) -> None:
        """Assistant messages should pass through directly."""

        class MockMessage:
            role = "assistant"
            content = "I'm doing well, thank you!"

        messages = [MockMessage()]
        system_prompt, conversation = self.provider._convert_messages(messages)

        assert system_prompt is None
        assert len(conversation) == 1
        assert conversation[0] == {
            "role": "assistant",
            "content": "I'm doing well, thank you!",
        }

    def test_convert_developer_message(self) -> None:
        """Developer messages should be wrapped in context_file tags."""

        class MockMessage:
            role = "developer"
            content = "This is context information."

        messages = [MockMessage()]
        system_prompt, conversation = self.provider._convert_messages(messages)

        assert system_prompt is None
        assert len(conversation) == 1
        assert conversation[0]["role"] == "user"
        assert "<context_file>" in conversation[0]["content"]
        assert "This is context information." in conversation[0]["content"]
        assert "</context_file>" in conversation[0]["content"]

    def test_convert_tool_message_skipped(self) -> None:
        """Tool messages should be skipped (not supported)."""

        class MockMessage:
            role = "tool"
            content = "Tool result"

        messages = [MockMessage()]
        system_prompt, conversation = self.provider._convert_messages(messages)

        assert system_prompt is None
        assert conversation == []

    def test_convert_mixed_messages(self) -> None:
        """Mixed message types should be handled correctly."""

        class SystemMsg:
            role = "system"
            content = "Be helpful."

        class UserMsg:
            role = "user"
            content = "Hi"

        class AssistantMsg:
            role = "assistant"
            content = "Hello!"

        class UserMsg2:
            role = "user"
            content = "How are you?"

        messages = [SystemMsg(), UserMsg(), AssistantMsg(), UserMsg2()]
        system_prompt, conversation = self.provider._convert_messages(messages)

        assert system_prompt == "Be helpful."
        assert len(conversation) == 3
        assert conversation[0] == {"role": "user", "content": "Hi"}
        assert conversation[1] == {"role": "assistant", "content": "Hello!"}
        assert conversation[2] == {"role": "user", "content": "How are you?"}


class TestTextExtraction:
    """Test text content extraction from various formats."""

    def setup_method(self) -> None:
        """Set up test provider instance."""
        self.provider = PerplexityProvider(api_key="test-key")

    def test_extract_string_content(self) -> None:
        """String content should be returned as-is."""
        result = self.provider._extract_text_content("Hello world")
        assert result == "Hello world"

    def test_extract_from_dict_blocks(self) -> None:
        """Dictionary content blocks should be extracted."""
        content = [
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
        ]
        result = self.provider._extract_text_content(content)
        assert "First part" in result
        assert "Second part" in result

    def test_extract_empty_list(self) -> None:
        """Empty list should return empty string."""
        result = self.provider._extract_text_content([])
        assert result == ""


class TestResponseParsing:
    """Test conversion of Perplexity API responses."""

    def setup_method(self) -> None:
        """Set up test provider instance."""
        self.provider = PerplexityProvider(api_key="test-key")

    def test_parse_basic_response(self) -> None:
        """Basic response should be parsed correctly."""
        data = {
            "choices": [
                {
                    "message": {"content": "This is the response."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }

        response = self.provider._convert_to_chat_response(data)

        assert len(response.content) == 1
        assert response.content[0].text == "This is the response."
        assert response.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.usage.total_tokens == 30

    def test_parse_response_with_citations(self) -> None:
        """Response with citations should include them in metadata."""
        data = {
            "choices": [
                {
                    "message": {"content": "Paris is the capital [1]."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "citations": [{"url": "https://example.com/paris", "title": "Paris Facts"}],
        }

        response = self.provider._convert_to_chat_response(data)

        assert response.metadata is not None
        assert "citations" in response.metadata
        assert len(response.metadata["citations"]) == 1
        assert response.metadata["citations"][0]["url"] == "https://example.com/paris"

    def test_parse_response_with_related_questions(self) -> None:
        """Response with related questions should include them in metadata."""
        data = {
            "choices": [
                {
                    "message": {"content": "The answer is 42."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            "related_questions": [
                "What is the meaning of life?",
                "Why 42?",
            ],
        }

        response = self.provider._convert_to_chat_response(data)

        assert response.metadata is not None
        assert "related_questions" in response.metadata
        assert len(response.metadata["related_questions"]) == 2

    def test_parse_empty_choices(self) -> None:
        """Empty choices should return empty response."""
        data = {"choices": [], "usage": {}}

        response = self.provider._convert_to_chat_response(data)

        assert response.content[0].text == ""
        assert response.finish_reason == "error"


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
        assert "streaming" in info.capabilities
        assert "citations" in info.capabilities
        assert "web_search" in info.capabilities

    def test_get_info_has_config_fields(self) -> None:
        """get_info() should include configuration fields."""
        info = self.provider.get_info()

        field_ids = [f.id for f in info.config_fields]
        assert "api_key" in field_ids
        assert "search_recency_filter" in field_ids
        assert "search_context_size" in field_ids

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

    def test_list_models_returns_expected_models(self) -> None:
        """list_models() should return all expected models."""
        models = self.provider.list_models()
        model_ids = [m.id for m in models]

        assert "sonar" in model_ids
        assert "sonar-pro" in model_ids
        assert "sonar-reasoning" in model_ids
        assert "sonar-reasoning-pro" in model_ids
        assert "sonar-deep-research" in model_ids
        assert "r1-1776" in model_ids

    def test_sonar_model_capabilities(self) -> None:
        """Sonar model should have correct capabilities."""
        models = self.provider.list_models()
        sonar = next(m for m in models if m.id == "sonar")

        assert sonar.context_window == 128000
        assert sonar.max_output_tokens == 4096
        assert "web_search" in sonar.capabilities
        assert "citations" in sonar.capabilities
        assert "streaming" in sonar.capabilities

    def test_sonar_pro_model_capabilities(self) -> None:
        """Sonar Pro model should have enhanced capabilities."""
        models = self.provider.list_models()
        sonar_pro = next(m for m in models if m.id == "sonar-pro")

        assert sonar_pro.context_window == 200000
        assert sonar_pro.max_output_tokens == 8192
        assert "deep_search" in sonar_pro.capabilities

    def test_r1_model_is_offline(self) -> None:
        """R1-1776 model should be offline (no web search)."""
        models = self.provider.list_models()
        r1 = next(m for m in models if m.id == "r1-1776")

        assert "offline" in r1.capabilities
        assert "web_search" not in r1.capabilities


class TestToolCalls:
    """Test tool call handling."""

    def setup_method(self) -> None:
        """Set up test provider instance."""
        self.provider = PerplexityProvider(api_key="test-key")

    def test_parse_tool_calls_returns_empty_list(self) -> None:
        """parse_tool_calls() should always return empty list."""

        class MockResponse:
            content = []
            usage = None
            finish_reason = "stop"
            metadata = None

        result = self.provider.parse_tool_calls(MockResponse())
        assert result == []


class TestConfiguration:
    """Test provider configuration handling."""

    def test_default_configuration(self) -> None:
        """Provider should have sensible defaults."""
        provider = PerplexityProvider(api_key="test-key")

        assert provider.default_model == "sonar"
        assert provider.max_tokens == 4096
        assert provider.temperature == 0.7
        assert provider.timeout == 60.0
        assert provider.max_retries == 3
        assert provider.search_context_size == "medium"
        assert provider.return_images is False
        assert provider.return_related_questions is False

    def test_custom_configuration(self) -> None:
        """Provider should accept custom configuration."""
        config = {
            "default_model": "sonar-pro",
            "max_tokens": 8192,
            "temperature": 0.5,
            "timeout": 120.0,
            "search_recency_filter": "week",
            "search_context_size": "high",
            "return_images": True,
        }
        provider = PerplexityProvider(api_key="test-key", config=config)

        assert provider.default_model == "sonar-pro"
        assert provider.max_tokens == 8192
        assert provider.temperature == 0.5
        assert provider.timeout == 120.0
        assert provider.search_recency_filter == "week"
        assert provider.search_context_size == "high"
        assert provider.return_images is True

    def test_client_lazy_initialization(self) -> None:
        """HTTP client should be lazily initialized."""
        provider = PerplexityProvider(api_key="test-key")

        assert provider._client is None

    def test_client_raises_without_api_key(self) -> None:
        """Accessing client without API key should raise ValueError."""
        provider = PerplexityProvider(api_key=None)

        with pytest.raises(ValueError, match="api_key must be provided"):
            _ = provider.client


class TestRequestBuilding:
    """Test request parameter building."""

    def setup_method(self) -> None:
        """Set up test provider instance."""
        self.provider = PerplexityProvider(
            api_key="test-key",
            config={
                "search_domain_filter": ["example.com"],
                "search_recency_filter": "week",
                "search_context_size": "high",
            },
        )

    def test_build_basic_request(self) -> None:
        """Basic request should include required parameters."""

        class MockRequest:
            model = "sonar"
            max_tokens = 1000
            temperature = 0.5
            messages = []

        messages = [{"role": "user", "content": "Hello"}]
        params = self.provider._build_request_params(MockRequest(), None, messages)

        assert params["model"] == "sonar"
        assert params["max_tokens"] == 1000
        assert params["temperature"] == 0.5
        assert params["messages"] == messages

    def test_build_request_with_system_prompt(self) -> None:
        """Request with system prompt should include it."""

        class MockRequest:
            model = "sonar"
            max_tokens = 1000
            temperature = 0.5
            messages = []

        messages = [{"role": "user", "content": "Hello"}]
        params = self.provider._build_request_params(
            MockRequest(), "Be helpful.", messages
        )

        assert params["system"] == "Be helpful."

    def test_build_request_with_search_options(self) -> None:
        """Request should include web search options from config."""

        class MockRequest:
            model = "sonar"
            max_tokens = 1000
            temperature = 0.5
            messages = []

        messages = [{"role": "user", "content": "Hello"}]
        params = self.provider._build_request_params(MockRequest(), None, messages)

        assert "web_search_options" in params
        assert params["web_search_options"]["search_domain_filter"] == ["example.com"]
        assert params["web_search_options"]["search_recency_filter"] == "week"
        assert params["web_search_options"]["search_context_size"] == "high"
