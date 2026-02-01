"""Unit tests for PerplexityProvider.

Tests cover:
- Message conversion for all roles
- Response parsing with citations
- Provider info and model listing
- Error handling
- SDK client lifecycle
"""

from unittest.mock import MagicMock

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
        """Tool messages should be skipped (not supported by Perplexity)."""

        class MockMessage:
            role = "tool"
            content = "Tool result"

        messages = [MockMessage()]
        system_prompt, conversation = self.provider._convert_messages(messages)

        assert system_prompt is None
        assert len(conversation) == 0

    def test_convert_mixed_messages(self) -> None:
        """Mixed message types should be handled correctly."""

        class SystemMsg:
            role = "system"
            content = "Be helpful."

        class UserMsg:
            role = "user"
            content = "Hello"

        class AssistantMsg:
            role = "assistant"
            content = "Hi there!"

        class UserMsg2:
            role = "user"
            content = "How are you?"

        messages = [SystemMsg(), UserMsg(), AssistantMsg(), UserMsg2()]
        system_prompt, conversation = self.provider._convert_messages(messages)

        assert system_prompt == "Be helpful."
        assert len(conversation) == 3
        assert conversation[0] == {"role": "user", "content": "Hello"}
        assert conversation[1] == {"role": "assistant", "content": "Hi there!"}
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

    def test_extract_dict_text_blocks(self) -> None:
        """Dict text blocks should be extracted."""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        result = self.provider._extract_text_content(content)
        assert result == "Hello\nWorld"

    def test_extract_textblock_objects(self) -> None:
        """TextBlock objects should be extracted."""
        from amplifier_core.message_models import TextBlock

        content = [
            TextBlock(type="text", text="Hello"),
            TextBlock(type="text", text="World"),
        ]
        result = self.provider._extract_text_content(content)
        assert result == "Hello\nWorld"

    def test_extract_empty_list(self) -> None:
        """Empty list should return empty string."""
        result = self.provider._extract_text_content([])
        assert result == ""


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
    ) -> MagicMock:
        """Create a mock SDK response object."""
        mock_response = MagicMock()
        mock_response.id = response_id
        mock_response.model = model

        # Mock choice
        mock_choice = MagicMock()
        mock_choice.message = MagicMock()
        mock_choice.message.content = content
        mock_choice.finish_reason = finish_reason
        mock_response.choices = [mock_choice]

        # Mock usage
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = prompt_tokens
        mock_response.usage.completion_tokens = completion_tokens

        # Mock citations
        mock_response.citations = citations

        return mock_response

    def test_convert_basic_response(self) -> None:
        """Basic response should be converted correctly."""
        mock_response = self._create_mock_response(
            content="The capital of France is Paris.",
            model="sonar",
            prompt_tokens=10,
            completion_tokens=20,
        )

        response = self.provider._convert_sdk_response(mock_response, "sonar")

        assert response.model == "sonar"
        assert "The capital of France is Paris." in response.content[0].text
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20

    def test_convert_response_with_citations(self) -> None:
        """Response with citations should include them in content."""
        mock_response = self._create_mock_response(
            content="Paris is the capital.",
            citations=["https://example.com/paris", "https://wiki.org/france"],
        )

        response = self.provider._convert_sdk_response(mock_response, "sonar")

        assert "Paris is the capital." in response.content[0].text
        assert "Sources:" in response.content[0].text
        assert "https://example.com/paris" in response.content[0].text
        assert "https://wiki.org/france" in response.content[0].text

    def test_convert_response_stop_reason_length(self) -> None:
        """Length finish reason should map to max_tokens."""
        mock_response = self._create_mock_response(finish_reason="length")

        response = self.provider._convert_sdk_response(mock_response, "sonar")

        assert response.stop_reason == "max_tokens"

    def test_convert_response_stop_reason_stop(self) -> None:
        """Stop finish reason should map to end_turn."""
        mock_response = self._create_mock_response(finish_reason="stop")

        response = self.provider._convert_sdk_response(mock_response, "sonar")

        assert response.stop_reason == "end_turn"

    def test_convert_empty_choices(self) -> None:
        """Empty choices should return empty content."""
        mock_response = MagicMock()
        mock_response.id = "test-id"
        mock_response.model = "sonar"
        mock_response.choices = []
        mock_response.usage = None
        mock_response.citations = None

        response = self.provider._convert_sdk_response(mock_response, "sonar")

        assert response.content[0].text == ""


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

        result = self.provider.parse_tool_calls(MockResponse())
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

    def test_client_lazy_initialization(self) -> None:
        """Client should be initialized on first access."""
        provider = PerplexityProvider(api_key="test-key")
        client = provider.client
        assert client is not None
        assert provider._client is not None

    def test_client_reused_on_subsequent_access(self) -> None:
        """Same client instance should be returned on subsequent access."""
        provider = PerplexityProvider(api_key="test-key")
        client1 = provider.client
        client2 = provider.client
        assert client1 is client2

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

        assert provider.default_model == "sonar"
        assert provider.max_tokens == 4096
        assert provider.temperature == 0.7
        assert provider.timeout == 60.0
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


class TestMount:
    """Test the mount() function."""

    @pytest.mark.asyncio
    async def test_mount_with_api_key_env(self, monkeypatch) -> None:
        """Mount should work with API key from environment."""
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key-from-env")

        from amplifier_module_provider_perplexity import mount
        from amplifier_core import TestCoordinator

        coordinator = TestCoordinator()
        cleanup = await mount(coordinator, {})

        assert cleanup is not None
        # Verify provider was mounted
        assert any(
            entry.get("name") == "perplexity"
            for entry in coordinator.mount_history
            if entry.get("mount_point") == "providers"
        )

        # Cleanup
        await cleanup()

    @pytest.mark.asyncio
    async def test_mount_without_api_key_returns_none(self, monkeypatch) -> None:
        """Mount should return None if no API key available."""
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

        from amplifier_module_provider_perplexity import mount
        from amplifier_core import TestCoordinator

        coordinator = TestCoordinator()
        result = await mount(coordinator, {})

        assert result is None
