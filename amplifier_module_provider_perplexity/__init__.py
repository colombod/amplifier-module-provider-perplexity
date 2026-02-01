"""
Perplexity AI Provider for Amplifier.

A provider module that integrates Perplexity's web-grounded AI search
capabilities into Amplifier, featuring real-time web search, citations,
and multiple specialized models.

Uses the official Perplexity Python SDK for robust API interaction.

Example:
    >>> from amplifier_module_provider_perplexity import PerplexityProvider
    >>> provider = PerplexityProvider(api_key="pplx-xxx")
    >>> response = await provider.complete(request)
"""

import logging
import os
from typing import Any

import perplexity
from perplexity import AsyncPerplexity

from amplifier_core import ConfigField, ModelInfo, ModuleCoordinator, ProviderInfo
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    Message,
    TextBlock,
    ToolCall,
    Usage,
)

__all__ = ["PerplexityProvider", "mount"]

logger = logging.getLogger(__name__)


class PerplexityProvider:
    """Perplexity AI provider with web-grounded search capabilities.

    This provider integrates with Perplexity's API using the official SDK
    to provide AI responses grounded in real-time web search results,
    complete with citations.

    Attributes:
        name: Provider identifier ("perplexity")
        api_label: Human-readable label ("Perplexity")
    """

    name = "perplexity"
    api_label = "Perplexity"

    def __init__(
        self,
        api_key: str | None = None,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ) -> None:
        """Initialize the Perplexity provider.

        Args:
            api_key: Perplexity API key. Falls back to PERPLEXITY_API_KEY env var.
            config: Provider configuration options.
            coordinator: Module coordinator for event emission.
        """
        self._api_key = api_key
        self._client: AsyncPerplexity | None = None
        self.config = config or {}
        self.coordinator = coordinator

        # Model and generation settings
        self.default_model: str = self.config.get("default_model", "sonar")
        self.max_tokens: int = self.config.get("max_tokens", 4096)
        self.temperature: float = self.config.get("temperature", 0.7)

        # HTTP settings (passed to SDK)
        self.timeout: float = self.config.get("timeout", 60.0)
        self.max_retries: int = self.config.get("max_retries", 2)

        # Perplexity-specific search settings
        self.search_recency_filter: str | None = self.config.get(
            "search_recency_filter"
        )
        self.search_domain_filter: list[str] | None = self.config.get(
            "search_domain_filter"
        )
        self.search_context_size: str = self.config.get("search_context_size", "medium")
        self.return_images: bool = self.config.get("return_images", False)
        self.return_related_questions: bool = self.config.get(
            "return_related_questions", False
        )

        # Debug mode
        self.debug: bool = self.config.get("debug", False)

    @property
    def client(self) -> AsyncPerplexity:
        """Lazily initialize the Perplexity SDK client.

        Returns:
            Configured AsyncPerplexity client for API calls.

        Raises:
            ValueError: If api_key is not provided.
        """
        if self._client is None:
            if self._api_key is None:
                raise ValueError("api_key must be provided for API calls")
            self._client = AsyncPerplexity(
                api_key=self._api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    async def close(self) -> None:
        """Close the SDK client if initialized.

        Call this method when done using the provider directly.
        When using via mount(), cleanup is handled automatically.
        """
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> "PerplexityProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit - ensures client cleanup."""
        await self.close()

    def get_info(self) -> ProviderInfo:
        """Get provider metadata and configuration schema.

        Returns:
            ProviderInfo containing provider details and config fields.
        """
        return ProviderInfo(
            id="perplexity",
            display_name="Perplexity AI",
            credential_env_vars=["PERPLEXITY_API_KEY"],
            capabilities=["streaming", "citations", "web_search"],
            config_fields=[
                ConfigField(
                    id="api_key",
                    display_name="API Key",
                    field_type="secret",
                    prompt="Enter your Perplexity API key",
                    env_var="PERPLEXITY_API_KEY",
                ),
                ConfigField(
                    id="search_recency_filter",
                    display_name="Search Recency",
                    field_type="choice",
                    prompt="Filter search results by recency",
                    choices=["none", "hour", "day", "week", "month", "year"],
                    required=False,
                    default="none",
                ),
                ConfigField(
                    id="search_context_size",
                    display_name="Search Context Size",
                    field_type="choice",
                    prompt="Amount of search context to include",
                    choices=["low", "medium", "high"],
                    required=False,
                    default="medium",
                ),
            ],
        )

    def list_models(self) -> list[ModelInfo]:
        """List available Perplexity models.

        Returns:
            List of ModelInfo for each supported model.
        """
        return [
            ModelInfo(
                id="sonar",
                display_name="Sonar",
                context_window=128000,
                max_output_tokens=4096,
                capabilities=["web_search", "citations", "streaming"],
            ),
            ModelInfo(
                id="sonar-pro",
                display_name="Sonar Pro",
                context_window=200000,
                max_output_tokens=8192,
                capabilities=["web_search", "citations", "streaming", "deep_search"],
            ),
            ModelInfo(
                id="sonar-reasoning",
                display_name="Sonar Reasoning",
                context_window=128000,
                max_output_tokens=4096,
                capabilities=["web_search", "citations", "streaming", "reasoning"],
            ),
            ModelInfo(
                id="sonar-reasoning-pro",
                display_name="Sonar Reasoning Pro",
                context_window=128000,
                max_output_tokens=8192,
                capabilities=[
                    "web_search",
                    "citations",
                    "streaming",
                    "reasoning",
                    "deep_search",
                ],
            ),
            ModelInfo(
                id="sonar-deep-research",
                display_name="Sonar Deep Research",
                context_window=128000,
                max_output_tokens=8192,
                capabilities=["web_search", "citations", "deep_research"],
            ),
            ModelInfo(
                id="r1-1776",
                display_name="R1-1776",
                context_window=128000,
                max_output_tokens=4096,
                capabilities=["offline"],
            ),
        ]

    async def complete(self, request: ChatRequest, **kwargs: Any) -> ChatResponse:
        """Complete a chat request using Perplexity API.

        Args:
            request: Chat request with messages and parameters.
            **kwargs: Additional parameters to override defaults.

        Returns:
            ChatResponse with the model's response and metadata.

        Raises:
            perplexity.APIError: On API errors after retries exhausted.
            ValueError: If API key is not configured.
        """
        # Convert messages to Perplexity format
        system_prompt, messages = self._convert_messages(request.messages)

        # Build request parameters
        model = kwargs.get("model") or request.model or self.default_model
        max_tokens = kwargs.get("max_tokens") or request.max_tokens or self.max_tokens
        temperature = (
            kwargs.get("temperature")
            if kwargs.get("temperature") is not None
            else (
                request.temperature
                if request.temperature is not None
                else self.temperature
            )
        )

        # Emit request event if coordinator available
        if self.coordinator:
            await self.coordinator.emit(
                "llm:request",
                {
                    "provider": self.name,
                    "model": model,
                    "messages_count": len(messages),
                },
            )

        # Build web search options
        web_search_options: dict[str, Any] = {}

        if self.search_domain_filter:
            web_search_options["search_domain_filter"] = self.search_domain_filter

        search_recency = (
            kwargs.get("search_recency_filter") or self.search_recency_filter
        )
        if search_recency and search_recency != "none":
            web_search_options["search_recency_filter"] = search_recency

        search_context = kwargs.get("search_context_size") or self.search_context_size
        if search_context:
            web_search_options["search_context_size"] = search_context

        # Make API request using SDK
        try:
            if self.debug:
                logger.debug(
                    f"Perplexity request: model={model}, messages={len(messages)}"
                )

            # Prepare messages for SDK (add system message if present)
            sdk_messages: list[dict[str, str]] = []
            if system_prompt:
                sdk_messages.append({"role": "system", "content": system_prompt})
            sdk_messages.extend(messages)

            # Call the SDK
            response = await self.client.chat.completions.create(
                model=model,
                messages=sdk_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                web_search_options=web_search_options if web_search_options else None,
            )

            # Convert SDK response to ChatResponse
            chat_response = self._convert_sdk_response(response, model)

            # Emit response event
            if self.coordinator:
                await self.coordinator.emit(
                    "llm:response",
                    {
                        "provider": self.name,
                        "model": model,
                        "usage": {
                            "input_tokens": chat_response.usage.input_tokens
                            if chat_response.usage
                            else 0,
                            "output_tokens": chat_response.usage.output_tokens
                            if chat_response.usage
                            else 0,
                        },
                    },
                )

            return chat_response

        except perplexity.RateLimitError as e:
            logger.warning(f"Perplexity rate limit hit: {e}")
            raise
        except perplexity.APIStatusError as e:
            logger.error(f"Perplexity API error: {e.status_code} - {e.message}")
            raise
        except perplexity.APIConnectionError as e:
            logger.error(f"Perplexity connection error: {e}")
            raise

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """Parse tool calls from response.

        Perplexity does not support tool calling, so this always returns
        an empty list.

        Args:
            response: The chat response to parse.

        Returns:
            Empty list (Perplexity doesn't support tools).
        """
        return []

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Convert Amplifier messages to Perplexity format.

        Handles system, developer, user, assistant, and tool roles:
        - system: Combined into single system prompt
        - developer: Wrapped in context_file tags as user message
        - user/assistant: Direct passthrough with text extraction
        - tool: Skipped (not supported)

        Args:
            messages: List of Amplifier Message objects.

        Returns:
            Tuple of (system_prompt, conversation_messages).
        """
        system_parts: list[str] = []
        conversation: list[dict[str, str]] = []

        for msg in messages:
            role = msg.role

            if role == "system":
                text = self._extract_text_content(msg.content)
                if text:
                    system_parts.append(text)

            elif role == "developer":
                # Wrap developer context in tags and add as user message
                text = self._extract_text_content(msg.content)
                if text:
                    wrapped = f"<context_file>\n{text}\n</context_file>"
                    conversation.append({"role": "user", "content": wrapped})

            elif role == "user":
                text = self._extract_text_content(msg.content)
                if text:
                    conversation.append({"role": "user", "content": text})

            elif role == "assistant":
                text = self._extract_text_content(msg.content)
                if text:
                    conversation.append({"role": "assistant", "content": text})

            elif role == "tool":
                # Tool messages not supported, skip
                if self.debug:
                    logger.debug("Skipping tool message (not supported by Perplexity)")

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, conversation

    def _extract_text_content(self, content: str | list[Any]) -> str:
        """Extract text content from message content.

        Args:
            content: Either a string or list of content blocks.

        Returns:
            Extracted text as a single string.
        """
        if isinstance(content, str):
            return content

        # Handle list of content blocks
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                text_parts.append(getattr(block, "text"))

        return "\n".join(text_parts)

    def _convert_sdk_response(self, response: Any, model: str) -> ChatResponse:
        """Convert SDK response to Amplifier ChatResponse.

        Args:
            response: Response from Perplexity SDK.
            model: Model used for the request.

        Returns:
            ChatResponse in Amplifier format.
        """
        # Extract content from SDK response
        content = ""
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if choice.message and choice.message.content:
                content = choice.message.content

        # Extract citations if present
        citations: list[str] = []
        if hasattr(response, "citations") and response.citations:
            citations = response.citations

        # Build content with citations
        full_content = content
        if citations:
            citation_text = "\n\n**Sources:**\n" + "\n".join(
                f"- [{i + 1}] {url}" for i, url in enumerate(citations)
            )
            full_content = content + citation_text

        # Extract usage
        usage = None
        if response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0
            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )

        # Determine stop reason
        stop_reason = "end_turn"
        if response.choices and response.choices[0].finish_reason:
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                stop_reason = "max_tokens"
            elif finish_reason == "stop":
                stop_reason = "end_turn"

        return ChatResponse(
            id=response.id or "pplx-response",
            model=response.model or model,
            content=[TextBlock(type="text", text=full_content)],
            stop_reason=stop_reason,
            usage=usage,
        )


async def mount(
    coordinator: ModuleCoordinator, config: dict[str, Any] | None = None
) -> Any:
    """Mount the Perplexity provider to an Amplifier coordinator.

    This function is the entry point for module loading. It creates
    the provider instance and registers it with the coordinator.

    Args:
        coordinator: The module coordinator to mount to.
        config: Optional configuration dict.

    Returns:
        Cleanup function to unmount and close resources, or None if
        API key is not available.
    """
    config = config or {}

    # Get API key from config or environment
    api_key = config.get("api_key") or os.environ.get("PERPLEXITY_API_KEY")

    if not api_key:
        logger.warning(
            "Perplexity provider not mounted: PERPLEXITY_API_KEY not set. "
            "Set the environment variable or provide api_key in config."
        )
        return None

    # Create provider instance
    provider = PerplexityProvider(
        api_key=api_key,
        config=config,
        coordinator=coordinator,
    )

    # Mount to coordinator
    await coordinator.mount("providers", provider, name=provider.name)
    logger.info(f"Perplexity provider mounted with model: {provider.default_model}")

    # Return cleanup function
    async def cleanup() -> None:
        await provider.close()
        logger.debug("Perplexity provider cleanup complete")

    return cleanup
