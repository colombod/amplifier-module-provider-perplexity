"""
Perplexity AI Provider for Amplifier.

A provider module that integrates Perplexity's web-grounded AI search
capabilities into Amplifier, featuring real-time web search, citations,
and multiple specialized models.

Example:
    >>> from amplifier_module_provider_perplexity import PerplexityProvider
    >>> provider = PerplexityProvider(api_key="pplx-xxx")
    >>> response = await provider.complete(request)
"""

import asyncio
import logging
import os
from typing import Any

import httpx

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

    This provider integrates with Perplexity's API to provide AI responses
    grounded in real-time web search results, complete with citations.

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
        self._client: httpx.AsyncClient | None = None
        self.config = config or {}
        self.coordinator = coordinator

        # Model and generation settings
        self.default_model: str = self.config.get("default_model", "sonar")
        self.max_tokens: int = self.config.get("max_tokens", 4096)
        self.temperature: float = self.config.get("temperature", 0.7)

        # HTTP settings
        self.timeout: float = self.config.get("timeout", 60.0)
        self.max_retries: int = self.config.get("max_retries", 3)

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
    def client(self) -> httpx.AsyncClient:
        """Lazily initialize HTTP client.

        Returns:
            Configured httpx.AsyncClient for API calls.

        Raises:
            ValueError: If api_key is not provided.
        """
        if self._client is None:
            if self._api_key is None:
                raise ValueError("api_key must be provided for API calls")
            self._client = httpx.AsyncClient(
                base_url="https://api.perplexity.ai",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

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
            httpx.HTTPStatusError: On API errors after retries exhausted.
            ValueError: If API key is not configured.
        """
        # Convert messages to Perplexity format
        system_prompt, messages = self._convert_messages(request.messages)

        # Build request parameters
        params = self._build_request_params(request, system_prompt, messages, **kwargs)

        # Emit request event if coordinator available
        if self.coordinator:
            await self.coordinator.emit(
                "llm:request",
                {
                    "provider": self.name,
                    "model": params.get("model"),
                    "messages_count": len(messages),
                },
            )

        # Make API request with retry
        data = await self._make_request(params)

        # Convert response
        response = self._convert_to_chat_response(data)

        # Emit response event
        if self.coordinator:
            await self.coordinator.emit(
                "llm:response",
                {
                    "provider": self.name,
                    "model": params.get("model"),
                    "usage": {
                        "input_tokens": response.usage.input_tokens
                        if response.usage
                        else 0,
                        "output_tokens": response.usage.output_tokens
                        if response.usage
                        else 0,
                    },
                },
            )

        return response

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
    ) -> tuple[str | None, list[dict[str, Any]]]:
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
        conversation: list[dict[str, Any]] = []

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

    def _build_request_params(
        self,
        request: ChatRequest,
        system_prompt: str | None,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build Perplexity API request parameters.

        Args:
            request: Original chat request.
            system_prompt: Extracted system prompt.
            messages: Converted conversation messages.
            **kwargs: Override parameters.

        Returns:
            Dictionary of API parameters.
        """
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

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add system prompt if present
        if system_prompt:
            params["system"] = system_prompt

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

        if web_search_options:
            params["web_search_options"] = web_search_options

        # Perplexity-specific options
        if self.return_images:
            params["return_images"] = True

        if self.return_related_questions:
            params["return_related_questions"] = True

        return params

    async def _make_request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Make HTTP request to Perplexity API with retry logic.

        Implements exponential backoff for rate limits and server errors.

        Args:
            params: Request parameters.

        Returns:
            Parsed JSON response.

        Raises:
            httpx.HTTPStatusError: On unrecoverable errors or retries exhausted.
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                if self.debug:
                    logger.debug(
                        f"API request attempt {attempt + 1}/{self.max_retries}"
                    )

                response = await self.client.post("/chat/completions", json=params)
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                # 401: Invalid key - don't retry
                if status_code == 401:
                    logger.error("Invalid API key")
                    raise

                # 429: Rate limited - use retry-after header
                if status_code == 429:
                    retry_after = e.response.headers.get("retry-after", "1")
                    try:
                        wait_time = float(retry_after)
                    except ValueError:
                        wait_time = 2**attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    last_exception = e
                    continue

                # 5xx: Server error - retry with backoff
                if status_code >= 500:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Server error {status_code}, retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                    last_exception = e
                    continue

                # Other errors - don't retry
                raise

            except httpx.RequestError as e:
                # Network errors - retry with backoff
                wait_time = 2**attempt
                logger.warning(f"Request error: {e}, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
                last_exception = e

        # Retries exhausted
        if last_exception:
            raise last_exception
        raise RuntimeError("Request failed with no exception captured")

    def _convert_to_chat_response(self, data: dict[str, Any]) -> ChatResponse:
        """Convert Perplexity API response to ChatResponse.

        Args:
            data: Raw API response data.

        Returns:
            ChatResponse with extracted content and metadata.
        """
        # Extract message content
        choices = data.get("choices", [])
        if not choices:
            return ChatResponse(
                content=[TextBlock(type="text", text="")],
                usage=None,
                finish_reason="error",
            )

        choice = choices[0]
        message = choice.get("message", {})
        content_text = message.get("content", "")
        finish_reason = choice.get("finish_reason", "stop")

        # Build content blocks
        content = [TextBlock(type="text", text=content_text)]

        # Extract usage
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Build metadata with Perplexity-specific fields
        metadata: dict[str, Any] = {}

        if "citations" in data:
            metadata["citations"] = data["citations"]

        if "related_questions" in data:
            metadata["related_questions"] = data["related_questions"]

        if "search_results" in data:
            metadata["search_results"] = data["search_results"]

        return ChatResponse(
            content=content,
            usage=usage,
            finish_reason=finish_reason,
            metadata=metadata if metadata else None,
        )


async def mount(
    coordinator: ModuleCoordinator, config: dict[str, Any] | None = None
) -> Any:
    """Mount the Perplexity provider module.

    Args:
        coordinator: Amplifier module coordinator.
        config: Optional provider configuration.

    Returns:
        Cleanup function to close HTTP client, or None if mounting failed.
    """
    config = config or {}
    api_key = config.get("api_key") or os.environ.get("PERPLEXITY_API_KEY")

    if not api_key:
        logger.warning("No API key found for Perplexity provider")
        return None

    provider = PerplexityProvider(api_key, config, coordinator)
    await coordinator.mount("providers", provider, name="perplexity")
    logger.info("Mounted PerplexityProvider")

    async def cleanup() -> None:
        if provider._client is not None:
            await provider._client.aclose()

    return cleanup
