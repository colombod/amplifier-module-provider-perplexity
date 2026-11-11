"""Buffered native Perplexity Agent Responses provider for Amplifier."""

import logging
import os
from collections.abc import Mapping
from copy import deepcopy
from types import TracebackType
from typing import Any, Self

import perplexity
from amplifier_core import ConfigField, ModelInfo, ModuleCoordinator, ProviderInfo
from amplifier_core.message_models import ChatRequest, ChatResponse, ToolCall
from amplifier_core.utils import redact_secrets
from perplexity import AsyncPerplexity

from ._agent_api import build_agent_request, convert_agent_response, request_body
from ._model_listing import ModelFilters, selector_schema

__all__ = ["PerplexityProvider", "mount"]

logger = logging.getLogger(__name__)


class PerplexityProvider:
    """Native Agent adapter; vendor search tools are not client tool calls."""

    name = "perplexity"
    api_label = "Perplexity"

    def __init__(
        self,
        api_key: str | None = None,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ) -> None:
        """Configure without creating a client; direct inference needs a key.

        mount() alone resolves environment credentials. Listing is static and
        does not imply account entitlement, capacity, or pricing guarantees.
        """
        self._api_key = api_key
        self._client: AsyncPerplexity | None = None
        self.config = config or {}
        self.coordinator = coordinator
        self._model_filters = ModelFilters.parse(
            self.config.get(
                "model_filters",
                {"include": ["agent-fast", "agent-low", "agent-medium", "agent-high"]},
            )
        )
        self.default_model: str = self.config.get("default_model", "agent-fast")
        self.max_tokens: int = self.config.get("max_tokens", 4096)
        self.max_steps: int = self.config.get("max_steps", 1)
        self.temperature: float | None = self.config.get("temperature")
        # Retain constructor-facing attributes; helper owns request validation.
        self.search_recency_filter: str | None = self.config.get(
            "search_recency_filter"
        )
        self.search_domain_filter: list[str] | None = self.config.get(
            "search_domain_filter"
        )
        self.search_context_size: str | None = self.config.get("search_context_size")
        self.return_images: bool = self.config.get("return_images", False)
        self.return_related_questions: bool = self.config.get(
            "return_related_questions", False
        )
        # Healthy generation has no implicit read/write deadline.
        self.timeout: float | perplexity.Timeout | None = self.config.get("timeout")
        self.max_retries: int = self.config.get("max_retries", 2)
        self.debug: bool = self.config.get("debug", False)
        self.raw: bool = self.config.get("raw", False)
        if type(self.debug) is not bool:
            raise ValueError("debug must be a boolean")
        if type(self.raw) is not bool:
            raise ValueError("raw must be a boolean")

    @property
    def client(self) -> AsyncPerplexity:
        """Lazily create and reuse the SDK client, preserving phase timeouts."""
        if self._client is None:
            if self._api_key is None:
                raise ValueError("api_key must be provided for API calls")
            self._client = AsyncPerplexity(
                api_key=self._api_key,
                timeout=(
                    self.timeout
                    if isinstance(self.timeout, perplexity.Timeout)
                    else perplexity.Timeout(self.timeout, connect=5.0, pool=5.0)
                ),
                max_retries=self.max_retries,
            )
        return self._client

    async def close(self) -> None:
        """Close the cached client; mounted use receives a cleanup callback."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> Self:
        """Return this provider for direct async-context-managed use."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close resources on context exit, without suppressing exceptions."""
        await self.close()

    def get_info(self) -> ProviderInfo:
        """Describe supported configuration, without client creation or I/O."""
        return ProviderInfo(
            id="perplexity",
            display_name="Perplexity AI",
            credential_env_vars=["PERPLEXITY_API_KEY"],
            capabilities=["citations", "web_search"],
            defaults={
                "default_model": self.default_model,
                "max_tokens": self.max_tokens,
                "max_steps": self.max_steps,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            },
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
                    id="search_type",
                    display_name="Search Type",
                    field_type="choice",
                    prompt="Inherit vendor search type or select fast/web",
                    choices=["inherit", "fast", "web"],
                    required=False,
                    default="inherit",
                ),
            ],
        )

    def get_model_list_info(self) -> dict[str, Any]:
        """Return fresh optional static-listing v1 metadata."""
        return {
            "version": 1,
            "source": "static",
            "filters_keyword": "filters",
            "filters_schema": selector_schema(),
            "default_filters": self._model_filters.as_dict(),
            "unfiltered": {},
            "override": "replace",
        }

    async def list_models(
        self, *, filters: Mapping[str, object] | None = None
    ) -> list[ModelInfo]:
        """Return fresh static rows; zero limits mean UNKNOWN, not usable zero.

        None uses the frozen configured policy; an explicit mapping replaces
        it, including {} for all eight rows. Exact inference is independent.
        """
        policy = self._model_filters if filters is None else ModelFilters.parse(filters)
        rows = (
            ("agent-fast", "Agent Fast"),
            ("agent-low", "Agent Low"),
            ("agent-medium", "Agent Medium"),
            ("agent-high", "Agent High"),
            ("sonar", "Sonar (Agent Fast alias)"),
            ("sonar-pro", "Sonar Pro (Agent Fast alias)"),
            ("sonar-reasoning-pro", "Sonar Reasoning Pro (Agent Low alias)"),
            ("sonar-deep-research", "Sonar Deep Research (Agent High alias)"),
        )
        models = [
            ModelInfo(
                id=model_id,
                display_name=display_name,
                context_window=0,
                max_output_tokens=0,
                capabilities=["web_search", "citations"],
                defaults={},
                pricing=None,
            )
            for model_id, display_name in rows
        ]
        selected = set(policy.select([model.id for model in models]))
        return [model for model in models if model.id in selected]

    async def complete(self, request: ChatRequest, **kwargs: Any) -> ChatResponse:
        """Validate and complete one buffered native Agent Responses request.

        Direct kwargs replace request_options by presence; the pure helper owns
        all option/history validation. Hooks are observations, not authorization.
        SDK errors and task cancellation propagate without a second retry layer.
        raw=True exposes full prompts with key-based (not text/PII) redaction.
        """
        request_options = kwargs.pop("request_options", None)
        if request_options is not None:
            if not isinstance(request_options, Mapping):
                raise ValueError("request_options must be a mapping or None")
            kwargs = {**request_options, **kwargs}
        route, built = build_agent_request(request, self.config, kwargs)
        # Separate wire, raw and caller graphs before awaiting any observer.
        params = deepcopy(built)
        event: dict[str, object] = {
            "provider": self.name,
            "model": route.selector,
            "messages_count": len(request.messages),
            "metadata": {"api": "agent", "effort": route.effort},
        }
        if self.raw:
            raw = request_body(built)
            if "timeout" in built:
                raw_timeout = built["timeout"]
                raw["timeout"] = (
                    raw_timeout.as_dict()
                    if isinstance(raw_timeout, perplexity.Timeout)
                    else raw_timeout
                )
            event["raw"] = redact_secrets(deepcopy(raw))
        if "timeout" in params:
            timeout = params["timeout"]
            if not isinstance(timeout, perplexity.Timeout):
                params["timeout"] = perplexity.Timeout(timeout, connect=5.0, pool=5.0)
        try:
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit("llm:request", deepcopy(event))
            response = await self.client.responses.create(**params)
            chat_response = convert_agent_response(response, route)
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                usage = chat_response.usage
                event_usage: dict[str, int] = {
                    "input_tokens": usage.input_tokens if usage else 0,
                    "output_tokens": usage.output_tokens if usage else 0,
                }
                if usage:
                    if usage.cache_read_tokens is not None:
                        event_usage["cache_read_tokens"] = usage.cache_read_tokens
                    if usage.cache_write_tokens is not None:
                        event_usage["cache_write_tokens"] = usage.cache_write_tokens
                metadata = chat_response.metadata or {}
                await self.coordinator.hooks.emit(
                    "llm:response",
                    deepcopy(
                        {
                            "provider": self.name,
                            "model": metadata.get("perplexity:model"),
                            "requested_selector": route.selector,
                            "usage": event_usage,
                            "metadata": {
                                "api": "agent",
                                "status": metadata["perplexity:status"],
                            },
                        }
                    ),
                )
            return chat_response
        except perplexity.RateLimitError as error:
            logger.warning(
                "Perplexity rate limit: %s status=%d",
                type(error).__name__,
                error.status_code,
            )
            raise
        except perplexity.APIStatusError as error:
            logger.error(
                "Perplexity API error: %s status=%d",
                type(error).__name__,
                error.status_code,
            )
            raise
        except perplexity.APIConnectionError as error:
            logger.error("Perplexity connection error: %s", type(error).__name__)
            raise

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """Return no client calls; vendor-managed server search is not dispatched."""
        return []


async def mount(
    coordinator: ModuleCoordinator, config: dict[str, Any] | None = None
) -> Any:
    """Mount through Core, preferring explicit credentials, and return cleanup."""
    config = config or {}
    api_key = config.get("api_key") or os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        logger.warning(
            "Perplexity provider not mounted: PERPLEXITY_API_KEY not set. "
            "Set the environment variable or provide api_key in config."
        )
        return None
    provider = PerplexityProvider(
        api_key=api_key, config=config, coordinator=coordinator
    )
    await coordinator.mount("providers", provider, name=provider.name)
    logger.info("Perplexity provider mounted")

    async def cleanup() -> None:
        await provider.close()
        logger.debug("Perplexity provider cleanup complete")

    return cleanup
