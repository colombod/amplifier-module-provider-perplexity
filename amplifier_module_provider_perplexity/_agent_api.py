"""Pure native Agent API adaptation, without transport or host effects."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from typing import Literal, NotRequired, TypedDict

from amplifier_core.llm_errors import LLMError
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    Message,
    TextBlock,
    Usage,
)
from perplexity import Timeout
from perplexity.generated.api import InputItem as InputItemParam
from perplexity.generated.api import ResponseCreateParams, ResponseCreateResponse
from pydantic import ValidationError

AgentEffort = Literal["fast", "low", "medium", "high"]


@dataclass(frozen=True)
class AgentRoute:
    """Resolved selector: exactly one managed effort or exact native model."""

    selector: str
    effort: AgentEffort | None
    model: str | None


class AgentParams(TypedDict):
    """Buffered SDK create arguments; timeout remains a transport-only override."""

    input: str | list[InputItemParam]
    max_output_tokens: int
    max_steps: int
    stream: Literal[False]
    background: Literal[False]
    store: Literal[False]
    model: NotRequired[str]
    effort: NotRequired[str]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    tools: NotRequired[list[Mapping[str, object]]]
    timeout: NotRequired[float | Timeout | None]


_EFFORTS: dict[str, AgentEffort] = {
    "agent-fast": "fast",
    "agent-low": "low",
    "agent-medium": "medium",
    "agent-high": "high",
    "sonar": "fast",
    "sonar-pro": "fast",
    "sonar-reasoning-pro": "low",
    "sonar-deep-research": "high",
}
_ESCAPE_HATCHES = (
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
)


def resolve_route(selector: str) -> AgentRoute:
    """Resolve case-sensitive aliases or preserve an exact ID, without listing.

    Raises ValueError for blank, padded, or unresolved glob selectors.
    """
    if (
        not isinstance(selector, str)
        or not selector
        or selector != selector.strip()
        or any(char in selector for char in "*?[]")
    ):
        raise ValueError(
            "model must be an exact nonblank selector without padding or globs"
        )
    effort = _EFFORTS.get(selector)
    return AgentRoute(selector, effort, selector if effort is None else None)


def convert_messages(messages: list[Message]) -> list[InputItemParam]:
    """Preserve native roles/order/empty text; reject unsupported history.

    Only Core text blocks are joined with newlines. SDK validation errors become
    a constant ValueError so their input diagnostics cannot expose prompt text.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    items: list[dict[str, str]] = []
    for message in messages:
        if message.role not in ("system", "developer", "user", "assistant"):
            raise ValueError("message role is unsupported by the Agent adapter")
        if isinstance(message.content, str):
            text = message.content
        else:
            parts: list[str] = []
            for block in message.content:
                match block:
                    case TextBlock(text=block_text):
                        parts.append(block_text)
                    case _:
                        raise ValueError(
                            "message content must contain only Core text blocks"
                        )
            text = "\n".join(parts)
        items.append({"type": "message", "role": message.role, "content": text})
    try:
        converted = ResponseCreateParams.model_validate({"input": items}).input
    except ValidationError:
        raise ValueError("message content is invalid for the Agent API") from None
    match converted:
        case list() as input_items:
            return input_items
        case _:
            raise ValueError("message input must be an array")


def _first_non_none(*values: object) -> object:
    """Select generation options without treating zero as missing."""
    return next((value for value in values if value is not None), None)


def _positive_int(value: object, option: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(
            f"{option} must be a positive integer within its supported range"
        )
    return value


def _generation_number(value: object, option: str, maximum: float) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= maximum
        or not isfinite(value)
    ):
        raise ValueError(f"{option} must be finite and within its supported range")
    return float(value)


def build_agent_request(
    request: ChatRequest,
    config: Mapping[str, object],
    options: Mapping[str, object],
) -> tuple[AgentRoute, AgentParams]:
    """Validate Core/effective options and build a pure native SDK request.

    The caller merges request_options and direct kwargs before calling. Unknown
    host bookkeeping is ignored, never forwarded. No client, hook, or transport
    is accessed. Raises ValueError for unsupported or invalid request intent.
    """
    if "max_output_tokens" in options:
        raise ValueError(
            "max_output_tokens keyword is unsupported; use "
            "ChatRequest.max_output_tokens or max_tokens"
        )
    for key in _ESCAPE_HATCHES:
        if key in options:
            raise ValueError(f"{key} is an unsupported native request override")
    if request.tools:
        raise ValueError("Core tools are unsupported by the Agent adapter")
    if request.tool_choice is not None:
        raise ValueError("Core tool_choice is unsupported by the Agent adapter")
    if request.stop:
        raise ValueError("Core stop sequences are unsupported by the Agent adapter")
    if request.response_format is not None and request.response_format.type != "text":
        raise ValueError("Core response_format supports only text in the Agent adapter")
    if request.stream not in (None, False) or (
        "stream" in options and options["stream"] is not False
    ):
        raise ValueError("stream must be False for the buffered Agent adapter")
    if (
        request.reasoning_effort is not None
        or options.get("reasoning_effort", config.get("reasoning_effort")) is not None
    ):
        raise ValueError("reasoning_effort is unsupported; use an Agent model selector")

    selector = _first_non_none(
        options.get("model"), request.model, config.get("default_model", "agent-fast")
    )
    match selector:
        case str() as model_selector:
            route = resolve_route(model_selector)
        case _:
            raise ValueError("model must be an exact nonblank selector")
    budget = _positive_int(
        _first_non_none(
            options.get("max_tokens"),
            request.max_output_tokens,
            config.get("max_tokens"),
            4096,
        ),
        "max_tokens",
    )
    steps = _positive_int(
        options.get("max_steps", config.get("max_steps", 1)), "max_steps", 500
    )
    temperature = _generation_number(
        _first_non_none(
            options.get("temperature"), request.temperature, config.get("temperature")
        ),
        "temperature",
        2,
    )
    top_p = _generation_number(
        _first_non_none(options.get("top_p"), request.top_p, config.get("top_p")),
        "top_p",
        1,
    )
    search = options.get("web_search", config.get("web_search", True))
    if type(search) is not bool:
        raise ValueError("web_search must be a boolean")
    domains = options.get("search_domain_filter", config.get("search_domain_filter"))
    copied_domains: list[str] = []
    if domains is not None:
        if not isinstance(domains, list) or len(domains) > 20:
            raise ValueError(
                "search_domain_filter must be a list of at most 20 domains or None"
            )
        for domain in domains:
            if (
                not isinstance(domain, str)
                or not domain.strip()
                or (domain.startswith("-") and not domain[1:].strip())
            ):
                raise ValueError("search_domain_filter must contain nonblank domains")
            copied_domains.append(domain)
        if len({domain.startswith("-") for domain in copied_domains}) > 1:
            raise ValueError(
                "search_domain_filter cannot mix inclusions and exclusions"
            )
    recency = options.get("search_recency_filter", config.get("search_recency_filter"))
    if recency == "none":
        recency = None
    if recency is not None and (
        not isinstance(recency, str)
        or recency not in ("hour", "day", "week", "month", "year")
    ):
        raise ValueError(
            "search_recency_filter must be hour, day, week, month, year, none, or None"
        )
    configured_type = config.get("search_type")
    if configured_type == "inherit":
        configured_type = None
    search_type = options.get("search_type", configured_type)
    if search_type is not None and (
        not isinstance(search_type, str) or search_type not in ("web", "fast")
    ):
        raise ValueError("search_type must be web, fast, or None")
    search_budgets: dict[str, int] = {}
    for name, native in (
        ("search_max_tokens", "max_tokens"),
        ("search_max_tokens_per_page", "max_tokens_per_page"),
    ):
        value = options.get(name, config.get(name))
        if value is not None:
            search_budgets[native] = _positive_int(value, name)
    if (
        options.get("search_context_size", config.get("search_context_size"))
        is not None
    ):
        raise ValueError(
            "search_context_size is unsupported; use "
            "search_max_tokens or search_max_tokens_per_page"
        )
    for name in ("return_images", "return_related_questions"):
        value = options.get(name, config.get(name, False))
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
        if value:
            raise ValueError(
                f"{name} is unsupported by the Agent adapter; "
                "consume supported returned metadata instead"
            )
    if not search:
        if route.effort is not None:
            raise ValueError(
                "web_search cannot be disabled for a managed Agent selector"
            )
        if (
            copied_domains
            or recency is not None
            or search_type is not None
            or search_budgets
        ):
            raise ValueError("web_search=False cannot apply active search options")

    params: AgentParams = {
        "input": convert_messages(request.messages),
        "max_output_tokens": budget,
        "max_steps": steps,
        "stream": False,
        "background": False,
        "store": False,
    }
    if route.effort is not None:
        params["effort"] = route.effort
    elif route.model is not None:
        params["model"] = route.model
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if search:
        tool: dict[str, object] = {"type": "web_search"}
        filters: dict[str, object] = {}
        if copied_domains:
            filters["search_domain_filter"] = copied_domains
        if recency is not None:
            filters["search_recency_filter"] = recency
        if filters:
            tool["filters"] = filters
        if search_type is not None:
            tool["search_type"] = search_type
        tool.update(search_budgets)
        params["tools"] = [tool]
    if "timeout" in options:
        timeout = options["timeout"]
        if timeout is not None and not isinstance(timeout, (int, float, Timeout)):
            raise ValueError("timeout must be a scalar, Timeout, or None")
        params["timeout"] = timeout
    return route, params


def request_body(params: AgentParams) -> dict[str, object]:
    """Return a separate deep native JSON snapshot, excluding transport timeout."""
    body = {key: value for key, value in params.items() if key != "timeout"}
    return ResponseCreateParams.model_validate(body).model_dump(
        mode="json", by_alias=True, exclude_unset=True
    )


def _response_error(message: str = "returned an invalid response") -> LLMError:
    return LLMError(
        f"Perplexity Agent API {message}.",
        provider="perplexity",
        retryable=False,
        status_code=200,
    )


def _response_mapping(value: object) -> Mapping[str, object] | None:
    """Check SDK JSON object boundaries, without propagating SDK union/Any types."""
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: item for key, item in value.items() if isinstance(key, str)}
    return None


def _response_state(status: object) -> str:
    """Reject unsafe states before any partial response can reach the host."""
    if status == "failed":
        raise _response_error("returned an error response") from None
    if status == "cancelled":
        raise _response_error("response was cancelled") from None
    if status in ("queued", "in_progress", "requires_action"):
        raise _response_error("returned an unsupported response state") from None
    if not isinstance(status, str) or status not in ("completed", "incomplete"):
        raise _response_error() from None
    return status


def _response_count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _cost_number(value: object) -> int | float | None:
    if type(value) is int and value >= 0:
        return value
    if type(value) is float and isfinite(value) and value >= 0:
        return value
    return None


def _response_cost(value: object, metadata: dict[str, object]) -> Decimal | None:
    """Keep documented JSON cost scalars; Decimal is exclusively Core Usage data."""
    if value is None:
        metadata["perplexity:usage_cost_status"] = "missing"
        return None
    cost = _response_mapping(value)
    metadata["perplexity:usage_cost_status"] = "invalid"
    if cost is None:
        return None
    kept: dict[str, object] = {}
    currency = cost.get("currency")
    if isinstance(currency, str) and currency.strip():
        kept["currency"] = currency
    for field in (
        "input_cost",
        "output_cost",
        "total_cost",
        "cache_creation_cost",
        "cache_read_cost",
        "tool_calls_cost",
    ):
        number = _cost_number(cost.get(field))
        if number is not None:
            kept[field] = number
    metadata["perplexity:usage_cost"] = kept
    total = _cost_number(cost.get("total_cost"))
    if total is None or not isinstance(currency, str) or not currency.strip():
        return None
    if currency != "USD":
        metadata["perplexity:usage_cost_status"] = "non_usd"
        return None
    metadata["perplexity:usage_cost_status"] = "known"
    # Decimal spelling of the SDK-exposed value, not lossless wire precision.
    return Decimal(str(total))


def _response_usage(value: object, metadata: dict[str, object]) -> Usage | None:
    if value is None:
        metadata["perplexity:usage_cost_status"] = "missing"
        return None
    usage = _response_mapping(value)
    if usage is None:
        raise _response_error() from None
    input_tokens = _response_count(usage.get("input_tokens"))
    output_tokens = _response_count(usage.get("output_tokens"))
    total_tokens = _response_count(usage.get("total_tokens"))
    if input_tokens is None or output_tokens is None or total_tokens is None:
        raise _response_error() from None
    input_details = _response_mapping(usage.get("input_tokens_details")) or {}
    output_details = _response_mapping(usage.get("output_tokens_details")) or {}
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_read_tokens=_response_count(input_details.get("cache_read_input_tokens")),
        cache_write_tokens=_response_count(
            input_details.get("cache_creation_input_tokens")
        ),
        reasoning_tokens=_response_count(output_details.get("reasoning_tokens")),
        cost_usd=_response_cost(usage.get("cost"), metadata),
    )


def _search_metadata(
    item: Mapping[str, object],
    output_index: int,
    results: list[dict[str, object]],
    envelopes: list[dict[str, object]],
) -> None:
    """Preserve native source IDs and output correlation, never citation positions."""
    returned = item.get("results")
    if not isinstance(returned, list):
        return
    native_results: list[dict[str, object]] = []
    for value in returned:
        source = _response_mapping(value)
        if source is None:
            continue
        kept: dict[str, object] = {}
        native_id = source.get("id")
        if type(native_id) is int:
            kept["id"] = native_id
        for field in ("url", "title", "snippet", "date", "last_updated", "source"):
            scalar = source.get(field)
            if isinstance(scalar, str) or (field in source and scalar is None):
                kept[field] = scalar
        native_results.append(kept)
    envelope: dict[str, object] = {
        "output_index": output_index,
        "type": "search_results",
        "results": native_results,
    }
    queries = item.get("queries")
    if isinstance(queries, list):
        envelope["queries"] = [query for query in queries if isinstance(query, str)]
    for field in ("id", "call_id", "tool_call_id"):
        scalar = item.get(field)
        if isinstance(scalar, str):
            envelope[field] = scalar
    results.extend(native_results)
    envelopes.append(envelope)


def _annotation_metadata(
    returned: object,
    output_index: int,
    content_index: int,
    annotations: list[dict[str, object]],
    citations: list[str],
) -> None:
    if not isinstance(returned, list):
        return
    kept_annotations: list[dict[str, object]] = []
    for value in returned:
        annotation = _response_mapping(value)
        if annotation is None:
            continue
        kept: dict[str, object] = {}
        for field in ("type", "url", "title"):
            scalar = annotation.get(field)
            if isinstance(scalar, str) or (field in annotation and scalar is None):
                kept[field] = scalar
        for field in ("start_index", "end_index"):
            index = annotation.get(field)
            if type(index) is int or (field in annotation and index is None):
                kept[field] = index
        kept_annotations.append(kept)
        url = kept.get("url")
        if kept.get("type") == "url_citation" and isinstance(url, str):
            citations.append(url)
    annotations.append(
        {
            "output_index": output_index,
            "content_index": content_index,
            "annotations": kept_annotations,
        }
    )


def convert_agent_response(
    response: ResponseCreateResponse, route: AgentRoute
) -> ChatResponse:
    """Adapt one buffered native response into ordinary Core text/usage/metadata.

    Uses the released SDK JSON dump once, then checks concrete nested shapes.
    State/envelope/required-usage faults raise fixed nonretryable Core LLMError
    without vendor diagnostics or exception chaining. Only assistant output_text
    becomes answer text; sources/annotations remain untrusted namespaced data.
    No transport, client, hook, tool dispatch, logging or continuation occurs.
    """
    try:
        dumped: object = response.model_dump(
            mode="json", exclude_unset=True, warnings=False
        )
    except (ValueError, TypeError):
        raise _response_error() from None
    data = _response_mapping(dumped)
    if data is None:
        raise _response_error() from None
    if data.get("error") is not None:
        raise _response_error("returned an error response") from None
    status = _response_state(data.get("status"))
    response_id = data.get("id")
    output = data.get("output")
    if (
        not isinstance(response_id, str)
        or data.get("object") != "response"
        or _response_count(data.get("created_at")) is None
        or not isinstance(output, list)
    ):
        raise _response_error() from None
    model = data.get("model")
    metadata: dict[str, object] = {
        "perplexity:response_id": response_id,
        "perplexity:model": model if isinstance(model, str) and model.strip() else None,
        "perplexity:requested_selector": route.selector,
        "perplexity:status": status,
    }
    if route.effort is not None:
        metadata["perplexity:resolved_effort"] = route.effort
    effort = data.get("effort")
    if isinstance(effort, str):
        metadata["perplexity:effort"] = effort
    finish_reason = "stop" if status == "completed" else None
    if status == "incomplete":
        details = _response_mapping(data.get("incomplete_details"))
        reason = details.get("reason") if details is not None else None
        if isinstance(reason, str):
            metadata["perplexity:incomplete_details"] = {"reason": reason}
            if reason == "max_output_tokens":
                finish_reason = "length"

    text: list[str] = []
    results: list[dict[str, object]] = []
    envelopes: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    citations: list[str] = []
    for output_index, value in enumerate(output):
        item = _response_mapping(value)
        if item is None or not isinstance(item.get("type"), str):
            raise _response_error() from None
        kind = item["type"]
        if kind in ("function_call", "mcp_approval_request"):
            raise _response_error("returned an unsupported response state") from None
        if kind == "search_results":
            _search_metadata(item, output_index, results, envelopes)
        if kind != "message" or item.get("role") != "assistant":
            continue
        _response_state(item.get("status"))
        content = item.get("content")
        if not isinstance(content, list):
            raise _response_error() from None
        for content_index, content_value in enumerate(content):
            part = _response_mapping(content_value)
            if part is None or part.get("type") != "output_text":
                continue
            part_text = part.get("text")
            if not isinstance(part_text, str):
                continue
            text.append(part_text)
            _annotation_metadata(
                part.get("annotations"),
                output_index,
                content_index,
                annotations,
                citations,
            )
    if envelopes:
        metadata["perplexity:search_results"] = results
        metadata["perplexity:search_result_items"] = envelopes
    if annotations:
        metadata["perplexity:annotations"] = annotations
        metadata["perplexity:citations"] = citations
    usage = _response_usage(data.get("usage"), metadata)
    return ChatResponse(
        content=[TextBlock(text="".join(text))],
        usage=usage,
        finish_reason=finish_reason,
        metadata=metadata,
    )
