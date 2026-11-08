"""Exercise typed Core requests and registered coordinator hooks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import perplexity
import pytest
from amplifier_core import ChatRequest, Message, MockCoordinator

from amplifier_module_provider_perplexity import PerplexityProvider


@pytest.mark.asyncio
async def test_typed_request_output_bound_and_hooks():
    coordinator = MockCoordinator()
    event_spy = AsyncMock(return_value={"action": "continue"})
    coordinator.hooks.register("llm:request", event_spy)
    coordinator.hooks.register("llm:response", event_spy)
    p = PerplexityProvider(
        api_key="test-key",
        config={"raw": True},
        coordinator=coordinator,
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            id="test-response",
            model="sonar-pro",
            citations=[],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="partial"), finish_reason="length"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=10),
        )
    )
    try:
        p.client.chat.completions.create = create
        result = await p.complete(
            ChatRequest(
                model="sonar-pro",
                max_output_tokens=100,
                temperature=0,
                messages=[Message(role="user", content="Question")],
            )
        )
        assert create.call_args.kwargs["max_tokens"] == 100
        assert create.call_args.kwargs["temperature"] == 0
        usage = result.usage
        metadata = result.metadata
        assert usage is not None
        assert metadata is not None
        assert result.finish_reason == "length" and usage.total_tokens == 40
        assert metadata["perplexity:response_id"] == "test-response"
        assert [c.args[0] for c in event_spy.call_args_list] == [
            "llm:request",
            "llm:response",
        ]
        event = event_spy.call_args_list[0].args[1]
        assert event["raw"]["messages"] == create.call_args.kwargs["messages"]
        assert "api_key" not in event["raw"]
        timeout = p.client.timeout
        assert isinstance(timeout, perplexity.Timeout)
        assert timeout.read is None and timeout.connect == 5
    finally:
        await p.close()
