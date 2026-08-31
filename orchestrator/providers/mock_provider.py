"""Deterministic offline provider used for tests and offline demo runs.

MockProvider never calls the network. It uses simple, inspectable rules
keyed off the system prompt / message content so orchestrator tests can
run fast and deterministically without an API key.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from orchestrator.providers.base import LLMMessage, LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    """Rule-based stand-in for a real LLM.

    ``responder`` receives ``(system, messages)`` and must return the text
    to respond with. If omitted, a generic canned response is returned.
    """

    name = "mock"

    def __init__(
        self,
        responder: Callable[[str, list[LLMMessage]], str] | None = None,
        model: str = "mock-model",
    ) -> None:
        self.responder = responder
        self.model = model
        self.calls: list[dict] = []

    async def complete(
        self,
        *,
        system: str,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append({"system": system, "messages": [m.content for m in messages]})
        if self.responder is not None:
            text = self.responder(system, messages)
        else:
            text = "Mock response: " + (messages[-1].content if messages else "")
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=sum(len(m.content) for m in messages) // 4,
            output_tokens=len(text) // 4,
            stop_reason="end_turn",
        )


def json_plan_responder(plan_obj: dict) -> Callable[[str, list[LLMMessage]], str]:
    """Build a responder that always returns a fixed plan as JSON."""

    def _respond(system: str, messages: list[LLMMessage]) -> str:
        return json.dumps(plan_obj)

    return _respond
