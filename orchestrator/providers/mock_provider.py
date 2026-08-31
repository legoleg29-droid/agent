"""Deterministic offline provider used for tests and offline demo runs.

MockProvider never calls the network. It uses simple, inspectable rules
keyed off the system prompt / message content so orchestrator tests can
run fast and deterministically without an API key.

To exercise the structured tool-use loop offline, a responder may return a
:class:`ScriptedToolUse` instead of plain text - MockProvider turns that
into the same shape of ``LLMResponse`` (with ``tool_calls`` and
``content_blocks``) that ``ClaudeProvider`` would produce for a real
``tool_use`` response, so agent code never has to special-case "mock vs
real".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from orchestrator.providers.base import LLMMessage, LLMProvider, LLMResponse, ToolCallRequest

Responder = Callable[[str, list[LLMMessage], "list[dict[str, Any]] | None"], Any]


@dataclass
class ScriptedToolUse:
    """A responder returns this to make MockProvider emit a tool_use turn."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    id: str | None = None


class MockProvider(LLMProvider):
    """Rule-based stand-in for a real LLM.

    ``responder(system, messages, tools)`` may return:
    - a ``str``: treated as the final text response.
    - a :class:`ScriptedToolUse`: emitted as a structured tool-use response.
    - an ``LLMResponse``: returned as-is, for full control.
    """

    name = "mock"

    def __init__(self, responder: Responder | None = None, model: str = "mock-model") -> None:
        self.responder = responder
        self.model = model
        self.calls: list[dict[str, Any]] = []
        self._tool_call_counter = 0

    async def complete(
        self,
        *,
        system: str,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "messages": [m.content for m in messages],
                "tool_names": [t.get("name") for t in (tools or [])],
            }
        )

        if self.responder is not None:
            result = self.responder(system, messages, tools)
        else:
            last = messages[-1].content if messages else ""
            result = "Mock response: " + (last if isinstance(last, str) else json.dumps(last))

        if isinstance(result, LLMResponse):
            return result

        if isinstance(result, ScriptedToolUse):
            self._tool_call_counter += 1
            call_id = result.id or f"mock_tool_{self._tool_call_counter}"
            content_blocks: list[dict[str, Any]] = []
            if result.text:
                content_blocks.append({"type": "text", "text": result.text})
            content_blocks.append(
                {"type": "tool_use", "id": call_id, "name": result.name, "input": result.arguments}
            )
            return LLMResponse(
                text=result.text,
                model=self.model,
                input_tokens=10,
                output_tokens=10,
                stop_reason="tool_use",
                tool_calls=[ToolCallRequest(id=call_id, name=result.name, arguments=result.arguments)],
                content_blocks=content_blocks,
            )

        text = str(result)
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=sum(len(m.content) for m in messages if isinstance(m.content, str)) // 4,
            output_tokens=len(text) // 4,
            stop_reason="end_turn",
            content_blocks=[{"type": "text", "text": text}],
        )
