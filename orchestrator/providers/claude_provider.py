"""Claude (Anthropic) implementation of the LLMProvider interface.

Uses Claude's native tool-use mechanism: when ``tools`` are passed to
``complete()``, they're forwarded to the Messages API as-is, and any
``tool_use`` content blocks in the response are surfaced as structured
``ToolCallRequest`` objects rather than requiring the caller to parse text.
"""

from __future__ import annotations

import os
from typing import Any

from orchestrator.providers.base import LLMMessage, LLMProvider, LLMResponse, ToolCallRequest

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


class ClaudeProvider(LLMProvider):
    """LLM provider backed by the Anthropic Messages API.

    Requires the ``anthropic`` package and an ``ANTHROPIC_API_KEY``
    environment variable (or an explicit ``api_key`` argument).
    """

    name = "claude"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "The 'anthropic' package is required to use ClaudeProvider. "
                "Install it with `pip install anthropic`."
            ) from exc

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Provide it via the environment "
                "or the api_key argument. See .env.example."
            )

        self._client = AsyncAnthropic(api_key=resolved_key)
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature

    async def complete(
        self,
        *,
        system: str,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            system=system,
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature if temperature is not None else self.default_temperature,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        if tools:
            kwargs["tools"] = tools

        response = await self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        content_blocks: list[dict[str, Any]] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
                content_blocks.append({"type": "text", "text": block.text})
            elif block_type == "tool_use":
                tool_calls.append(ToolCallRequest(id=block.id, name=block.name, arguments=dict(block.input)))
                content_blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})

        return LLMResponse(
            text="".join(text_parts),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            tool_calls=tool_calls,
            content_blocks=content_blocks,
            raw=response,
        )
