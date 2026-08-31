"""Provider-agnostic LLM abstraction.

Agents and orchestrator components depend only on :class:`LLMProvider`.
Concrete providers (Claude, and later others) implement this interface so
the model backend can be swapped without touching orchestration logic.

Tool use is first-class here: a provider may be given ``tools`` (schemas)
on a completion request and may respond with structured ``tool_calls``
instead of (or alongside) text, mirroring Claude's native tool-use
mechanism. Agents drive the tool loop by feeding tool results back as
messages - see ``orchestrator/agents/_common.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Union

# A message's content is either plain text, or a list of content blocks in
# Anthropic's block format (text / tool_use / tool_result), which is what's
# required to carry a tool-use round trip. Kept provider-agnostic in shape
# (plain dicts) so a different provider can adapt it without this module
# depending on any vendor SDK.
MessageContent = Union[str, list[dict[str, Any]]]


@dataclass
class LLMMessage:
    role: str  # "user" | "assistant"
    content: MessageContent


@dataclass
class ToolCallRequest:
    """A structured tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    content_blocks: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def requested_tool_use(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ProviderConfig:
    model: str
    max_tokens: int = 2048
    temperature: float = 0.2
    extra: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract LLM provider. All model access goes through this interface."""

    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Send a completion request.

        ``tools`` are provider-native tool schemas (for Claude:
        ``[{"name", "description", "input_schema"}, ...]``). When the model
        decides to use one, the response carries it in ``tool_calls``
        instead of forcing the caller to parse free text.
        """
        raise NotImplementedError
