"""Provider-agnostic LLM abstraction.

Agents and orchestrator components depend only on :class:`LLMProvider`.
Concrete providers (Claude, and later others) implement this interface so
the model backend can be swapped without touching orchestration logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


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
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Send a single-turn (or short multi-turn) completion request."""
        raise NotImplementedError
