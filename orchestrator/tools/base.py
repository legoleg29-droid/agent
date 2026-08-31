"""Tool abstraction.

Agents never call external APIs directly - they request a tool by name
through the :class:`~orchestrator.tools.runtime.ToolRuntime`, which is the
single seam where a future MCP (Model Context Protocol) integration could
plug in additional tools without changing agent code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: str | None = None


class BaseTool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[dict[str, Any]] = {}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError
