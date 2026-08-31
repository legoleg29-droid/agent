"""Tool abstraction.

Agents never call external APIs directly and never depend on a tool's
implementation details - they request a tool by id/name through the
:class:`~orchestrator.tools.runtime.ToolRuntime`, which enforces
permissions, validates arguments, executes with a timeout, and validates
output before handing a structured :class:`ToolResult` back. This is also
the seam where MCP-backed tools plug in and look identical to native ones -
see ``orchestrator/tools/mcp_adapter.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar


class ToolErrorCode(str, Enum):
    INVALID_ARGUMENTS = "invalid_arguments"
    UNAVAILABLE = "unavailable"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    MALFORMED_OUTPUT = "malformed_output"


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: str | None = None
    error_code: ToolErrorCode | None = None

    def to_public_dict(self) -> dict[str, Any]:
        """JSON-safe view suitable for feeding back to an LLM as a tool_result."""
        if self.success:
            return {"success": True, "output": self.output}
        return {"success": False, "error": self.error, "error_code": self.error_code.value if self.error_code else None}


class BaseTool(ABC):
    id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[dict[str, Any]] = {}
    output_schema: ClassVar[dict[str, Any]] = {}
    permissions: ClassVar[list[str]] = []
    capabilities: ClassVar[list[str]] = []
    timeout_seconds: ClassVar[float] = 30.0
    source: ClassVar[str] = "native"  # "native" | "mcp" | "api"

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "permissions": self.permissions,
            "capabilities": self.capabilities,
            "source": self.source,
        }

    def claude_schema(self) -> dict[str, Any]:
        """This tool's schema in Claude's native tool-use format."""
        return {
            "name": self.id,
            "description": self.description,
            "input_schema": self.input_schema or {"type": "object", "properties": {}},
        }
