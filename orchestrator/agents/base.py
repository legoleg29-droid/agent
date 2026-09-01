"""BaseAgent interface.

Every agent - built-in or user-defined - implements this interface. The
orchestrator and router only ever talk to agents through it, never through
concrete subclasses, which is what lets new agents be added without
touching orchestration code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from orchestrator.providers.base import LLMProvider
from orchestrator.tools.runtime import ToolRuntime


@dataclass
class AgentInput:
    """What a task hands to an agent. Deliberately narrow, not the full history."""

    objective: str
    expected_output: str = ""
    task_context: dict[str, Any] = field(default_factory=dict)
    upstream_outputs: dict[str, Any] = field(default_factory=dict)
    memory_context: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    # Set only for a repair re-invocation (see orchestrator/core/repair.py):
    # previous output, exactly which acceptance criteria failed and why, so
    # the agent can focus on fixing that instead of starting over blind.
    repair_feedback: dict[str, Any] | None = None


@dataclass
class AgentOutput:
    success: bool
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    tool_calls: int = 0
    tokens_used: int = 0
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "content": self.content,
            "data": self.data,
            "error": self.error,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentOutput:
        return cls(
            success=data["success"],
            content=data.get("content", ""),
            data=data.get("data", {}) or {},
            error=data.get("error"),
            tool_calls=data.get("tool_calls", 0),
            tokens_used=data.get("tokens_used", 0),
            model=data.get("model"),
        )


class BaseAgent(ABC):
    id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    capabilities: ClassVar[list[str]] = []
    available_tools: ClassVar[list[str]] = []
    # Permissions this agent holds (e.g. "filesystem.read"). The ToolRuntime
    # checks these against a tool's declared required permissions - an
    # agent listing a tool in ``available_tools`` without the matching
    # permission will have its tool calls rejected at execution time, not
    # just hidden from the prompt.
    permissions: ClassVar[list[str]] = []
    system_instructions: ClassVar[str] = ""

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "objective": {"type": "string"},
            "expected_output": {"type": "string"},
            "task_context": {"type": "object"},
            "upstream_outputs": {"type": "object"},
        },
        "required": ["objective"],
    }
    output_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "content": {"type": "string"},
            "data": {"type": "object"},
            "error": {"type": ["string", "null"]},
        },
        "required": ["success", "content"],
    }

    def __init__(self, provider: LLMProvider, tool_runtime: ToolRuntime | None = None) -> None:
        self.provider = provider
        self.tool_runtime = tool_runtime

    @abstractmethod
    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "available_tools": self.available_tools,
            "permissions": self.permissions,
        }
