"""Central registry of available tools.

Tools are looked up by ``id`` (unique). The registry is the single place
that knows what's available - it never cares whether a given tool is
native, MCP-backed, or a thin wrapper over a REST API (see
``orchestrator/tools/mcp_adapter.py``); agents only ever see the common
``BaseTool`` surface.
"""

from __future__ import annotations

from orchestrator.tools.base import BaseTool


class ToolNotFoundError(KeyError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.id] = tool

    def unregister(self, tool_id: str) -> None:
        self._tools.pop(tool_id, None)

    def get(self, tool_id: str) -> BaseTool:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise ToolNotFoundError(f"No tool registered with id '{tool_id}'") from exc

    def has(self, tool_id: str) -> bool:
        return tool_id in self._tools

    def is_available(self, tool_id: str) -> bool:
        """Validate that a tool id is currently registered and usable."""
        return tool_id in self._tools

    def list_tools(self, *, source: str | None = None) -> list[BaseTool]:
        tools = list(self._tools.values())
        if source is not None:
            tools = [t for t in tools if t.source == source]
        return tools

    def discover(self) -> list[dict]:
        """Full schema/metadata for every registered tool - what agents/
        planners are shown to decide what's usable."""
        return [t.describe() for t in self._tools.values()]

    def search_by_capability(self, capability: str) -> list[BaseTool]:
        return [t for t in self._tools.values() if capability in t.capabilities]

    def describe_all(self) -> list[dict]:
        return self.discover()

    def claude_schemas(self, tool_ids: list[str] | None = None) -> list[dict]:
        """Claude-native tool schemas, optionally restricted to ``tool_ids``
        (used to scope which tools a given agent may use)."""
        tools = self._tools.values() if tool_ids is None else (self._tools[i] for i in tool_ids if i in self._tools)
        return [t.claude_schema() for t in tools]
