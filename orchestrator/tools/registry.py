"""Registry of available tools."""

from __future__ import annotations

from orchestrator.tools.base import BaseTool


class ToolNotFoundError(KeyError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"No tool registered with name '{name}'") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def describe_all(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self._tools.values()
        ]
