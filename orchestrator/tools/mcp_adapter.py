"""MCP (Model Context Protocol) adapter.

Wraps tools discovered from an external MCP server so they present the
exact same ``BaseTool`` surface as native tools - permissions, schemas,
timeouts, and the ``ToolRuntime`` validate/execute/observe flow all apply
identically. Agents and the orchestrator never know (or need to know)
whether a given tool id is native, MCP-backed, or a REST API wrapper.

This module defines the *shape* of an MCP client (``MCPClient``) rather
than a concrete transport, since wiring up the real MCP wire protocol is
outside this repository's scope; ``register_mcp_server`` is the piece that
makes discovery automatic - point it at any object satisfying the
``MCPClient`` protocol (a real MCP client, or a test double) and its tools
appear in the registry with zero hardcoding of individual tool names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from orchestrator.tools.base import BaseTool, ToolErrorCode, ToolResult
from orchestrator.tools.registry import ToolRegistry

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


@dataclass
class MCPToolDescriptor:
    """What an MCP server reports about one of its tools."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    permissions: list[str] = field(default_factory=list)


class MCPClient(Protocol):
    """Shape a real MCP client (or a test double) must satisfy.

    Deliberately minimal: list the tools a server exposes, and invoke one
    by name. A concrete implementation would speak the actual MCP wire
    protocol (stdio/SSE JSON-RPC) underneath these two methods.
    """

    async def list_tools(self) -> list[MCPToolDescriptor]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


def _claude_safe_id(server_name: str, tool_name: str) -> str:
    """Build a registry/Claude-safe id, namespaced by server to avoid collisions."""
    safe_server = _SAFE_NAME_RE.sub("_", server_name)
    safe_tool = _SAFE_NAME_RE.sub("_", tool_name)
    return f"mcp__{safe_server}__{safe_tool}"


class MCPToolAdapter(BaseTool):
    """A single MCP-exposed tool, wrapped to look like a native ``BaseTool``."""

    source = "mcp"

    def __init__(
        self,
        *,
        server_name: str,
        descriptor: MCPToolDescriptor,
        client: MCPClient,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.server_name = server_name
        self.id = _claude_safe_id(server_name, descriptor.name)
        self.name = descriptor.name
        self.description = f"[MCP:{server_name}] {descriptor.description}"
        self.input_schema = descriptor.input_schema
        self.output_schema = descriptor.output_schema
        self.permissions = descriptor.permissions
        self.capabilities = ["mcp", server_name]
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._remote_name = descriptor.name

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            output = await self._client.call_tool(self._remote_name, kwargs)
        except Exception as exc:  # noqa: BLE001 - never let a transport error crash the runtime
            return ToolResult(
                success=False,
                error=f"MCP tool '{self._remote_name}' on server '{self.server_name}' failed: {exc}",
                error_code=ToolErrorCode.EXECUTION_ERROR,
            )
        return ToolResult(success=True, output=output)


async def register_mcp_server(
    registry: ToolRegistry,
    client: MCPClient,
    server_name: str,
    *,
    timeout_seconds: float = 30.0,
) -> list[str]:
    """Discover every tool an MCP server exposes and register it.

    Returns the list of registered tool ids. Nothing about individual
    tools is hardcoded here - whatever the server reports via
    ``list_tools()`` becomes available through the registry automatically.
    """
    descriptors = await client.list_tools()
    registered_ids: list[str] = []
    for descriptor in descriptors:
        adapter = MCPToolAdapter(
            server_name=server_name,
            descriptor=descriptor,
            client=client,
            timeout_seconds=timeout_seconds,
        )
        registry.register(adapter)
        registered_ids.append(adapter.id)
    return registered_ids
