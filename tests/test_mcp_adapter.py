"""Tests for the MCP adapter abstraction.

Uses an in-memory fake MCP client (a test double for a real MCP server
connection) to verify that: (1) tools discovered from a server register
automatically with zero hardcoding of tool names, and (2) once registered,
an MCP-backed tool is indistinguishable from a native tool to the
ToolRuntime/agents - same call flow, same permission enforcement.
"""

import pytest

from orchestrator.core.logging_utils import EventLog
from orchestrator.tools.mcp_adapter import MCPToolDescriptor, register_mcp_server
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.runtime import ToolRuntime


class FakeMCPClient:
    """In-memory stand-in for a real MCP server connection."""

    def __init__(self, descriptors: list[MCPToolDescriptor], responses: dict[str, object]):
        self._descriptors = descriptors
        self._responses = responses
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self) -> list[MCPToolDescriptor]:
        return self._descriptors

    async def call_tool(self, name: str, arguments: dict) -> object:
        self.calls.append((name, arguments))
        if name not in self._responses:
            raise RuntimeError(f"no fake response configured for '{name}'")
        return self._responses[name]


@pytest.mark.asyncio
async def test_mcp_tools_auto_register_with_no_hardcoded_names():
    descriptors = [
        MCPToolDescriptor(name="lookup_weather", description="Look up the weather", input_schema={"type": "object", "properties": {"city": {"type": "string"}}}),
        MCPToolDescriptor(name="translate", description="Translate text", input_schema={"type": "object", "properties": {"text": {"type": "string"}}}),
    ]
    client = FakeMCPClient(descriptors, responses={"lookup_weather": {"temp_c": 21}})
    registry = ToolRegistry()

    registered_ids = await register_mcp_server(registry, client, server_name="demo_server")

    assert registered_ids == ["mcp__demo_server__lookup_weather", "mcp__demo_server__translate"]
    assert registry.has("mcp__demo_server__lookup_weather")
    assert registry.has("mcp__demo_server__translate")


@pytest.mark.asyncio
async def test_mcp_tool_behaves_identically_to_a_native_tool_through_the_runtime():
    descriptors = [
        MCPToolDescriptor(
            name="lookup_weather",
            description="Look up the weather",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            permissions=["external_network"],
        )
    ]
    client = FakeMCPClient(descriptors, responses={"lookup_weather": {"temp_c": 21}})
    registry = ToolRegistry()
    await register_mcp_server(registry, client, server_name="demo_server")

    runtime = ToolRuntime(registry, EventLog(verbose=False))

    denied = await runtime.call("mcp__demo_server__lookup_weather", city="Berlin", agent_permissions=[])
    assert not denied.success
    assert denied.error_code.value == "permission_denied"

    granted = await runtime.call(
        "mcp__demo_server__lookup_weather", city="Berlin", agent_permissions=["external_network"]
    )
    assert granted.success
    assert granted.output == {"temp_c": 21}
    assert client.calls == [("lookup_weather", {"city": "Berlin"})]


@pytest.mark.asyncio
async def test_mcp_transport_errors_become_structured_tool_results_not_crashes():
    descriptors = [MCPToolDescriptor(name="broken", description="always fails")]
    client = FakeMCPClient(descriptors, responses={})  # no response configured -> raises
    registry = ToolRegistry()
    await register_mcp_server(registry, client, server_name="demo_server")
    runtime = ToolRuntime(registry, EventLog(verbose=False))

    result = await runtime.call("mcp__demo_server__broken", agent_permissions=[])
    assert not result.success
    assert result.error_code.value == "execution_error"


@pytest.mark.asyncio
async def test_mcp_tool_ids_are_namespaced_per_server_to_avoid_collisions():
    descriptor = MCPToolDescriptor(name="search", description="search tool")
    client_a = FakeMCPClient([descriptor], responses={"search": "a"})
    client_b = FakeMCPClient([descriptor], responses={"search": "b"})
    registry = ToolRegistry()

    await register_mcp_server(registry, client_a, server_name="server_a")
    await register_mcp_server(registry, client_b, server_name="server_b")

    assert registry.has("mcp__server_a__search")
    assert registry.has("mcp__server_b__search")
    assert registry.get("mcp__server_a__search").id != registry.get("mcp__server_b__search").id
