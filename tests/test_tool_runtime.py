import asyncio

import pytest

from orchestrator.core.logging_utils import EventLog
from orchestrator.tools.base import BaseTool, ToolErrorCode, ToolResult
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.runtime import ToolRuntime, _summarize_args


class EchoTool(BaseTool):
    id = "echo"
    name = "Echo"
    description = "Echoes back its input."
    input_schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    output_schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    permissions: list[str] = []

    async def execute(self, *, text: str) -> ToolResult:
        return ToolResult(success=True, output={"text": text})


class PermissionedTool(BaseTool):
    id = "guarded"
    name = "Guarded"
    description = "Requires a permission."
    input_schema = {"type": "object", "properties": {}}
    permissions = ["dangerous.thing"]

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="ok")


class SlowTool(BaseTool):
    id = "slow"
    name = "Slow"
    description = "Takes longer than its timeout."
    input_schema = {"type": "object", "properties": {}}
    permissions: list[str] = []
    timeout_seconds = 0.05

    async def execute(self, **kwargs) -> ToolResult:
        await asyncio.sleep(1)
        return ToolResult(success=True, output="too late")


class BrokenTool(BaseTool):
    id = "broken"
    name = "Broken"
    description = "Always raises."
    input_schema = {"type": "object", "properties": {}}
    permissions: list[str] = []

    async def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("kaboom")


class MalformedOutputTool(BaseTool):
    id = "malformed"
    name = "Malformed"
    description = "Returns output that violates its own schema."
    input_schema = {"type": "object", "properties": {}}
    output_schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
    permissions: list[str] = []

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output={"count": "not a number"})


def make_runtime(*tools: BaseTool) -> tuple[ToolRuntime, EventLog]:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    event_log = EventLog(verbose=False)
    return ToolRuntime(registry, event_log), event_log


@pytest.mark.asyncio
async def test_successful_call_returns_result_and_logs_lifecycle():
    runtime, event_log = make_runtime(EchoTool())
    result = await runtime.call("echo", text="hi", agent_permissions=[])
    assert result.success
    assert result.output == {"text": "hi"}

    tags = [e.tag for e in event_log.events]
    assert tags == ["TOOL_REQUEST", "TOOL_PERMISSION", "TOOL_VALIDATION", "TOOL_EXECUTION", "TOOL_RESULT"]


@pytest.mark.asyncio
async def test_unavailable_tool_returns_structured_error():
    runtime, event_log = make_runtime()
    result = await runtime.call("nope", agent_permissions=[])
    assert not result.success
    assert result.error_code.value == "unavailable"
    assert any(e.tag == "TOOL_ERROR" for e in event_log.events)


@pytest.mark.asyncio
async def test_permission_check_happens_in_runtime_not_prompt_only():
    runtime, event_log = make_runtime(PermissionedTool())

    denied = await runtime.call("guarded", agent_permissions=[])
    assert not denied.success
    assert denied.error_code.value == "permission_denied"

    granted = await runtime.call("guarded", agent_permissions=["dangerous.thing"])
    assert granted.success

    permission_events = [e for e in event_log.events if e.tag == "TOOL_PERMISSION"]
    assert permission_events[0].status == "denied"
    assert permission_events[1].status == "granted"


@pytest.mark.asyncio
async def test_invalid_arguments_rejected_before_execution():
    runtime, event_log = make_runtime(EchoTool())
    result = await runtime.call("echo", agent_permissions=[])  # missing required 'text'
    assert not result.success
    assert result.error_code.value == "invalid_arguments"
    assert not any(e.tag == "TOOL_EXECUTION" for e in event_log.events)


@pytest.mark.asyncio
async def test_timeout_produces_structured_error():
    runtime, event_log = make_runtime(SlowTool())
    result = await runtime.call("slow", agent_permissions=[])
    assert not result.success
    assert result.error_code.value == "timeout"


@pytest.mark.asyncio
async def test_execution_exception_is_never_silently_swallowed():
    runtime, event_log = make_runtime(BrokenTool())
    result = await runtime.call("broken", agent_permissions=[])
    assert not result.success
    assert result.error_code.value == "execution_error"
    assert "kaboom" in result.error
    error_events = [e for e in event_log.events if e.tag == "TOOL_ERROR"]
    assert error_events and "kaboom" in error_events[0].message


@pytest.mark.asyncio
async def test_malformed_output_is_rejected():
    runtime, event_log = make_runtime(MalformedOutputTool())
    result = await runtime.call("malformed", agent_permissions=[])
    assert not result.success
    assert result.error_code.value == "malformed_output"


@pytest.mark.asyncio
async def test_per_call_timeout_override():
    runtime, _ = make_runtime(EchoTool())
    result = await runtime.call("echo", text="x", agent_permissions=[], timeout=0.001)
    assert result.success  # EchoTool is instant, so even a tiny timeout is fine


def test_arg_summary_never_includes_raw_secret_values():
    summary = _summarize_args({"api_key": "sk-super-secret-value", "query": "hello world"})
    assert summary["api_key"] == "<redacted>"
    assert "sk-super-secret-value" not in str(summary)
    assert summary["query"] == "str(len=11)"  # metadata only, not the raw value
