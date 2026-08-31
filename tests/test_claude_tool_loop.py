"""Tests for the structured Claude tool-use loop in LLMAgent.

Uses MockProvider's ScriptedToolUse to emit the same shape of response
ClaudeProvider would produce for a native tool_use turn, so these tests
exercise the real tool loop (agents/_common.py) rather than any text
parsing.
"""

import pytest

from orchestrator.agents._common import MAX_TOOL_ROUNDS, LLMAgent
from orchestrator.agents.base import AgentInput
from orchestrator.core.logging_utils import EventLog
from orchestrator.providers.mock_provider import MockProvider, ScriptedToolUse
from orchestrator.tools.calculator_tool import CalculatorTool
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.runtime import ToolRuntime


class CalcAgent(LLMAgent):
    id = "calc_agent"
    name = "Calc Agent"
    description = "test agent"
    capabilities = ["math"]
    available_tools = ["calculator"]
    permissions: list[str] = []
    system_instructions = "You are a calculator-using assistant."


def build_agent(responder):
    provider = MockProvider(responder=responder)
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    runtime = ToolRuntime(registry, EventLog(verbose=False))
    agent = CalcAgent(provider, runtime)
    return agent, provider


@pytest.mark.asyncio
async def test_tool_use_loop_calls_tool_and_returns_final_text():
    call_count = {"n": 0}

    def responder(system, messages, tools):
        call_count["n"] += 1
        if call_count["n"] == 1:
            assert tools and tools[0]["name"] == "calculator"
            return ScriptedToolUse(name="calculator", arguments={"expression": "12345 * 6789"})
        # second round: tool_result must be in the transcript
        assert messages[-1].role == "user"
        assert isinstance(messages[-1].content, list)
        assert messages[-1].content[0]["type"] == "tool_result"
        return "The result of 12345 * 6789 is 83810205."

    agent, _ = build_agent(responder)
    output = await agent.execute(AgentInput(objective="Calculate 12345 * 6789."))

    assert output.success
    assert "83810205" in output.content
    assert output.tool_calls == 1


@pytest.mark.asyncio
async def test_agent_without_tool_use_just_returns_text():
    agent, _ = build_agent(lambda system, messages, tools: "no tools needed, here's the answer")
    output = await agent.execute(AgentInput(objective="Say hello"))
    assert output.success
    assert output.tool_calls == 0


@pytest.mark.asyncio
async def test_requesting_a_tool_outside_available_tools_is_rejected():
    rounds = {"n": 0}

    def responder(system, messages, tools):
        rounds["n"] += 1
        if rounds["n"] == 1:
            return ScriptedToolUse(name="file_write", arguments={"path": "x", "content": "y"})
        # the agent should see a permission_denied tool_result and give up gracefully
        payload = messages[-1].content[0]["content"]
        assert "not in this agent's available_tools" in payload
        return "I was not permitted to use that tool."

    agent, _ = build_agent(responder)
    output = await agent.execute(AgentInput(objective="write a file"))
    assert output.success
    assert output.tool_calls == 0  # the disallowed call never reached the runtime


@pytest.mark.asyncio
async def test_exceeding_max_tool_rounds_fails_cleanly():
    def responder(system, messages, tools):
        return ScriptedToolUse(name="calculator", arguments={"expression": "1+1"})

    agent, _ = build_agent(responder)
    output = await agent.execute(AgentInput(objective="loop forever"))
    assert not output.success
    assert "max" in output.error.lower()
    assert output.tool_calls == MAX_TOOL_ROUNDS


@pytest.mark.asyncio
async def test_agent_without_tool_runtime_reports_clear_error_on_tool_request():
    provider = MockProvider(
        responder=lambda system, messages, tools: ScriptedToolUse(name="calculator", arguments={"expression": "1+1"})
    )
    agent = CalcAgent(provider, tool_runtime=None)
    output = await agent.execute(AgentInput(objective="Calculate 1+1"))
    assert not output.success
    assert "ToolRuntime" in output.error
