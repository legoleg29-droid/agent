"""Orchestrator-level validation: required tools must be registered, and
the routed agent must both declare and have permission for them, before
any LLM call is made. This is item 13 of the Phase 2 spec.
"""

import json

import pytest

from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.core.task_graph import TaskStatus
from orchestrator.providers.mock_provider import MockProvider
from orchestrator.tools.calculator_tool import CalculatorTool
from orchestrator.tools.registry import ToolRegistry
from tests.doubles import StubAgent, always_succeeds


def plan_with_required_tool(tool_id: str) -> dict:
    return {
        "tasks": [
            {
                "id": "t1",
                "objective": "do something",
                "capability": "research",
                "dependencies": [],
                "required_tools": [tool_id],
            }
        ]
    }


def test_missing_tool_registration_is_rejected_by_the_orchestrators_preflight_check():
    """Exercised directly against the orchestrator's validation helper:
    the Planner already rejects a plan referencing an unregistered tool at
    plan-parse time (see test_planner.py::test_plan_rejects_unknown_tool),
    so a task can only reach execution with a required tool that *was*
    registered at planning time. This test proves the orchestrator's own
    pre-execution check (defense in depth against a tool being unregistered
    between planning and execution) independently catches the same case.
    """
    from orchestrator.core.task_graph import Task

    registry = AgentRegistry()
    agent = StubAgent("research_agent", ["research"], always_succeeds(), available_tools=["calculator"])
    registry.register(agent)

    provider = MockProvider()
    orchestrator = Orchestrator(provider, registry, ToolRegistry(), verbose_logging=False)
    task = Task(id="t1", objective="do something", capability="research", required_tools=["calculator"])

    error = orchestrator._validate_tool_requirements(task, agent)
    assert error is not None
    assert "not registered" in error


@pytest.mark.asyncio
async def test_agent_not_declaring_required_tool_fails_fast():
    registry = AgentRegistry()
    # Agent has the capability but doesn't list the tool the task needs.
    agent = StubAgent("research_agent", ["research"], always_succeeds(), available_tools=[])
    registry.register(agent)

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())

    plan = plan_with_required_tool("calculator")
    provider = MockProvider(
        responder=lambda s, m, t: json.dumps(plan) if m[-1].content.startswith("User goal:") else "final"
    )
    orchestrator = Orchestrator(provider, registry, tool_registry, max_retries_per_task=0, max_replans=0, verbose_logging=False)
    result = await orchestrator.run("do something")

    assert not result.succeeded
    assert agent.call_count == 0
    assert "does not declare required tool" in result.graph.tasks["t1"].error


@pytest.mark.asyncio
async def test_agent_missing_permission_for_required_tool_fails_fast():
    registry = AgentRegistry()
    # Declares the tool but lacks the permission the tool requires.
    agent = StubAgent("research_agent", ["research"], always_succeeds(), available_tools=["guarded_tool"])
    registry.register(agent)

    from orchestrator.tools.base import BaseTool, ToolResult

    class GuardedTool(BaseTool):
        id = "guarded_tool"
        name = "Guarded"
        description = "needs a permission"
        input_schema = {"type": "object", "properties": {}}
        permissions = ["special.permission"]

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="ok")

    tool_registry = ToolRegistry()
    tool_registry.register(GuardedTool())

    plan = plan_with_required_tool("guarded_tool")
    provider = MockProvider(
        responder=lambda s, m, t: json.dumps(plan) if m[-1].content.startswith("User goal:") else "final"
    )
    orchestrator = Orchestrator(provider, registry, tool_registry, max_retries_per_task=0, max_replans=0, verbose_logging=False)
    result = await orchestrator.run("do something")

    assert not result.succeeded
    assert agent.call_count == 0
    assert "lacks permission" in result.graph.tasks["t1"].error


@pytest.mark.asyncio
async def test_task_with_satisfied_tool_requirements_runs_normally():
    registry = AgentRegistry()
    agent = StubAgent("research_agent", ["research"], always_succeeds(), available_tools=["calculator"])
    registry.register(agent)

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())

    plan = plan_with_required_tool("calculator")
    provider = MockProvider(
        responder=lambda s, m, t: json.dumps(plan) if m[-1].content.startswith("User goal:") else "final"
    )
    orchestrator = Orchestrator(provider, registry, tool_registry, max_retries_per_task=0, max_replans=0, verbose_logging=False)
    result = await orchestrator.run("do something")

    assert result.succeeded
    assert agent.call_count == 1
    assert result.graph.tasks["t1"].status == TaskStatus.SUCCEEDED
