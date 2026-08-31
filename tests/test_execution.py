import json

import pytest

from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.core.task_graph import TaskStatus
from orchestrator.providers.mock_provider import MockProvider
from orchestrator.tools.registry import ToolRegistry
from tests.doubles import StubAgent, always_succeeds

PLAN = {
    "tasks": [
        {"id": "t1", "objective": "Research competitors", "capability": "research", "dependencies": [], "required_tools": []},
        {"id": "t2", "objective": "Analyze findings", "capability": "analysis", "dependencies": ["t1"], "required_tools": []},
    ]
}


def build_orchestrator():
    registry = AgentRegistry()
    research = StubAgent("research_agent", ["research"], always_succeeds("Competitor research findings, long enough to pass."))
    analysis = StubAgent("analysis_agent", ["analysis"], always_succeeds("Analysis of the findings, long enough to pass."))
    registry.register(research)
    registry.register(analysis)

    provider = MockProvider(responder=lambda system, messages, tools: json.dumps(PLAN) if messages[-1].content.startswith("User goal:") else "Final synthesized result.")
    orchestrator = Orchestrator(provider, registry, ToolRegistry(), verbose_logging=False)
    return orchestrator, research, analysis


@pytest.mark.asyncio
async def test_execution_runs_tasks_in_dependency_order_and_succeeds():
    orchestrator, research, analysis = build_orchestrator()
    result = await orchestrator.run("Research competitors and analyze them")

    assert result.succeeded
    assert result.graph.tasks["t1"].status == TaskStatus.SUCCEEDED
    assert result.graph.tasks["t2"].status == TaskStatus.SUCCEEDED
    assert research.call_count == 1
    assert analysis.call_count == 1


@pytest.mark.asyncio
async def test_downstream_task_receives_upstream_output_not_full_history():
    orchestrator, research, analysis = build_orchestrator()
    await orchestrator.run("Research competitors and analyze them")

    analysis_input = analysis.received_inputs[0]
    assert "t1" in analysis_input.upstream_outputs
    assert "Competitor research findings" in analysis_input.upstream_outputs["t1"]


@pytest.mark.asyncio
async def test_independent_tasks_execute_without_a_router_error():
    registry = AgentRegistry()
    a = StubAgent("a", ["research"], always_succeeds("first branch output long enough."))
    b = StubAgent("b", ["writing"], always_succeeds("second branch output long enough."))
    registry.register(a)
    registry.register(b)

    plan = {
        "tasks": [
            {"id": "x1", "objective": "branch a", "capability": "research", "dependencies": [], "required_tools": []},
            {"id": "x2", "objective": "branch b", "capability": "writing", "dependencies": [], "required_tools": []},
        ]
    }
    provider = MockProvider(responder=lambda s, m, t: json.dumps(plan) if m[-1].content.startswith("User goal:") else "final")
    orchestrator = Orchestrator(provider, registry, ToolRegistry(), verbose_logging=False)
    result = await orchestrator.run("do two independent things")

    assert result.succeeded
    assert a.call_count == 1 and b.call_count == 1
