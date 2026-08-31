"""End-to-end test of the full orchestration pipeline using the real,
production agent and tool implementations with a scripted MockProvider
standing in for Claude. Mirrors the example from the spec:

    "Research competitors, analyze them, and create a strategy."
    Task 1 -> ResearchAgent
    Task 2 -> AnalysisAgent (depends on Task 1)
    Task 3 -> WriterAgent (depends on Task 2)
"""

import json

import pytest

from orchestrator.agents.analysis_agent import AnalysisAgent
from orchestrator.agents.registry import AgentRegistry
from orchestrator.agents.research_agent import ResearchAgent
from orchestrator.agents.writer_agent import WriterAgent
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.core.task_graph import TaskStatus
from orchestrator.providers.mock_provider import MockProvider, ScriptedToolUse
from orchestrator.tools.calculator_tool import CalculatorTool
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.web_search_tool import WebSearchTool

PLAN = {
    "tasks": [
        {
            "id": "research_competitors",
            "objective": "Research the main competitors in the BI analytics market",
            "capability": "research",
            "dependencies": [],
            "required_tools": ["web_search"],
            "expected_output": "A summary of competitors and their positioning",
        },
        {
            "id": "analyze_competitors",
            "objective": "Analyze the competitive landscape from the research",
            "capability": "analysis",
            "dependencies": ["research_competitors"],
            "required_tools": [],
            "expected_output": "Key insights, gaps and opportunities",
        },
        {
            "id": "write_strategy",
            "objective": "Write a go-to-market strategy based on the analysis",
            "capability": "strategy",
            "dependencies": ["analyze_competitors"],
            "required_tools": [],
            "expected_output": "A clear strategic recommendation",
        },
    ]
}


def e2e_responder(system: str, messages, tools):
    if system.startswith("You are the planning module"):
        return json.dumps(PLAN)
    if system.startswith("You are a research analyst"):
        if len(messages) == 1:
            assert tools and tools[0]["name"] == "web_search", "research agent should offer web_search via Claude tool-use"
            return ScriptedToolUse(name="web_search", arguments={"query": "BI analytics competitors"})
        return (
            "Research findings: Acme, Globex, and Initech are the primary competitors "
            "in BI analytics, each pursuing a different strategy in the market."
        )
    if system.startswith("You are a business/data analyst"):
        return (
            "Analysis: Acme is well-funded and horizontal, Globex targets SMBs with "
            "no-code tooling, and Initech is pivoting to vertical logistics analytics - "
            "leaving a mid-market AI-native gap unaddressed by any competitor."
        )
    if system.startswith("You are a professional writer"):
        return (
            "Strategy: Position as the AI-native mid-market BI platform, undercutting "
            "Acme on price while out-featuring Globex's no-code builder, and avoiding "
            "direct collision with Initech's new logistics vertical."
        )
    if system.startswith("You are the final synthesis stage"):
        return (
            "Final Strategy Report: Enter the mid-market BI segment as the AI-native "
            "alternative, differentiated on price versus Acme and depth versus Globex, "
            "while sidestepping Initech's logistics-vertical retreat."
        )
    raise AssertionError(f"Unexpected prompt for system: {system[:60]!r}")


@pytest.mark.asyncio
async def test_full_orchestration_pipeline_end_to_end():
    provider = MockProvider(responder=e2e_responder)

    tool_registry = ToolRegistry()
    tool_registry.register(WebSearchTool())
    tool_registry.register(CalculatorTool())

    agent_registry = AgentRegistry()
    orchestrator = Orchestrator(provider, agent_registry, tool_registry, verbose_logging=False)

    # Agents share the orchestrator's tool runtime so TOOL events are logged
    # to the same observability stream.
    agent_registry.register(ResearchAgent(provider, orchestrator.tool_runtime))
    agent_registry.register(AnalysisAgent(provider, orchestrator.tool_runtime))
    agent_registry.register(WriterAgent(provider, orchestrator.tool_runtime))
    result = await orchestrator.run("Research competitors, analyze them, and create a strategy.")

    assert result.succeeded
    assert len(result.graph.tasks) == 3
    for task in result.graph.tasks.values():
        assert task.status == TaskStatus.SUCCEEDED

    research_task = result.graph.tasks["research_competitors"]
    assert research_task.agent_id == "research_agent"
    assert research_task.result.tool_calls == 1  # used web_search once

    analysis_task = result.graph.tasks["analyze_competitors"]
    assert analysis_task.agent_id == "analysis_agent"

    writer_task = result.graph.tasks["write_strategy"]
    assert writer_task.agent_id == "writer_agent"

    assert "strategy" in result.final_output.lower()

    tags_seen = {e["tag"] for e in result.events}
    assert {
        "ORCHESTRATOR",
        "PLANNER",
        "ROUTER",
        "TASK",
        "AGENT",
        "TOOL_REQUEST",
        "TOOL_PERMISSION",
        "TOOL_VALIDATION",
        "TOOL_EXECUTION",
        "TOOL_RESULT",
        "EVALUATOR",
        "COMPLETE",
    }.issubset(tags_seen)

    # context.tool_results carries a structured record of the web_search call,
    # not just an unstructured dump.
    tool_result_entries = [r for r in result.tool_results if r["tool"] == "web_search"]
    assert len(tool_result_entries) == 1
    assert tool_result_entries[0]["status"] == "success"
    assert tool_result_entries[0]["task_id"] == "research_competitors"
