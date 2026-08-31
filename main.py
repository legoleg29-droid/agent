#!/usr/bin/env python3
"""CLI entry point for the AI Agent Orchestrator.

Usage:
    python main.py "Research competitors, analyze them, and create a strategy."
    python main.py --mock "Any goal"   # run offline with MockProvider (no API key needed)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from orchestrator.agents.analysis_agent import AnalysisAgent
from orchestrator.agents.coding_agent import CodingAgent
from orchestrator.agents.registry import AgentRegistry
from orchestrator.agents.research_agent import ResearchAgent
from orchestrator.agents.writer_agent import WriterAgent
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.providers.base import LLMProvider
from orchestrator.tools.calculator_tool import CalculatorTool
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.web_search_tool import WebSearchTool


def _demo_responder(system: str, messages) -> str:
    """Deterministic offline stand-in for Claude, used by --mock.

    Produces a plausible research -> analysis -> writing pipeline for any
    goal, and a plausible final answer for each stage, without ever
    calling the network. Real runs use ClaudeProvider instead.
    """
    import json

    last = messages[-1].content if messages else ""

    if system.startswith("You are the planning module"):
        return json.dumps(
            {
                "tasks": [
                    {
                        "id": "gather_information",
                        "objective": "Gather information relevant to the goal",
                        "capability": "research",
                        "dependencies": [],
                        "required_tools": ["web_search"],
                        "expected_output": "Key relevant facts",
                    },
                    {
                        "id": "analyze_information",
                        "objective": "Analyze the gathered information for insights",
                        "capability": "analysis",
                        "dependencies": ["gather_information"],
                        "required_tools": [],
                        "expected_output": "Key insights and patterns",
                    },
                    {
                        "id": "produce_final_deliverable",
                        "objective": "Produce the final deliverable that satisfies the goal",
                        "capability": "synthesis",
                        "dependencies": ["analyze_information"],
                        "required_tools": [],
                        "expected_output": "The final written deliverable",
                    },
                ]
            }
        )
    if system.startswith("You are a research analyst"):
        if len(messages) == 1:
            return 'TOOL_CALL: web_search({"query": "relevant background information"})'
        return (
            "Research findings: based on available information, this space has several "
            "active players and a clear trend toward AI-native, lower-cost alternatives "
            "to legacy incumbents."
        )
    if system.startswith("You are a business/data analyst"):
        return (
            "Analysis: the research indicates a gap for a focused, AI-native offering "
            "that undercuts incumbents on price while matching them on core features, "
            "with the greatest opportunity in the underserved mid-market segment."
        )
    if system.startswith("You are a professional writer"):
        return (
            "Final deliverable: recommend positioning as the AI-native, mid-market "
            "focused alternative - leading with price and speed-to-value, backed by the "
            "insights above, and validated against the identified gap in the market."
        )
    if system.startswith("You are the final synthesis stage"):
        return (
            "Summary: research identified the competitive landscape, analysis surfaced "
            "an underserved mid-market AI-native gap, and the resulting strategy is to "
            "enter there with a price- and speed-led offering."
        )
    return f"Mock response for: {last[:80]}"


def build_provider(use_mock: bool) -> LLMProvider:
    if use_mock:
        from orchestrator.providers.mock_provider import MockProvider

        print("[main] Using MockProvider (offline, no API key required).", file=sys.stderr)
        return MockProvider(responder=_demo_responder)

    from orchestrator.providers.claude_provider import ClaudeProvider

    return ClaudeProvider()


def build_orchestrator(provider: LLMProvider) -> Orchestrator:
    tool_registry = ToolRegistry()
    tool_registry.register(WebSearchTool())
    tool_registry.register(CalculatorTool())

    agent_registry = AgentRegistry()
    orchestrator = Orchestrator(
        provider,
        agent_registry,
        tool_registry,
        max_retries_per_task=int(os.environ.get("ORCHESTRATOR_MAX_RETRIES_PER_TASK", 2)),
        max_replans=int(os.environ.get("ORCHESTRATOR_MAX_REPLANS", 2)),
    )

    # Agents share the orchestrator's tool runtime so tool calls are logged
    # to the same observability stream as everything else.
    agent_registry.register(ResearchAgent(provider, orchestrator.tool_runtime))
    agent_registry.register(AnalysisAgent(provider, orchestrator.tool_runtime))
    agent_registry.register(CodingAgent(provider, orchestrator.tool_runtime))
    agent_registry.register(WriterAgent(provider, orchestrator.tool_runtime))

    return orchestrator


async def main_async(goal: str, use_mock: bool) -> int:
    provider = build_provider(use_mock)
    orchestrator = build_orchestrator(provider)

    result = await orchestrator.run(goal)

    print("\n" + "=" * 72)
    print("TASK GRAPH")
    print("=" * 72)
    for task in result.graph.topological_order():
        print(f"  [{task.status.value.upper():10}] {task.id} (agent={task.agent_id}, retries={task.retry_count})")

    print("\n" + "=" * 72)
    print("FINAL RESULT")
    print("=" * 72)
    print(result.final_output)
    print()

    return 0 if result.succeeded else 1


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="AI Agent Orchestrator")
    parser.add_argument("goal", help="High-level goal for the orchestrator to accomplish")
    parser.add_argument(
        "--mock", action="store_true", help="Run offline with a deterministic MockProvider instead of Claude"
    )
    args = parser.parse_args()

    exit_code = asyncio.run(main_async(args.goal, args.mock))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
