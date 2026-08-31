#!/usr/bin/env python3
"""CLI entry point for the AI Agent Orchestrator.

Usage:
    python main.py "Research competitors, analyze them, and create a strategy."
    python main.py --mock "Any goal"   # run offline with MockProvider (no API key needed)

Phase 2 acceptance examples (work with --mock, no API key required):
    python main.py --mock "Calculate 12345 * 6789."
    python main.py --mock "Create a text file containing the result of 12345 * 6789."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys

from dotenv import load_dotenv

from orchestrator.agents.analysis_agent import AnalysisAgent
from orchestrator.agents.coding_agent import CodingAgent
from orchestrator.agents.registry import AgentRegistry
from orchestrator.agents.research_agent import ResearchAgent
from orchestrator.agents.writer_agent import WriterAgent
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.providers.base import LLMMessage, LLMProvider
from orchestrator.providers.mock_provider import ScriptedToolUse
from orchestrator.tools.calculator_tool import CalculatorTool
from orchestrator.tools.file_tools import FileReadTool, FileWriteTool, ListFilesTool
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.sandbox import FileSandbox
from orchestrator.tools.web_search_tool import WebSearchTool

_DEFAULT_PLAN = {
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

_MATH_EXPRESSION_RE = re.compile(r"[0-9][0-9.\s+\-*/%()]*[0-9]")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_expression(text: str) -> str:
    matches = _MATH_EXPRESSION_RE.findall(text)
    return max(matches, key=len).strip() if matches else "0"


def _last_number(text: str) -> str:
    matches = _NUMBER_RE.findall(text)
    return matches[-1] if matches else "unknown"


def _last_tool_output(messages: list[LLMMessage]):
    """Pull the payload out of the most recent tool_result turn, mirroring
    what a real model sees when deciding how to respond after a tool call."""
    last = messages[-1] if messages else None
    if last is None or not isinstance(last.content, list):
        return None
    for block in last.content:
        if block.get("type") == "tool_result":
            payload = json.loads(block["content"])
            if payload.get("success"):
                return payload.get("output")
    return None


def _demo_responder(system: str, messages: list[LLMMessage], tools):
    """Deterministic offline stand-in for Claude, used by --mock.

    Reads the goal to decide on a plausible plan (a calculation, a file
    write, or the default research -> analysis -> writing pipeline), and
    uses the provider's structured tool-use mechanism (ScriptedToolUse)
    exactly as ClaudeProvider would - never text parsing. Real runs use
    ClaudeProvider instead.
    """
    raw_first = messages[0].content if messages and isinstance(messages[0].content, str) else ""
    # The planner's prompt is "User goal: ...\n\nAvailable capabilities: ...\nAvailable
    # tools: ..." - only the goal line itself should drive plan-shape decisions below,
    # not tool/capability names that happen to appear later in the same prompt.
    goal_text = raw_first.split("\n\n", 1)[0].removeprefix("User goal: ")
    goal_lower = goal_text.lower()

    if system.startswith("You are the planning module"):
        has_math = bool(re.search(r"\d\s*[+\-*/]\s*\d", goal_text))
        wants_file = "file" in goal_lower and any(k in goal_lower for k in ("write", "create", "save"))

        if has_math and wants_file:
            return json.dumps(
                {
                    "tasks": [
                        {
                            "id": "compute_result",
                            "objective": goal_text,
                            "capability": "analysis",
                            "dependencies": [],
                            "required_tools": ["calculator"],
                            "expected_output": "The numeric result",
                        },
                        {
                            "id": "write_result_file",
                            "objective": "Save the computed result to a text file",
                            "capability": "writing",
                            "dependencies": ["compute_result"],
                            "required_tools": ["file_write"],
                            "expected_output": "Confirmation the file was written",
                        },
                    ]
                }
            )
        if has_math:
            return json.dumps(
                {
                    "tasks": [
                        {
                            "id": "compute_result",
                            "objective": goal_text,
                            "capability": "analysis",
                            "dependencies": [],
                            "required_tools": ["calculator"],
                            "expected_output": "The numeric result",
                        }
                    ]
                }
            )
        if wants_file:
            return json.dumps(
                {
                    "tasks": [
                        {
                            "id": "write_result_file",
                            "objective": goal_text,
                            "capability": "writing",
                            "dependencies": [],
                            "required_tools": ["file_write"],
                            "expected_output": "Confirmation the file was written",
                        }
                    ]
                }
            )
        return json.dumps(_DEFAULT_PLAN)

    if system.startswith("You are a business/data analyst"):
        if len(messages) == 1:
            return ScriptedToolUse(name="calculator", arguments={"expression": _extract_expression(goal_text)})
        output = _last_tool_output(messages)
        result = output.get("result") if isinstance(output, dict) else output
        return f"The result is {result}."

    if system.startswith("You are a professional writer"):
        if len(messages) == 1:
            # Pull the number out of the full prompt (objective + any upstream
            # task outputs), not just the first line, since a standalone
            # "save the result" task has no digits of its own - the number
            # comes from the analysis task's output in upstream_outputs.
            content = f"The result is {_last_number(raw_first)}."
            return ScriptedToolUse(name="file_write", arguments={"path": "result.txt", "content": content})
        output = _last_tool_output(messages)
        if output:
            return f"Saved the result to {output['path']} ({output['bytes_written']} bytes written)."
        return "Failed to write the result to a file."

    if system.startswith("You are a research analyst"):
        if len(messages) == 1:
            return ScriptedToolUse(name="web_search", arguments={"query": "relevant background information"})
        return (
            "Research findings: based on available information, this space has several "
            "active players and a clear trend toward AI-native, lower-cost alternatives "
            "to legacy incumbents."
        )
    if system.startswith("You are the final synthesis stage"):
        return (
            "Summary: research identified the competitive landscape, analysis surfaced "
            "an underserved mid-market AI-native gap, and the resulting strategy is to "
            "enter there with a price- and speed-led offering."
        )
    return f"Mock response for: {goal_text[:80]}"


def build_provider(use_mock: bool) -> LLMProvider:
    if use_mock:
        from orchestrator.providers.mock_provider import MockProvider

        print("[main] Using MockProvider (offline, no API key required).", file=sys.stderr)
        return MockProvider(responder=_demo_responder)

    from orchestrator.providers.claude_provider import ClaudeProvider

    return ClaudeProvider()


def build_orchestrator(provider: LLMProvider) -> Orchestrator:
    sandbox_dir = os.environ.get("ORCHESTRATOR_SANDBOX_DIR", "./sandbox")
    sandbox = FileSandbox(sandbox_dir)
    tool_timeout = float(os.environ.get("ORCHESTRATOR_TOOL_TIMEOUT_SECONDS", 30))

    tool_registry = ToolRegistry()
    tool_registry.register(WebSearchTool())
    tool_registry.register(CalculatorTool())
    tool_registry.register(FileReadTool(sandbox))
    tool_registry.register(FileWriteTool(sandbox))
    tool_registry.register(ListFilesTool(sandbox))

    agent_registry = AgentRegistry()
    orchestrator = Orchestrator(
        provider,
        agent_registry,
        tool_registry,
        max_retries_per_task=int(os.environ.get("ORCHESTRATOR_MAX_RETRIES_PER_TASK", 2)),
        max_replans=int(os.environ.get("ORCHESTRATOR_MAX_REPLANS", 2)),
    )
    orchestrator.tool_runtime.default_timeout_seconds = tool_timeout

    # Agents share the orchestrator's tool runtime so tool calls are logged
    # to the same observability stream as everything else, and so their
    # declared permissions are enforced by the same ToolRuntime.
    agent_registry.register(ResearchAgent(provider, orchestrator.tool_runtime))
    agent_registry.register(AnalysisAgent(provider, orchestrator.tool_runtime))
    agent_registry.register(CodingAgent(provider, orchestrator.tool_runtime))
    agent_registry.register(WriterAgent(provider, orchestrator.tool_runtime))

    print(f"[main] Filesystem sandbox root: {sandbox.root}", file=sys.stderr)
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

    if result.tool_results:
        print("\n" + "=" * 72)
        print("TOOL CALLS")
        print("=" * 72)
        for entry in result.tool_results:
            print(f"  [{entry['status'].upper():7}] {entry['tool']} (task={entry['task_id']})")

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
