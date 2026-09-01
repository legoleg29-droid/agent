"""The two Phase 5 end-to-end acceptance scenarios from the spec:

1. Coding self-repair: Planner -> Coding Agent -> broken implementation ->
   Evaluator independently re-runs the tests (never trusting the agent's
   own claim) -> REPAIR_REQUIRED -> Coding Agent fixes it using the failure
   feedback -> Evaluator -> PASS, without the orchestrator restarting the
   whole workflow.

2. Replanning: A -> B -> C, B fails permanently. A stays completed and is
   never re-executed; B is replaced by B2 (auto-rewiring C's dependency);
   C then runs against B2's output.
"""

import json

import pytest

from orchestrator.agents.base import AgentInput, AgentOutput
from orchestrator.agents.coding_agent import CodingAgent
from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.core.task_graph import TaskStatus
from orchestrator.providers.mock_provider import MockProvider, ScriptedToolUse
from orchestrator.tools.code_execution_tool import RunPythonTestsTool
from orchestrator.tools.file_tools import FileReadTool, FileWriteTool, ListFilesTool
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.sandbox import FileSandbox
from tests.doubles import StubAgent

# -- E2E test 1: coding self-repair ------------------------------------------

WRONG_CODE = "def fib(n):\n    return n\n"
CORRECT_CODE = (
    "def fib(n):\n"
    "    a, b = 0, 1\n"
    "    for _ in range(n):\n"
    "        a, b = b, a + b\n"
    "    return a\n"
)
TEST_CODE = "from fib import fib\n\n\ndef test_fib():\n    assert fib(0) == 0\n    assert fib(1) == 1\n    assert fib(10) == 55\n"

FIB_PLAN = {
    "tasks": [
        {
            "id": "t1",
            "objective": "Create a Python utility that calculates Fibonacci numbers and include tests.",
            "capability": "coding",
            "dependencies": [],
            "required_tools": ["file_write", "run_python_tests"],
            "expected_output": "fib.py plus a passing test suite",
            "acceptance_criteria": [
                {"type": "tool_succeeded", "tool": "run_python_tests"},
                {"type": "min_length", "length": 10},
            ],
        }
    ]
}


def _fib_responder(system, messages, tools):
    first = messages[0].content
    if isinstance(first, str) and first.startswith("User goal:"):
        return json.dumps(FIB_PLAN)

    if isinstance(first, str) and first.startswith("Objective (unchanged):"):
        # Repair attempt: fix fib.py, re-run the tests, then report done.
        round_index = (len(messages) - 1) // 2
        if round_index == 0:
            return ScriptedToolUse(name="file_write", arguments={"path": "fib.py", "content": CORRECT_CODE})
        if round_index == 1:
            return ScriptedToolUse(name="run_python_tests", arguments={"path": "."})
        return "Fixed the implementation based on the failing-test feedback; tests should pass now."

    if isinstance(first, str) and first.startswith("Objective:"):
        # Original attempt: write a broken implementation, write the tests,
        # run them (they fail), then claim success anyway - the evaluator
        # must catch this independently, never trusting the claim.
        round_index = (len(messages) - 1) // 2
        if round_index == 0:
            return ScriptedToolUse(name="file_write", arguments={"path": "fib.py", "content": WRONG_CODE})
        if round_index == 1:
            return ScriptedToolUse(name="file_write", arguments={"path": "test_fib.py", "content": TEST_CODE})
        if round_index == 2:
            return ScriptedToolUse(name="run_python_tests", arguments={"path": "."})
        return "Implemented fibonacci in fib.py with tests in test_fib.py."

    return "Final synthesized result."


@pytest.mark.asyncio
async def test_coding_self_repair_end_to_end(tmp_path):
    sandbox = FileSandbox(tmp_path)
    tool_registry = ToolRegistry()
    tool_registry.register(FileReadTool(sandbox))
    tool_registry.register(FileWriteTool(sandbox))
    tool_registry.register(ListFilesTool(sandbox))
    tool_registry.register(RunPythonTestsTool(sandbox, timeout_seconds=30))

    registry = AgentRegistry()
    provider = MockProvider(responder=_fib_responder)
    orch = Orchestrator(
        provider,
        registry,
        tool_registry,
        verbose_logging=False,
        max_retries_per_task=1,
        max_repairs_per_task=2,
        sandbox=sandbox,
    )
    registry.register(CodingAgent(provider, orch.tool_runtime))

    result = await orch.run("Create a Python utility that calculates Fibonacci numbers and include tests.")

    assert result.succeeded, result.final_output
    task = result.graph.tasks["t1"]
    assert task.status == TaskStatus.SUCCEEDED
    # Exactly one repair happened - the agent was not simply retried from
    # scratch, and the whole (one-task) workflow was not restarted.
    assert task.repair_count == 1
    assert len(result.graph.tasks) == 1
    assert not any(e["tag"] == "REPLAN_STARTED" for e in result.events)
    repair_started = [e for e in result.events if e["tag"] == "REPAIR_STARTED"]
    repair_completed = [e for e in result.events if e["tag"] == "REPAIR_COMPLETED"]
    assert len(repair_started) == 1
    assert len(repair_completed) == 1
    # The evaluator's PASS is backed by an actual, independently-verified
    # test run - not the agent's own claim of success.
    assert sandbox.read_text("fib.py") == CORRECT_CODE


# -- E2E test 2: replanning A -> B -> C, B fails -> B2 -----------------------

ABC_PLAN = {
    "tasks": [
        {"id": "a", "objective": "do A", "capability": "research", "dependencies": [], "required_tools": []},
        {"id": "b", "objective": "do B", "capability": "research", "dependencies": ["a"], "required_tools": []},
        {"id": "c", "objective": "do C", "capability": "research", "dependencies": ["b"], "required_tools": []},
    ]
}
B_REPLAN_PATCH = {
    "reason": "B's approach cannot work - replacing it with B2",
    "operations": [
        {
            "op": "replace_task",
            "task_id": "b",
            "task": {
                "id": "b2",
                "objective": "do B (recovered approach)",
                "capability": "research",
                "dependencies": ["a"],
                "required_tools": [],
            },
        }
    ],
}


def _abc_responder(system, messages, tools):
    first = messages[0].content
    if isinstance(first, str) and first.startswith("User goal:"):
        return json.dumps(ABC_PLAN)
    if isinstance(first, str) and "Failed task:" in first:
        return json.dumps(B_REPLAN_PATCH)
    return "Final synthesized result."


@pytest.mark.asyncio
async def test_replanning_b_to_b2_never_re_executes_a():
    registry = AgentRegistry()
    call_log: list[str] = []

    def behavior(_call_index: int, agent_input: AgentInput) -> AgentOutput:
        call_log.append(agent_input.objective)
        if agent_input.objective == "do B":
            return AgentOutput(success=False, error="cannot do B - the approach is fundamentally broken")
        return AgentOutput(success=True, content=f"Completed: {agent_input.objective}, with enough length to pass checks.")

    agent = StubAgent("research_agent", ["research"], behavior)
    registry.register(agent)

    provider = MockProvider(responder=_abc_responder)
    orch = Orchestrator(
        provider, registry, ToolRegistry(), verbose_logging=False, max_retries_per_task=0, max_replans=1
    )
    result = await orch.run("goal")

    assert result.succeeded, result.final_output
    graph = result.graph

    assert graph.tasks["a"].status == TaskStatus.SUCCEEDED
    assert graph.tasks["b"].status == TaskStatus.FAILED
    assert "b2" in graph.tasks
    assert graph.tasks["b2"].status == TaskStatus.SUCCEEDED
    assert graph.tasks["c"].status == TaskStatus.SUCCEEDED
    # C was automatically rewired off the failed B onto the replacement B2.
    assert graph.tasks["c"].dependencies == ["b2"]

    # A ran exactly once - completed work was never unnecessarily re-run.
    assert call_log.count("do A") == 1
    assert call_log.count("do B") == 1
    assert call_log.count("do B (recovered approach)") == 1
    assert call_log.count("do C") == 1

    replan_started = [e for e in result.events if e["tag"] == "REPLAN_STARTED"]
    replan_completed = [e for e in result.events if e["tag"] == "REPLAN_COMPLETED"]
    plan_versions = [e for e in result.events if e["tag"] == "PLAN_VERSION_CREATED"]
    assert len(replan_started) == 1
    assert len(replan_completed) == 1
    assert len(plan_versions) == 2  # initial plan + the replan
    assert result.execution_state.plan_version == 2
    assert len(result.execution_state.plan_versions) == 2
    assert result.execution_state.replan_history
    assert result.execution_state.replan_history[0]["failed_task_id"] == "b"
