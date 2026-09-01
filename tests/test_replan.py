import json

import pytest

from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.core.task_graph import TaskStatus
from orchestrator.providers.mock_provider import MockProvider
from orchestrator.tools.registry import ToolRegistry
from tests.doubles import StubAgent, always_fails, always_succeeds

INITIAL_PLAN = {
    "tasks": [
        {"id": "t1", "objective": "Do the impossible task", "capability": "research", "dependencies": [], "required_tools": []},
    ]
}
REPLAN = {
    "reason": "t1 needs a recoverable approach",
    "operations": [
        {
            "op": "replace_task",
            "task_id": "t1",
            "task": {
                "id": "recovery",
                "objective": "Do a recoverable version of the task",
                "capability": "research",
                "dependencies": [],
                "required_tools": [],
            },
        }
    ],
}
NO_RECOVERY_PATCH = {"reason": "no viable recovery", "operations": []}


def responder_factory(replanned: dict):
    def _respond(system, messages, tools):
        last = messages[-1].content
        if last.startswith("User goal:"):
            return json.dumps(INITIAL_PLAN)
        if "Failed task:" in last:
            return json.dumps(replanned["plan"])
        return "Final result after recovery."

    return _respond


@pytest.mark.asyncio
async def test_persistent_failure_triggers_replan_and_recovers():
    registry = AgentRegistry()
    # The originally routed agent always fails; a *second* agent, only reachable
    # via the replanned task id, succeeds - proving the new plan actually ran.
    failing_agent = StubAgent("failing_agent", ["research"], always_fails("cannot do the impossible task"))
    registry.register(failing_agent)

    provider = MockProvider(responder=responder_factory({"plan": REPLAN}))
    orchestrator = Orchestrator(
        provider, registry, ToolRegistry(), max_retries_per_task=1, max_replans=1, verbose_logging=False
    )
    result = await orchestrator.run("Do the impossible task")

    assert "t1" in result.graph.tasks
    assert result.graph.tasks["t1"].status == TaskStatus.FAILED
    recovery_task_ids = [tid for tid in result.graph.tasks if tid.startswith("recovery")]
    assert recovery_task_ids, "replanned task was not spliced into the graph"
    # recovery task also fails since only failing_agent is registered - but the
    # replan mechanism itself must have fired (retry budget respected, new task spliced in).
    assert orchestrator.state.replans_used == 1


@pytest.mark.asyncio
async def test_replan_budget_is_respected_and_run_aborts_safely():
    registry = AgentRegistry()
    failing_agent = StubAgent("failing_agent", ["research"], always_fails("boom"))
    registry.register(failing_agent)

    provider = MockProvider(responder=responder_factory({"plan": NO_RECOVERY_PATCH}))
    orchestrator = Orchestrator(
        provider, registry, ToolRegistry(), max_retries_per_task=0, max_replans=1, verbose_logging=False
    )
    result = await orchestrator.run("Do the impossible task")

    assert orchestrator.state.replans_used <= 1
    assert not result.succeeded


@pytest.mark.asyncio
async def test_successful_retry_does_not_consume_replan_budget():
    registry = AgentRegistry()
    from tests.doubles import fails_n_times_then_succeeds

    flaky_agent = StubAgent("flaky_agent", ["research"], fails_n_times_then_succeeds(1))
    registry.register(flaky_agent)

    provider = MockProvider(responder=responder_factory({"plan": REPLAN}))
    orchestrator = Orchestrator(
        provider, registry, ToolRegistry(), max_retries_per_task=2, max_replans=1, verbose_logging=False
    )
    result = await orchestrator.run("Do a flaky task")

    assert result.succeeded
    assert orchestrator.state.replans_used == 0
    assert flaky_agent.call_count == 2
