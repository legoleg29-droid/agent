"""Orchestrator-level pause_execution/cancel_execution: both the "live run
signaled from another coroutine" path and the "not currently running,
best-effort persisted-state" path."""

import asyncio
import json

import pytest

from orchestrator.agents.base import AgentOutput
from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.core.task_graph import TaskStatus
from orchestrator.providers.mock_provider import MockProvider
from orchestrator.state.models import ExecutionState, ExecutionStatus
from orchestrator.state.store import ExecutionNotFoundError, InMemoryStateStore
from orchestrator.tools.registry import ToolRegistry
from tests.doubles import StubAgent


class SlowAgent(StubAgent):
    def __init__(self, *a, delay=0.05, **kw):
        super().__init__(*a, **kw)
        self.delay = delay

    async def execute(self, agent_input):
        self.call_count += 1
        await asyncio.sleep(self.delay)
        return AgentOutput(success=True, content="slow output, long enough to pass evaluation.")


def five_task_plan():
    return {
        "tasks": [
            {"id": f"t{i}", "objective": "x", "capability": "research", "dependencies": [], "required_tools": []}
            for i in range(5)
        ]
    }


def five_task_chain_plan():
    """A dependency chain, not 5 independent tasks - so not everything
    becomes READY in the very first scheduler tick, letting a pause
    requested mid-run actually prevent later tasks from starting."""
    tasks = []
    for i in range(5):
        deps = [f"t{i - 1}"] if i > 0 else []
        tasks.append({"id": f"t{i}", "objective": "x", "capability": "research", "dependencies": deps, "required_tools": []})
    return {"tasks": tasks}


@pytest.mark.asyncio
async def test_cancel_execution_stops_a_live_run_without_completing_unfinished_tasks():
    registry = AgentRegistry()
    agent = SlowAgent("research_agent", ["research"], lambda *a: None, delay=0.05)
    registry.register(agent)
    provider = MockProvider(
        responder=lambda s, m, t: json.dumps(five_task_plan()) if m[-1].content.startswith("User goal:") else "final"
    )
    store = InMemoryStateStore()
    orch = Orchestrator(provider, registry, ToolRegistry(), state_store=store, verbose_logging=False, max_concurrent_tasks=2)

    async def canceller():
        await asyncio.sleep(0.02)
        orch.cancel_execution("exec_cancel_live")

    result, _ = await asyncio.gather(orch.run("goal", execution_id="exec_cancel_live"), canceller())

    assert not result.succeeded
    assert result.execution_state.status == ExecutionStatus.CANCELLED
    statuses = {t.status for t in result.graph.tasks.values()}
    assert TaskStatus.CANCELLED in statuses
    assert TaskStatus.PENDING not in statuses
    assert TaskStatus.RUNNING not in statuses

    persisted = store.load("exec_cancel_live")
    assert persisted.status == ExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_pause_execution_stops_a_live_run_and_resume_continues_it():
    registry = AgentRegistry()
    agent = SlowAgent("research_agent", ["research"], lambda *a: None, delay=0.05)
    registry.register(agent)
    provider = MockProvider(
        responder=lambda s, m, t: json.dumps(five_task_chain_plan()) if m[-1].content.startswith("User goal:") else "final"
    )
    store = InMemoryStateStore()
    orch = Orchestrator(provider, registry, ToolRegistry(), state_store=store, verbose_logging=False, max_concurrent_tasks=2)

    async def pauser():
        await asyncio.sleep(0.02)
        orch.pause_execution("exec_pause_live")

    result, _ = await asyncio.gather(orch.run("goal", execution_id="exec_pause_live"), pauser())

    assert result.execution_state.status == ExecutionStatus.PAUSED
    assert not result.succeeded  # paused runs report as not-yet-succeeded
    remaining_pending = [t for t in result.graph.tasks.values() if t.status == TaskStatus.PENDING]
    assert remaining_pending, "expected some tasks to still be waiting after a pause"

    # Resume on a fresh Orchestrator instance sharing the same store.
    registry_2 = AgentRegistry()
    agent_2 = SlowAgent("research_agent", ["research"], lambda *a: None, delay=0.001)
    registry_2.register(agent_2)
    orch_2 = Orchestrator(provider, registry_2, ToolRegistry(), state_store=store, verbose_logging=False)
    resumed = await orch_2.resume_execution("exec_pause_live")

    assert resumed.succeeded
    assert all(t.status == TaskStatus.SUCCEEDED for t in resumed.graph.tasks.values())
    # the tasks that already completed before the pause were never re-run
    assert agent_2.call_count == len(remaining_pending)


def test_cancel_execution_on_a_not_running_persisted_execution_is_best_effort():
    store = InMemoryStateStore()
    state = ExecutionState.create("goal", execution_id="exec_offline")
    store.save(state)

    orch = Orchestrator(MockProvider(), AgentRegistry(), ToolRegistry(), state_store=store, verbose_logging=False)
    orch.cancel_execution("exec_offline")

    assert store.load("exec_offline").status == ExecutionStatus.CANCELLED


def test_pause_execution_on_a_not_running_persisted_execution_is_best_effort():
    store = InMemoryStateStore()
    state = ExecutionState.create("goal", execution_id="exec_offline_2")
    store.save(state)

    orch = Orchestrator(MockProvider(), AgentRegistry(), ToolRegistry(), state_store=store, verbose_logging=False)
    orch.pause_execution("exec_offline_2")

    assert store.load("exec_offline_2").status == ExecutionStatus.PAUSED


def test_cancel_execution_on_unknown_id_raises():
    orch = Orchestrator(MockProvider(), AgentRegistry(), ToolRegistry(), state_store=InMemoryStateStore(), verbose_logging=False)
    with pytest.raises(ExecutionNotFoundError):
        orch.cancel_execution("does_not_exist")


def test_pause_execution_on_unknown_id_raises():
    orch = Orchestrator(MockProvider(), AgentRegistry(), ToolRegistry(), state_store=InMemoryStateStore(), verbose_logging=False)
    with pytest.raises(ExecutionNotFoundError):
        orch.pause_execution("does_not_exist")
