"""Agent state must be scoped per (execution, task, agent) and must never
leak between tasks, agents, executions, or users via shared mutable
defaults or module-level state."""

import json

import pytest

from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.providers.mock_provider import MockProvider
from orchestrator.state.models import AgentState
from orchestrator.tools.registry import ToolRegistry
from tests.doubles import StubAgent, always_succeeds


def test_agent_state_instances_do_not_share_mutable_defaults():
    a = AgentState(execution_id="exec_1", task_id="t1", agent_id="agent_1")
    b = AgentState(execution_id="exec_1", task_id="t2", agent_id="agent_1")

    a.reasoning_context["seen"] = True
    a.metadata["note"] = "for task 1 only"

    assert b.reasoning_context == {}
    assert b.metadata == {}


def test_agent_state_is_scoped_to_its_execution_task_and_agent():
    state = AgentState(execution_id="exec_1", task_id="t1", agent_id="research_agent", current_objective="research X")
    assert state.execution_id == "exec_1"
    assert state.task_id == "t1"
    assert state.agent_id == "research_agent"
    assert state.current_objective == "research X"


@pytest.mark.asyncio
async def test_two_executions_on_the_same_orchestrator_do_not_share_task_context():
    """Regression guard: task_context (which carries a fresh AgentState per
    call) must never be reused/mutated across separate task runs."""
    registry = AgentRegistry()
    seen_agent_states = []

    class RecordingAgent(StubAgent):
        async def execute(self, agent_input):
            seen_agent_states.append(agent_input.task_context["agent_state"])
            return await super().execute(agent_input)

    agent = RecordingAgent("research_agent", ["research"], always_succeeds())
    registry.register(agent)

    plan = {"tasks": [{"id": "t1", "objective": "a", "capability": "research", "dependencies": [], "required_tools": []}]}
    provider = MockProvider(
        responder=lambda s, m, t: json.dumps(plan) if m[-1].content.startswith("User goal:") else "final"
    )

    orchestrator = Orchestrator(provider, registry, ToolRegistry(), verbose_logging=False)
    await orchestrator.run("goal one", execution_id="exec_A")
    await orchestrator.run("goal two", execution_id="exec_B")

    assert len(seen_agent_states) == 2
    first, second = seen_agent_states
    assert first.execution_id == "exec_A"
    assert second.execution_id == "exec_B"
    assert first is not second


@pytest.mark.asyncio
async def test_cross_execution_isolation_of_persisted_state():
    from orchestrator.state.store import InMemoryStateStore

    registry = AgentRegistry()
    registry.register(StubAgent("research_agent", ["research"], always_succeeds("result for this run, long enough.")))

    plan = {"tasks": [{"id": "t1", "objective": "a", "capability": "research", "dependencies": [], "required_tools": []}]}
    provider = MockProvider(
        responder=lambda s, m, t: json.dumps(plan) if m[-1].content.startswith("User goal:") else "final"
    )
    store = InMemoryStateStore()
    orchestrator = Orchestrator(provider, registry, ToolRegistry(), state_store=store, verbose_logging=False)

    result_a = await orchestrator.run("goal A", execution_id="exec_A")
    result_b = await orchestrator.run("goal B", execution_id="exec_B")

    assert result_a.execution_id == "exec_A"
    assert result_b.execution_id == "exec_B"

    state_a = store.load("exec_A")
    state_b = store.load("exec_B")
    assert state_a.user_goal == "goal A"
    assert state_b.user_goal == "goal B"
    assert state_a.task_states.keys() == state_b.task_states.keys() == {"t1"}
    # each execution's task_states are independent objects
    assert state_a.task_states["t1"] is not state_b.task_states["t1"]
