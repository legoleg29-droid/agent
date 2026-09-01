"""Orchestrator-level integration tests for Phase 4: retry backoff through
the real _run_task/DAGScheduler wiring, the exact crash-recovery scenario
from the spec (A completed, B completed, C running, D pending -> crash ->
resume), replanning integration with failure isolation, and parallel
context safety (no cross-task contamination)."""

import json

import pytest

from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.core.task_graph import Task, TaskGraph, TaskStatus
from orchestrator.providers.mock_provider import MockProvider
from orchestrator.state.models import ExecutionState
from orchestrator.state.store import InMemoryStateStore
from orchestrator.tools.registry import ToolRegistry
from tests.doubles import AgentInput, AgentOutput, StubAgent, always_succeeds, fails_n_times_then_succeeds


def plan_responder(plan):
    return lambda s, m, t: json.dumps(plan) if m[-1].content.startswith("User goal:") else "final synthesized result"


# -- Retry with backoff ---------------------------------------------------


@pytest.mark.asyncio
async def test_retry_transitions_through_retrying_status_with_backoff():
    registry = AgentRegistry()
    agent = StubAgent("research_agent", ["research"], fails_n_times_then_succeeds(1))
    registry.register(agent)
    plan = {"tasks": [{"id": "t1", "objective": "a", "capability": "research", "dependencies": [], "required_tools": []}]}
    provider = MockProvider(responder=plan_responder(plan))

    orch = Orchestrator(
        provider, registry, ToolRegistry(), verbose_logging=False, max_retries_per_task=2, retry_backoff_base_seconds=0.01
    )
    result = await orch.run("goal")

    assert result.succeeded
    assert agent.call_count == 2
    retrying_events = [e for e in result.events if e["tag"] == "TASK_RETRYING"]
    assert len(retrying_events) == 1
    assert retrying_events[0]["retry_count"] == 1
    assert "backoff_seconds" in retrying_events[0]


@pytest.mark.asyncio
async def test_retry_backoff_increases_with_attempt_number():
    registry = AgentRegistry()
    orch = Orchestrator(
        MockProvider(), registry, ToolRegistry(), retry_backoff_base_seconds=1.0, max_retry_backoff_seconds=100.0
    )
    assert orch._retry_backoff_seconds(1) == 1.0
    assert orch._retry_backoff_seconds(2) == 2.0
    assert orch._retry_backoff_seconds(3) == 4.0


@pytest.mark.asyncio
async def test_retry_backoff_is_capped():
    registry = AgentRegistry()
    orch = Orchestrator(MockProvider(), registry, ToolRegistry(), retry_backoff_base_seconds=1.0, max_retry_backoff_seconds=3.0)
    assert orch._retry_backoff_seconds(10) == 3.0


@pytest.mark.asyncio
async def test_exhausted_retries_do_not_retry_indefinitely():
    from tests.doubles import always_fails

    registry = AgentRegistry()
    agent = StubAgent("research_agent", ["research"], always_fails("permanent"))
    registry.register(agent)
    plan = {"tasks": [{"id": "t1", "objective": "a", "capability": "research", "dependencies": [], "required_tools": []}]}
    provider = MockProvider(responder=plan_responder(plan))

    orch = Orchestrator(
        provider, registry, ToolRegistry(), verbose_logging=False, max_retries_per_task=2, max_replans=0, retry_backoff_base_seconds=0.001
    )
    result = await orch.run("goal")

    assert not result.succeeded
    # 1 initial attempt + 2 retries = 3 total calls, never more
    assert agent.call_count == 3


# -- Crash recovery: the exact scenario from the spec ----------------------
#
#   A
#   |-- B
#   |-- C
#   |-- D
#
#   A COMPLETED, B COMPLETED, C RUNNING, D PENDING -> simulated shutdown ->
#   restart -> resume_execution(execution_id) -> A and B are NOT re-executed,
#   C is safely recovered, D waits then runs, no duplicate completed work.


@pytest.mark.asyncio
async def test_crash_recovery_matches_the_exact_spec_scenario():
    store = InMemoryStateStore()

    graph = TaskGraph([
        Task(id="A", objective="a", capability="research"),
        Task(id="B", objective="b", capability="research", dependencies=["A"]),
        Task(id="C", objective="c", capability="research", dependencies=["A"]),
        Task(id="D", objective="d", capability="research", dependencies=["A"]),
    ])
    graph.tasks["A"].status = TaskStatus.SUCCEEDED
    graph.tasks["A"].result = AgentOutput(success=True, content="A's result, long enough to pass.")
    graph.tasks["B"].status = TaskStatus.SUCCEEDED
    graph.tasks["B"].result = AgentOutput(success=True, content="B's result, long enough to pass.")
    graph.tasks["C"].status = TaskStatus.RUNNING  # mid-flight when the crash happened
    # D stays PENDING

    state = ExecutionState.create("goal", execution_id="exec_crash_scenario")
    state.current_plan = graph.to_dict()
    state.completed_tasks = ["A", "B"]
    store.save(state)

    registry = AgentRegistry()
    agent = StubAgent("research_agent", ["research"], always_succeeds("recovered output, long enough to pass eval."))
    registry.register(agent)
    provider = MockProvider(responder=lambda s, m, t: "final")

    orch = Orchestrator(provider, registry, ToolRegistry(), state_store=store, verbose_logging=False)
    result = await orch.resume_execution("exec_crash_scenario")

    assert result.succeeded
    # A and B were never re-executed - the stub agent was only invoked for
    # the tasks that genuinely still needed to run: C (recovered) and D.
    assert agent.call_count == 2
    assert {i.task_context["task_id"] for i in agent.received_inputs} == {"C", "D"}

    assert result.graph.tasks["A"].status == TaskStatus.SUCCEEDED
    assert result.graph.tasks["A"].result.content == "A's result, long enough to pass."  # untouched, original result preserved
    assert result.graph.tasks["B"].status == TaskStatus.SUCCEEDED
    assert result.graph.tasks["C"].status == TaskStatus.SUCCEEDED  # recovered
    assert result.graph.tasks["D"].status == TaskStatus.SUCCEEDED  # ran after waiting for A


# -- Replan integration + failure isolation through the full orchestrator --


@pytest.mark.asyncio
async def test_permanent_failure_blocks_only_its_dependents_others_still_complete():
    registry = AgentRegistry()
    from tests.doubles import always_fails

    good_agent = StubAgent("research_agent", ["research"], always_succeeds("fine output, long enough to pass eval."))
    bad_agent = StubAgent("writer_agent", ["writing"], always_fails("permanent failure"))
    registry.register(good_agent)
    registry.register(bad_agent)

    plan = {
        "tasks": [
            {"id": "A", "objective": "a", "capability": "research", "dependencies": [], "required_tools": []},
            {"id": "B", "objective": "b", "capability": "research", "dependencies": ["A"], "required_tools": []},
            {"id": "C", "objective": "c", "capability": "writing", "dependencies": ["A"], "required_tools": []},
            {"id": "D", "objective": "d", "capability": "research", "dependencies": ["C"], "required_tools": []},
        ]
    }
    provider = MockProvider(responder=plan_responder(plan))
    orch = Orchestrator(
        provider, registry, ToolRegistry(), verbose_logging=False, max_retries_per_task=0, max_replans=0
    )
    result = await orch.run("goal")

    assert not result.succeeded
    assert result.graph.tasks["A"].status == TaskStatus.SUCCEEDED
    assert result.graph.tasks["B"].status == TaskStatus.SUCCEEDED  # independent of the failure - still ran
    assert result.graph.tasks["C"].status == TaskStatus.FAILED
    assert result.graph.tasks["D"].status == TaskStatus.BLOCKED  # depends on the failed task
    assert good_agent.call_count == 2  # A and B - never invoked for C's subtree


@pytest.mark.asyncio
async def test_replan_hook_is_reachable_and_can_splice_in_a_recovery_task():
    """Confirms the clean integration point: a permanently-failed task
    triggers the same Planner.replan() path used since Phase 3, and any
    newly spliced-in task is picked up by the DAG scheduler like any other
    ready task - no bespoke replan-specific scheduling logic needed."""
    registry = AgentRegistry()
    from tests.doubles import always_fails

    agent = StubAgent("research_agent", ["research"], always_fails("boom"))
    registry.register(agent)

    plan = {"tasks": [{"id": "t1", "objective": "a", "capability": "research", "dependencies": [], "required_tools": []}]}
    replan = {
        "reason": "recover t1",
        "operations": [
            {
                "op": "replace_task",
                "task_id": "t1",
                "task": {"id": "t1_recovery", "objective": "recover", "capability": "research", "dependencies": [], "required_tools": []},
            }
        ],
    }

    def responder(system, messages, tools):
        last = messages[-1].content
        if last.startswith("User goal:"):
            return json.dumps(plan)
        if "Failed task:" in last:
            return json.dumps(replan)
        return "final"

    provider = MockProvider(responder=responder)
    orch = Orchestrator(
        provider, registry, ToolRegistry(), verbose_logging=False, max_retries_per_task=0, max_replans=1
    )
    result = await orch.run("goal")

    recovery_ids = [tid for tid in result.graph.tasks if tid.startswith("t1_recovery")]
    assert recovery_ids, "replanned recovery task was not spliced into the graph"
    assert orch.state.replans_used == 1


# -- Parallel context safety: no cross-task contamination -------------------


@pytest.mark.asyncio
async def test_concurrent_siblings_never_see_each_others_private_context():
    """B and C run concurrently (both depend only on A); B must not receive
    C's output/context and vice versa - only what each explicitly depends
    on."""
    registry = AgentRegistry()

    class RecordingAgent(StubAgent):
        async def execute(self, agent_input: AgentInput) -> AgentOutput:
            self.received_inputs.append(agent_input)
            self.call_count += 1
            return AgentOutput(success=True, content=f"output from {self.id}, long enough to pass evaluation threshold.")

    research = RecordingAgent("research_agent", ["research"], lambda *a: None)
    analysis_b = RecordingAgent("analysis_b_agent", ["analysis_b"], lambda *a: None)
    analysis_c = RecordingAgent("analysis_c_agent", ["analysis_c"], lambda *a: None)
    registry.register(research)
    registry.register(analysis_b)
    registry.register(analysis_c)

    plan = {
        "tasks": [
            {"id": "A", "objective": "a", "capability": "research", "dependencies": [], "required_tools": []},
            {"id": "B", "objective": "b", "capability": "analysis_b", "dependencies": ["A"], "required_tools": []},
            {"id": "C", "objective": "c", "capability": "analysis_c", "dependencies": ["A"], "required_tools": []},
        ]
    }
    provider = MockProvider(responder=plan_responder(plan))
    orch = Orchestrator(provider, registry, ToolRegistry(), verbose_logging=False)
    result = await orch.run("goal")

    assert result.succeeded
    b_input = analysis_b.received_inputs[0]
    c_input = analysis_c.received_inputs[0]

    # Both only ever see A's output (their real dependency) - never each other.
    assert set(b_input.upstream_outputs) == {"A"}
    assert set(c_input.upstream_outputs) == {"A"}
    assert "output from analysis_c_agent" not in json.dumps(b_input.upstream_outputs)
    assert "output from analysis_b_agent" not in json.dumps(c_input.upstream_outputs)
