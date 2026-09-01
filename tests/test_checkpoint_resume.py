"""START -> SAVE STATE -> EXECUTE -> SAVE STATE -> CRASH/STOP -> RESTART ->
RESUME. Simulates a genuine crash (an exception that escapes run()
entirely, mid-execution) rather than a clean task failure, then proves a
fresh Orchestrator instance sharing the same StateStore can resume without
re-running the task that already completed.
"""

import json

import pytest

from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.core.task_graph import TaskStatus
from orchestrator.providers.mock_provider import MockProvider
from orchestrator.state.store import ExecutionNotFoundError, InMemoryStateStore
from orchestrator.tools.registry import ToolRegistry
from tests.doubles import StubAgent, always_succeeds

PLAN = {
    "tasks": [
        {"id": "t1", "objective": "Research competitors", "capability": "research", "dependencies": [], "required_tools": []},
        {"id": "t2", "objective": "Summarize findings", "capability": "writing", "dependencies": ["t1"], "required_tools": []},
    ]
}


def responder(system, messages, tools):
    if messages[-1].content.startswith("User goal:"):
        return json.dumps(PLAN)
    return "final synthesized result"


class SimulatedCrash(BaseException):
    """Deliberately NOT an Exception subclass, so it is never caught and
    converted into a normal task failure by ``_run_task`` - it genuinely
    escapes ``run()``, the way a killed process would. Avoids
    KeyboardInterrupt/SystemExit, which pytest's own runner treats
    specially."""


class CrashingAgent(StubAgent):
    async def execute(self, agent_input):
        self.call_count += 1
        raise SimulatedCrash("simulated crash")


@pytest.mark.asyncio
async def test_execution_interrupted_after_task_one_resumes_without_rerunning_it():
    registry = AgentRegistry()
    research_agent = StubAgent("research_agent", ["research"], always_succeeds("Findings about three competitors."))
    crashing_writer = CrashingAgent("writer_agent", ["writing"], always_succeeds())
    registry.register(research_agent)
    registry.register(crashing_writer)

    store = InMemoryStateStore()
    provider = MockProvider(responder=responder)
    orchestrator_1 = Orchestrator(provider, registry, ToolRegistry(), state_store=store, verbose_logging=False)

    with pytest.raises(SimulatedCrash):
        await orchestrator_1.run("Research three competitors and summarize the findings.", execution_id="exec_crash_test")

    # Task 1 completed and was checkpointed before the crash.
    persisted = store.load("exec_crash_test")
    assert persisted is not None
    assert "t1" in persisted.completed_tasks
    assert research_agent.call_count == 1

    # "Restart the runtime": a brand-new Orchestrator instance, sharing only
    # the persisted store, with a working (non-crashing) writer this time.
    registry_2 = AgentRegistry()
    working_writer = StubAgent("writer_agent", ["writing"], always_succeeds("Summary of the three competitors."))
    registry_2.register(research_agent)  # same instance - proves it is never called again
    registry_2.register(working_writer)
    orchestrator_2 = Orchestrator(provider, registry_2, ToolRegistry(), state_store=store, verbose_logging=False)

    result = await orchestrator_2.resume_execution("exec_crash_test")

    assert result.succeeded
    assert result.graph.tasks["t1"].status == TaskStatus.SUCCEEDED
    assert result.graph.tasks["t2"].status == TaskStatus.SUCCEEDED
    # Task 1's agent was NOT invoked again - only the pre-crash call counts.
    assert research_agent.call_count == 1
    # Task 2 (the one that never got to run before the crash) did run.
    assert working_writer.call_count == 1

    final_state = store.load("exec_crash_test")
    assert final_state.status.value == "completed"


@pytest.mark.asyncio
async def test_resume_of_unknown_execution_raises():
    registry = AgentRegistry()
    provider = MockProvider()
    orchestrator = Orchestrator(provider, registry, ToolRegistry(), state_store=InMemoryStateStore(), verbose_logging=False)

    with pytest.raises(ExecutionNotFoundError):
        await orchestrator.resume_execution("does_not_exist")


@pytest.mark.asyncio
async def test_resume_resets_a_task_that_was_running_at_crash_time():
    """A task that never got to report success or failure (e.g. the process
    died mid tool-call) must be retried on resume, not left stuck."""
    from orchestrator.core.task_graph import Task, TaskGraph
    from orchestrator.state.models import ExecutionState

    registry = AgentRegistry()
    agent = StubAgent("research_agent", ["research"], always_succeeds("recovered output, long enough to pass."))
    registry.register(agent)

    graph = TaskGraph([Task(id="t1", objective="a", capability="research")])
    graph.tasks["t1"].status = TaskStatus.RUNNING  # as if the crash happened mid-call

    state = ExecutionState.create("goal", execution_id="exec_stuck")
    state.current_plan = graph.to_dict()

    store = InMemoryStateStore()
    store.save(state)

    provider = MockProvider(responder=lambda s, m, t: "final")
    orchestrator = Orchestrator(provider, registry, ToolRegistry(), state_store=store, verbose_logging=False)
    result = await orchestrator.resume_execution("exec_stuck")

    assert result.succeeded
    assert agent.call_count == 1
    assert result.graph.tasks["t1"].status == TaskStatus.SUCCEEDED
