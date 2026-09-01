import json

import pytest

from orchestrator.agents.base import AgentInput, AgentOutput
from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.evaluation_models import EvaluationResult, EvaluationStatus
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.core.repair import RepairManager
from orchestrator.core.task_graph import Task, TaskStatus
from orchestrator.providers.mock_provider import MockProvider
from orchestrator.tools.registry import ToolRegistry
from tests.doubles import StubAgent


def plan_responder(plan):
    return lambda s, m, t: json.dumps(plan) if m[-1].content.startswith("User goal:") else "final synthesized result"


# -- RepairManager unit test -------------------------------------------------


@pytest.mark.asyncio
async def test_repair_manager_passes_failure_feedback_to_the_agent():
    received: list[AgentInput] = []

    async def execute(agent_input: AgentInput) -> AgentOutput:
        received.append(agent_input)
        return AgentOutput(success=True, content="fixed")

    class FakeAgent:
        id = "fake"

        async def execute(self, agent_input):
            return await execute(agent_input)

    task = Task(id="t1", objective="obj", capability="research")
    task.repair_count = 0
    base_input = AgentInput(objective="obj", expected_output="", upstream_outputs={"dep": "dep output"})
    evaluation = EvaluationResult(
        status=EvaluationStatus.REPAIR_REQUIRED,
        failed_criteria=["contains 'fibonacci'"],
        reasons=["missing required term"],
        evidence=["contains(fibonacci)=False"],
    )

    manager = RepairManager()
    output = await manager.repair(
        agent=FakeAgent(),
        task=task,
        previous_output=AgentOutput(success=True, content="old output"),
        evaluation=evaluation,
        base_input=base_input,
    )

    assert output.success and output.content == "fixed"
    assert len(received) == 1
    fb = received[0].repair_feedback
    assert fb["previous_output"] == "old output"
    assert fb["failed_criteria"] == ["contains 'fibonacci'"]
    assert fb["reasons"] == ["missing required term"]
    assert received[0].upstream_outputs == {"dep": "dep output"}  # base context is preserved


# -- Orchestrator-level repair loop ------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_repairs_in_place_without_restarting_the_task():
    registry = AgentRegistry()

    def behavior(_call_index, agent_input: AgentInput) -> AgentOutput:
        if agent_input.repair_feedback:
            return AgentOutput(success=True, content="Now mentions fibonacci explicitly, as required.")
        return AgentOutput(success=True, content="An unrelated but sufficiently long answer about something else.")

    agent = StubAgent("coding_agent", ["coding"], behavior)
    registry.register(agent)

    plan = {
        "tasks": [
            {
                "id": "t1",
                "objective": "write fibonacci",
                "capability": "coding",
                "dependencies": [],
                "required_tools": [],
                "acceptance_criteria": [{"type": "contains", "text": "fibonacci"}, {"type": "min_length", "length": 10}],
            }
        ]
    }
    provider = MockProvider(responder=plan_responder(plan))
    orch = Orchestrator(
        provider, registry, ToolRegistry(), verbose_logging=False, max_retries_per_task=2, max_repairs_per_task=2
    )
    result = await orch.run("goal")

    assert result.succeeded
    assert agent.call_count == 2  # original attempt + exactly one repair
    task = result.graph.tasks["t1"]
    assert task.status == TaskStatus.SUCCEEDED
    assert task.repair_count == 1
    repair_started = [e for e in result.events if e["tag"] == "REPAIR_STARTED"]
    repair_completed = [e for e in result.events if e["tag"] == "REPAIR_COMPLETED"]
    assert len(repair_started) == 1
    assert len(repair_completed) == 1
    # the repair history is recorded on the persisted execution state, not just events
    assert result.execution_state.repair_history
    assert result.execution_state.repair_history[0]["outcome"] == "succeeded"


@pytest.mark.asyncio
async def test_repair_budget_exhausted_escalates_and_terminates_without_looping_forever():
    registry = AgentRegistry()

    def behavior(_call_index, _agent_input: AgentInput) -> AgentOutput:
        # never actually fixes the problem, no matter how many times it's asked
        return AgentOutput(success=True, content="An answer that never contains the required term but is long enough.")

    agent = StubAgent("coding_agent", ["coding"], behavior)
    registry.register(agent)

    plan = {
        "tasks": [
            {
                "id": "t1",
                "objective": "write fibonacci",
                "capability": "coding",
                "dependencies": [],
                "required_tools": [],
                "acceptance_criteria": [{"type": "contains", "text": "fibonacci"}, {"type": "min_length", "length": 10}],
            }
        ]
    }
    provider = MockProvider(responder=plan_responder(plan))
    orch = Orchestrator(
        provider,
        registry,
        ToolRegistry(),
        verbose_logging=False,
        max_retries_per_task=0,
        max_repairs_per_task=1,
        max_replans=0,
    )
    result = await orch.run("goal")

    assert not result.succeeded
    task = result.graph.tasks["t1"]
    assert task.status == TaskStatus.FAILED
    assert task.repair_count == 1  # bounded by max_repairs_per_task, never unbounded
    assert agent.call_count == 2  # original attempt + exactly one repair attempt, then it stopped
