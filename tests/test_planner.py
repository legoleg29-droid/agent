import json

import pytest

from orchestrator.core.planner import Planner, PlanParseError
from orchestrator.core.task_graph import CycleError
from orchestrator.providers.mock_provider import MockProvider


def make_planner(plan_json: dict) -> Planner:
    provider = MockProvider(responder=lambda system, messages, tools: json.dumps(plan_json))
    return Planner(provider)


@pytest.mark.asyncio
async def test_plan_produces_valid_task_graph():
    plan_json = {
        "tasks": [
            {"id": "t1", "objective": "Research X", "capability": "research", "dependencies": [], "required_tools": ["web_search"], "expected_output": "facts"},
            {"id": "t2", "objective": "Analyze X", "capability": "analysis", "dependencies": ["t1"], "required_tools": [], "expected_output": "insights"},
        ]
    }
    planner = make_planner(plan_json)
    graph = await planner.plan("Do X", capabilities=["research", "analysis"], tools=["web_search"])

    assert set(graph.tasks) == {"t1", "t2"}
    assert graph.tasks["t2"].dependencies == ["t1"]


@pytest.mark.asyncio
async def test_plan_rejects_unknown_capability():
    plan_json = {"tasks": [{"id": "t1", "objective": "Do X", "capability": "nonexistent", "dependencies": [], "required_tools": []}]}
    planner = make_planner(plan_json)
    with pytest.raises(PlanParseError):
        await planner.plan("Do X", capabilities=["research"], tools=[])


@pytest.mark.asyncio
async def test_plan_rejects_unknown_tool():
    plan_json = {"tasks": [{"id": "t1", "objective": "Do X", "capability": "research", "required_tools": ["not_a_tool"]}]}
    planner = make_planner(plan_json)
    with pytest.raises(PlanParseError):
        await planner.plan("Do X", capabilities=["research"], tools=["web_search"])


@pytest.mark.asyncio
async def test_plan_rejects_unknown_dependency():
    plan_json = {"tasks": [{"id": "t1", "objective": "Do X", "capability": "research", "dependencies": ["ghost"]}]}
    planner = make_planner(plan_json)
    with pytest.raises(PlanParseError):
        await planner.plan("Do X", capabilities=["research"], tools=[])


@pytest.mark.asyncio
async def test_plan_rejects_cycles():
    plan_json = {
        "tasks": [
            {"id": "t1", "objective": "A", "capability": "research", "dependencies": ["t2"]},
            {"id": "t2", "objective": "B", "capability": "research", "dependencies": ["t1"]},
        ]
    }
    planner = make_planner(plan_json)
    with pytest.raises(PlanParseError):
        await planner.plan("Do X", capabilities=["research"], tools=[])


@pytest.mark.asyncio
async def test_plan_rejects_non_json_response():
    provider = MockProvider(responder=lambda s, m, t: "I refuse to produce JSON today.")
    planner = Planner(provider)
    with pytest.raises(PlanParseError):
        await planner.plan("Do X", capabilities=["research"], tools=[])


@pytest.mark.asyncio
async def test_replan_produces_a_minimal_patch_not_a_wholesale_plan():
    from orchestrator.core.task_graph import Task, TaskGraph, TaskStatus

    graph = TaskGraph()
    graph.add_task(Task(id="already_done", objective="prior work", capability="research", status=TaskStatus.SUCCEEDED))
    graph.add_task(Task(id="t1", objective="do the thing", capability="research", dependencies=["already_done"], status=TaskStatus.FAILED))

    patch_json = {
        "reason": "t1 needs a different approach",
        "operations": [
            {
                "op": "replace_task",
                "task_id": "t1",
                "task": {
                    "id": "t1_v2",
                    "objective": "Retry X differently",
                    "capability": "research",
                    "dependencies": ["already_done"],
                    "required_tools": [],
                },
            }
        ],
    }
    provider = MockProvider(responder=lambda system, messages, tools: json.dumps(patch_json))
    planner = Planner(provider)

    ops, reason = await planner.replan(
        "Do X",
        graph=graph,
        capabilities=["research"],
        tools=[],
        failed_task=graph.tasks["t1"],
        failure_reason="original task failed",
    )
    assert len(ops) == 1
    assert ops[0].task["id"] == "t1_v2"
    assert "already_done" in ops[0].task["dependencies"]
    assert reason == "t1 needs a different approach"
