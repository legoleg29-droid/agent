"""Stress test: >=20 tasks with a mixed dependency shape, concurrency
capped at 5. Verifies the concurrency cap is never exceeded, every valid
task eventually completes, nothing executes twice, and the scheduler
terminates (doesn't hang)."""

import asyncio
import random

import pytest

from orchestrator.core.retry import Action
from orchestrator.core.scheduler import DAGScheduler, ResourceLimits, SchedulerResult
from orchestrator.core.task_graph import Task, TaskGraph, TaskStatus


def build_25_task_graph() -> TaskGraph:
    """5 independent chains of 5 tasks each, so there's real width (up to
    5 ready at once) as well as depth (dependency ordering to respect)."""
    tasks = []
    for chain in range(5):
        for depth in range(5):
            task_id = f"c{chain}_t{depth}"
            deps = [f"c{chain}_t{depth - 1}"] if depth > 0 else []
            tasks.append(Task(id=task_id, objective="x", capability="x", dependencies=deps))
    return TaskGraph(tasks)


@pytest.mark.asyncio
async def test_stress_25_tasks_concurrency_5():
    graph = build_25_task_graph()
    assert len(graph.tasks) == 25

    running = 0
    max_running = 0
    call_counts: dict[str, int] = {}
    rng = random.Random(42)

    async def execute(task: Task) -> Action:
        nonlocal running, max_running
        call_counts[task.id] = call_counts.get(task.id, 0) + 1
        running += 1
        max_running = max(max_running, running)
        await asyncio.sleep(rng.uniform(0.001, 0.01))
        running -= 1
        task.status = TaskStatus.SUCCEEDED
        return Action.CONTINUE

    scheduler = DAGScheduler(graph, execute_task=execute, resource_limits=ResourceLimits(max_concurrent_tasks=5))
    result = await asyncio.wait_for(scheduler.run(), timeout=10)  # scheduler must terminate

    assert result == SchedulerResult.COMPLETED
    assert max_running <= 5, f"concurrency cap violated: saw {max_running} running at once"
    assert all(t.status == TaskStatus.SUCCEEDED for t in graph.tasks.values())
    assert all(count == 1 for count in call_counts.values()), f"a task executed more than once: {call_counts}"
    assert len(call_counts) == 25
    assert scheduler.metrics.completed_tasks == 25
    assert scheduler.metrics.peak_concurrency <= 5
    assert scheduler.metrics.peak_concurrency >= 2  # sanity: real concurrency actually happened


@pytest.mark.asyncio
async def test_stress_with_some_permanent_failures_stays_consistent():
    """Mix in a few permanent failures scattered across independent chains -
    each failure should only block its own chain's remaining tasks, and
    the scheduler still terminates cleanly with a consistent final state."""
    graph = build_25_task_graph()
    fail_at = {"c1_t2", "c3_t0"}
    call_counts: dict[str, int] = {}

    async def execute(task: Task) -> Action:
        call_counts[task.id] = call_counts.get(task.id, 0) + 1
        await asyncio.sleep(0.001)
        if task.id in fail_at:
            task.status = TaskStatus.FAILED
            task.error = "simulated permanent failure"
            return Action.ABORT
        task.status = TaskStatus.SUCCEEDED
        return Action.CONTINUE

    scheduler = DAGScheduler(graph, execute_task=execute, resource_limits=ResourceLimits(max_concurrent_tasks=5))
    result = await asyncio.wait_for(scheduler.run(), timeout=10)

    assert result == SchedulerResult.FAILED
    assert all(count == 1 for count in call_counts.values())

    # chain 1: t0, t1 succeed, t2 fails, t3/t4 blocked
    assert graph.tasks["c1_t0"].status == TaskStatus.SUCCEEDED
    assert graph.tasks["c1_t1"].status == TaskStatus.SUCCEEDED
    assert graph.tasks["c1_t2"].status == TaskStatus.FAILED
    assert graph.tasks["c1_t3"].status == TaskStatus.BLOCKED
    assert graph.tasks["c1_t4"].status == TaskStatus.BLOCKED

    # chain 3: t0 fails immediately, everything downstream is blocked
    assert graph.tasks["c3_t0"].status == TaskStatus.FAILED
    for depth in range(1, 5):
        assert graph.tasks[f"c3_t{depth}"].status == TaskStatus.BLOCKED

    # untouched chains (0, 2, 4) complete fully - failure isolation held
    for chain in (0, 2, 4):
        for depth in range(5):
            assert graph.tasks[f"c{chain}_t{depth}"].status == TaskStatus.SUCCEEDED

    assert graph.is_complete()
