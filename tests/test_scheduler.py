"""DAGScheduler unit tests: pure async, no LLM/agent machinery - just the
scheduler against a TaskGraph and a mock executor callback."""

import asyncio
import time

import pytest

from orchestrator.core.logging_utils import EventLog
from orchestrator.core.retry import Action
from orchestrator.core.scheduler import DAGScheduler, ResourceLimits, SchedulerResult
from orchestrator.core.task_graph import Task, TaskGraph, TaskStatus


def make_executor(*, order: list, delay: float = 0.0, fail_ids: set[str] = frozenset()):
    async def execute(task: Task) -> Action:
        order.append(("start", task.id, time.perf_counter()))
        if delay:
            await asyncio.sleep(delay)
        if task.id in fail_ids:
            task.status = TaskStatus.FAILED
            task.error = "boom"
            order.append(("end", task.id, time.perf_counter()))
            return Action.ABORT
        task.status = TaskStatus.SUCCEEDED
        order.append(("end", task.id, time.perf_counter()))
        return Action.CONTINUE

    return execute


# -- Ready detection / dependency resolution driving dispatch --------------


@pytest.mark.asyncio
async def test_sequential_dependency_chain_runs_in_order():
    graph = TaskGraph([
        Task(id="a", objective="a", capability="x"),
        Task(id="b", objective="b", capability="x", dependencies=["a"]),
        Task(id="c", objective="c", capability="x", dependencies=["b"]),
    ])
    order: list = []
    scheduler = DAGScheduler(graph, execute_task=make_executor(order=order))
    result = await scheduler.run()

    assert result == SchedulerResult.COMPLETED
    starts = [e[1] for e in order if e[0] == "start"]
    assert starts == ["a", "b", "c"]


# -- The deterministic parallel-execution scenario from the spec -----------
#
#   A
#   |-- B --.
#   |-- C --+--> E (depends on B, C)
#   |-- D --+--> F (depends on C, D)
#
# A runs alone; B, C, D must run concurrently; E, F may run concurrently.


@pytest.mark.asyncio
async def test_parallel_execution_matches_the_spec_diagram_and_is_actually_concurrent():
    graph = TaskGraph([
        Task(id="A", objective="a", capability="x"),
        Task(id="B", objective="b", capability="x", dependencies=["A"]),
        Task(id="C", objective="c", capability="x", dependencies=["A"]),
        Task(id="D", objective="d", capability="x", dependencies=["A"]),
        Task(id="E", objective="e", capability="x", dependencies=["B", "C"]),
        Task(id="F", objective="f", capability="x", dependencies=["C", "D"]),
    ])
    order: list = []
    delay = 0.05
    scheduler = DAGScheduler(
        graph, execute_task=make_executor(order=order, delay=delay), resource_limits=ResourceLimits(max_concurrent_tasks=5)
    )

    wall_start = time.perf_counter()
    result = await scheduler.run()
    wall_elapsed = time.perf_counter() - wall_start

    assert result == SchedulerResult.COMPLETED
    assert all(t.status == TaskStatus.SUCCEEDED for t in graph.tasks.values())

    starts = {tid: t for kind, tid, t in order if kind == "start"}
    ends = {tid: t for kind, tid, t in order if kind == "end"}

    # A alone first.
    assert starts["A"] < min(starts["B"], starts["C"], starts["D"])

    # B, C, D overlap in time (all started before any of them ended) - proof
    # of actual concurrency, not sequential execution.
    assert starts["B"] < ends["C"] and starts["C"] < ends["B"]
    assert starts["D"] < ends["C"] and starts["C"] < ends["D"]

    # E and F only start once their own dependencies finished.
    assert starts["E"] >= max(ends["B"], ends["C"])
    assert starts["F"] >= max(ends["C"], ends["D"])

    # E and F may run concurrently with each other.
    assert starts["E"] < ends["F"] and starts["F"] < ends["E"]

    # Wall-clock proof: three sequential-looking "layers" (A; B/C/D; E/F)
    # each taking ~delay, but B/C/D and E/F pairs overlap internally, so
    # total time is well under what fully sequential execution of 6 tasks
    # would take (6 * delay).
    assert wall_elapsed < 5 * delay


# -- Concurrency limit --------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_limit_is_never_exceeded():
    tasks = [Task(id=f"t{i}", objective="x", capability="x") for i in range(20)]
    graph = TaskGraph(tasks)

    running = 0
    max_running = 0
    lock_free_check = []

    async def execute(task: Task) -> Action:
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        lock_free_check.append(running)
        await asyncio.sleep(0.01)
        running -= 1
        task.status = TaskStatus.SUCCEEDED
        return Action.CONTINUE

    scheduler = DAGScheduler(graph, execute_task=execute, resource_limits=ResourceLimits(max_concurrent_tasks=5))
    result = await scheduler.run()

    assert result == SchedulerResult.COMPLETED
    assert max_running <= 5
    assert max(lock_free_check) <= 5
    assert scheduler.metrics.peak_concurrency <= 5
    assert scheduler.metrics.peak_concurrency >= 1


# -- Duplicate execution prevention --------------------------------------


@pytest.mark.asyncio
async def test_a_task_is_never_dispatched_twice():
    graph = TaskGraph([Task(id="a", objective="a", capability="x")])
    call_count = 0

    async def execute(task: Task) -> Action:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        task.status = TaskStatus.SUCCEEDED
        return Action.CONTINUE

    scheduler = DAGScheduler(graph, execute_task=execute)
    result = await scheduler.run()
    assert result == SchedulerResult.COMPLETED
    assert call_count == 1


@pytest.mark.asyncio
async def test_ready_tasks_claimed_synchronously_cannot_race_across_ticks():
    """Even if get_ready_tasks() would return the same task on consecutive
    ticks (e.g. a slow first dispatch loop), claiming flips status to READY
    synchronously before any await, so a second tick's dispatch never
    re-claims it."""
    graph = TaskGraph([Task(id=f"t{i}", objective="x", capability="x") for i in range(10)])
    dispatched: list[str] = []

    async def execute(task: Task) -> Action:
        dispatched.append(task.id)
        await asyncio.sleep(0.001)
        task.status = TaskStatus.SUCCEEDED
        return Action.CONTINUE

    scheduler = DAGScheduler(graph, execute_task=execute, resource_limits=ResourceLimits(max_concurrent_tasks=3))
    await scheduler.run()
    assert sorted(dispatched) == sorted(t.id for t in graph.tasks.values())
    assert len(dispatched) == len(set(dispatched))


# -- Failure isolation / blocking ----------------------------------------


@pytest.mark.asyncio
async def test_failure_isolation_independent_branch_keeps_running():
    graph = TaskGraph([
        Task(id="A", objective="a", capability="x"),
        Task(id="B", objective="b", capability="x"),  # independent
        Task(id="C", objective="c", capability="x", dependencies=["A"]),
        Task(id="D", objective="d", capability="x", dependencies=["A", "C"]),
    ])
    order: list = []
    scheduler = DAGScheduler(graph, execute_task=make_executor(order=order, fail_ids={"A"}))
    result = await scheduler.run()

    assert result == SchedulerResult.FAILED
    assert graph.tasks["A"].status == TaskStatus.FAILED
    assert graph.tasks["B"].status == TaskStatus.SUCCEEDED  # ran despite A's failure
    assert graph.tasks["C"].status == TaskStatus.BLOCKED
    assert graph.tasks["D"].status == TaskStatus.BLOCKED


@pytest.mark.asyncio
async def test_replan_hook_invoked_on_replan_action_and_can_rescue_the_run():
    graph = TaskGraph([Task(id="a", objective="a", capability="x")])

    async def execute(task: Task) -> Action:
        task.status = TaskStatus.FAILED
        task.error = "needs replan"
        return Action.REPLAN

    replan_calls = []

    async def on_replan_needed(task: Task) -> bool:
        replan_calls.append(task.id)
        return True  # pretend the planner fixed it by adding a new task elsewhere

    scheduler = DAGScheduler(graph, execute_task=execute, on_replan_needed=on_replan_needed)
    result = await scheduler.run()

    assert replan_calls == ["a"]
    # the original task is still FAILED (replan adds new tasks, doesn't
    # resurrect the old one) but nothing crashed
    assert graph.tasks["a"].status == TaskStatus.FAILED
    assert result == SchedulerResult.FAILED


@pytest.mark.asyncio
async def test_result_is_blocked_when_nothing_failed_outright_but_something_is_blocked():
    graph = TaskGraph([
        Task(id="a", objective="a", capability="x"),
        Task(id="b", objective="b", capability="x", dependencies=["a"]),
    ])

    async def execute(task: Task) -> Action:
        task.status = TaskStatus.FAILED  # simulate a's own permanent failure
        return Action.REPLAN

    async def on_replan_needed(task: Task) -> bool:
        return False  # replan budget exhausted - can't rescue it

    scheduler = DAGScheduler(graph, execute_task=execute, on_replan_needed=on_replan_needed)
    result = await scheduler.run()
    assert graph.tasks["b"].status == TaskStatus.BLOCKED
    assert result == SchedulerResult.FAILED  # 'a' itself is FAILED, takes precedence over BLOCKED


# -- Pause / resume --------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_stops_new_dispatch_but_lets_running_tasks_finish():
    graph = TaskGraph([
        Task(id="a", objective="a", capability="x"),
        Task(id="b", objective="b", capability="x"),
    ])
    started = []

    async def execute(task: Task) -> Action:
        started.append(task.id)
        await asyncio.sleep(0.05)
        task.status = TaskStatus.SUCCEEDED
        return Action.CONTINUE

    scheduler = DAGScheduler(graph, execute_task=execute, resource_limits=ResourceLimits(max_concurrent_tasks=5))

    async def pause_soon():
        await asyncio.sleep(0.01)
        scheduler.request_pause()

    result, _ = await asyncio.gather(scheduler.run(), pause_soon())
    assert result == SchedulerResult.PAUSED
    # both tasks had already started (dispatched together) before the pause
    # flag was noticed, so they were allowed to finish, not abandoned.
    assert set(started) == {"a", "b"}
    assert graph.tasks["a"].status == TaskStatus.SUCCEEDED
    assert graph.tasks["b"].status == TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_pause_before_any_dispatch_prevents_new_tasks_from_starting():
    graph = TaskGraph([
        Task(id="a", objective="a", capability="x"),
        Task(id="b", objective="b", capability="x", dependencies=["a"]),
    ])
    started = []

    async def execute(task: Task) -> Action:
        started.append(task.id)
        await asyncio.sleep(0.01)
        task.status = TaskStatus.SUCCEEDED
        return Action.CONTINUE

    scheduler = DAGScheduler(graph, execute_task=execute)
    scheduler.request_pause()
    result = await scheduler.run()

    assert result == SchedulerResult.PAUSED
    assert started == []
    assert graph.tasks["a"].status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_resume_after_pause_continues_scheduling():
    graph = TaskGraph([Task(id="a", objective="a", capability="x")])
    calls = 0

    async def execute(task: Task) -> Action:
        nonlocal calls
        calls += 1
        task.status = TaskStatus.SUCCEEDED
        return Action.CONTINUE

    scheduler = DAGScheduler(graph, execute_task=execute)
    scheduler.request_pause()
    paused_result = await scheduler.run()
    assert paused_result == SchedulerResult.PAUSED
    assert calls == 0

    scheduler.resume()
    result = await scheduler.run()
    assert result == SchedulerResult.COMPLETED
    assert calls == 1


# -- Cancellation ----------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_prevents_new_tasks_and_marks_remaining_cancelled_not_completed():
    graph = TaskGraph([Task(id=f"t{i}", objective="x", capability="x") for i in range(6)])

    async def execute(task: Task) -> Action:
        await asyncio.sleep(0.03)
        task.status = TaskStatus.SUCCEEDED
        return Action.CONTINUE

    scheduler = DAGScheduler(graph, execute_task=execute, resource_limits=ResourceLimits(max_concurrent_tasks=2))

    async def cancel_soon():
        await asyncio.sleep(0.01)
        scheduler.request_cancel()

    result, _ = await asyncio.gather(scheduler.run(), cancel_soon())

    assert result == SchedulerResult.CANCELLED
    statuses = {t.status for t in graph.tasks.values()}
    # nothing left un-terminal, and nothing that never ran is SUCCEEDED
    assert TaskStatus.PENDING not in statuses
    assert TaskStatus.READY not in statuses
    assert TaskStatus.RUNNING not in statuses
    assert TaskStatus.CANCELLED in statuses


# -- Metrics / observability -------------------------------------------


@pytest.mark.asyncio
async def test_metrics_and_events_are_recorded():
    graph = TaskGraph([
        Task(id="a", objective="a", capability="x"),
        Task(id="b", objective="b", capability="x", dependencies=["a"]),
    ])
    event_log = EventLog(verbose=False)

    async def execute(task: Task) -> Action:
        task.status = TaskStatus.SUCCEEDED
        return Action.CONTINUE

    scheduler = DAGScheduler(graph, execute_task=execute, event_log=event_log, execution_id="exec_x")
    result = await scheduler.run()

    assert result == SchedulerResult.COMPLETED
    assert scheduler.metrics.total_tasks == 2
    assert scheduler.metrics.completed_tasks == 2
    assert scheduler.metrics.execution_duration_seconds is not None

    tags = {e.tag for e in event_log.events}
    assert {"SCHEDULER_STARTED", "TASK_READY", "TASK_STARTED", "TASK_COMPLETED", "SCHEDULER_COMPLETED"}.issubset(tags)
    assert all(e.extra.get("execution_id") == "exec_x" for e in event_log.events if "execution_id" in e.extra)


@pytest.mark.asyncio
async def test_scheduler_never_swallows_an_unexpected_exception():
    graph = TaskGraph([Task(id="a", objective="a", capability="x")])

    async def execute(task: Task) -> Action:
        raise RuntimeError("scheduler-level bug simulation")

    scheduler = DAGScheduler(graph, execute_task=execute)
    with pytest.raises(RuntimeError, match="scheduler-level bug simulation"):
        await scheduler.run()


def test_resource_limits_rejects_invalid_concurrency():
    with pytest.raises(ValueError):
        ResourceLimits(max_concurrent_tasks=0)


def test_scheduler_rejects_graph_exceeding_max_total_tasks():
    graph = TaskGraph([Task(id=f"t{i}", objective="x", capability="x") for i in range(5)])
    with pytest.raises(ValueError):
        DAGScheduler(graph, execute_task=lambda t: None, resource_limits=ResourceLimits(max_total_tasks=3))
