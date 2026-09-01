"""DAG Scheduler: concurrency-bounded, dependency-aware task dispatch.

Deliberately decoupled from ``ContextManager``/``MemoryManager``/
``ExecutionState`` - it operates purely on a ``TaskGraph`` plus one
injected async callback that actually executes a task and returns the
Phase 1-3 ``Action`` outcome (continue/retry/replan/abort). That keeps it
independently testable with plain mock callables (see
``tests/test_scheduler.py``) while the ``Orchestrator`` supplies the real
callback (bound to ``_run_task``, which already owns routing, tool
execution, evaluation, and retry backoff).

Concurrency uses native ``asyncio`` primitives only - a ``Semaphore`` to
bound how many tasks run at once, and ``asyncio.wait(..., FIRST_COMPLETED)``
to drive the dispatch loop as tasks finish. No thread pools, no external
scheduling library.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from orchestrator.core.logging_utils import EventLog
from orchestrator.core.retry import Action
from orchestrator.core.task_graph import Task, TaskGraph, TaskStatus

TaskExecutor = Callable[[Task], Awaitable[Action]]
ReplanHook = Callable[[Task], Awaitable[bool]]
Checkpointer = Callable[[], None]


@dataclass
class ResourceLimits:
    """Clean interface for the resource caps the scheduler respects today
    (``max_concurrent_tasks``, ``max_execution_time_seconds``,
    ``max_total_tasks``) plus fields reserved for limits enforced
    elsewhere/later (``max_concurrent_agents`` defaults to
    ``max_concurrent_tasks`` since one task runs one agent;
    ``max_concurrent_tool_calls``/``max_tokens`` are not yet enforced
    anywhere - deliberately not over-engineered ahead of need)."""

    max_concurrent_tasks: int = 5
    max_concurrent_agents: int | None = None
    max_concurrent_tool_calls: int | None = None
    max_execution_time_seconds: float | None = None
    max_total_tasks: int | None = None
    max_retries: int | None = None  # informational; enforced per-task via Task.max_retries
    max_tokens: int | None = None  # reserved for future cost tracking

    def __post_init__(self) -> None:
        if self.max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks must be >= 1")
        if self.max_concurrent_agents is None:
            self.max_concurrent_agents = self.max_concurrent_tasks


@dataclass
class SchedulerMetrics:
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    blocked_tasks: int = 0
    cancelled_tasks: int = 0
    retried_tasks: int = 0
    running_tasks: int = 0
    peak_concurrency: int = 0
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def execution_duration_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "blocked_tasks": self.blocked_tasks,
            "cancelled_tasks": self.cancelled_tasks,
            "retried_tasks": self.retried_tasks,
            "peak_concurrency": self.peak_concurrency,
            "execution_duration_seconds": self.execution_duration_seconds,
        }


class SchedulerResult(str, Enum):
    COMPLETED = "completed"          # every task reached a successful/terminal-ok state
    FAILED = "failed"                # at least one task failed permanently
    BLOCKED = "blocked"              # nothing failed outright, but some tasks are blocked on a failed dependency
    CANCELLED = "cancelled"
    PAUSED = "paused"


class SchedulerCancelled(Exception):
    """Not raised for control flow within the scheduler itself - available
    for callers that want to treat a CANCELLED result as an error."""


class DAGScheduler:
    def __init__(
        self,
        graph: TaskGraph,
        *,
        execute_task: TaskExecutor,
        resource_limits: ResourceLimits | None = None,
        event_log: EventLog | None = None,
        execution_id: str | None = None,
        checkpoint: Checkpointer | None = None,
        on_replan_needed: ReplanHook | None = None,
    ) -> None:
        self.graph = graph
        self.execute_task = execute_task
        self.resource_limits = resource_limits or ResourceLimits()
        self.event_log = event_log
        self.execution_id = execution_id
        self.checkpoint = checkpoint
        self.on_replan_needed = on_replan_needed

        if self.resource_limits.max_total_tasks is not None and len(graph.tasks) > self.resource_limits.max_total_tasks:
            raise ValueError(
                f"Graph has {len(graph.tasks)} tasks, exceeding max_total_tasks="
                f"{self.resource_limits.max_total_tasks}"
            )

        self._semaphore = asyncio.Semaphore(self.resource_limits.max_concurrent_tasks)
        self.metrics = SchedulerMetrics()
        self._cancel_requested = False
        self._pause_requested = False

    # -- External control (safe to call from another coroutine/task) ----

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def request_pause(self) -> None:
        self._pause_requested = True

    def resume(self) -> None:
        self._pause_requested = False

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_requested

    @property
    def is_paused(self) -> bool:
        return self._pause_requested

    # -- Main loop --------------------------------------------------------

    async def run(self) -> SchedulerResult:
        self.metrics.started_at = time.time()
        self.metrics.total_tasks = len(self.graph.tasks)
        self._emit("SCHEDULER_STARTED", f"Scheduler starting with {self.metrics.total_tasks} task(s)")

        in_flight: dict[asyncio.Task, Task] = {}
        try:
            while True:
                if self._time_limit_exceeded():
                    self._cancel_requested = True

                if self._cancel_requested:
                    result = await self._handle_cancellation(in_flight)
                    return result

                if not self._pause_requested:
                    self._dispatch_ready_tasks(in_flight)

                newly_blocked = self.graph.mark_blocked_by_failed_dependencies()
                for task in newly_blocked:
                    self.metrics.blocked_tasks += 1
                    self._emit("TASK_BLOCKED", f"Task '{task.id}' blocked - a dependency failed permanently", task_id=task.id, status="blocked")
                    self._checkpoint()

                if not in_flight:
                    if self._pause_requested:
                        self._emit("SCHEDULER_WAITING", "Scheduler paused - no tasks in flight", status="paused")
                        return SchedulerResult.PAUSED
                    break  # nothing running, nothing new became ready -> done

                self._emit(
                    "SCHEDULER_WAITING",
                    f"Waiting on {len(in_flight)} in-flight task(s)",
                    extra={"in_flight": [t.id for t in in_flight.values()]},
                )
                done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)
                for fut in done:
                    in_flight.pop(fut, None)
                    exc = fut.exception()
                    if exc is not None:
                        raise exc  # never silently swallow a scheduler-level failure
        finally:
            self.metrics.finished_at = time.time()

        self.graph.mark_blocked_by_failed_dependencies()
        result = self._final_result()
        self._emit(
            "SCHEDULER_COMPLETED",
            f"Scheduler finished: {result.value}",
            status=result.value,
            extra=self.metrics.to_dict(),
        )
        return result

    def _dispatch_ready_tasks(self, in_flight: dict[asyncio.Task, Task]) -> None:
        for task in self.graph.get_ready_tasks():
            # Claim synchronously (no await between the status check inside
            # get_ready_tasks() and this assignment) so a task can never be
            # claimed twice by overlapping scheduler ticks/retries/resume.
            task.status = TaskStatus.READY
            self._emit("TASK_READY", f"Task '{task.id}' is ready", task_id=task.id, status="ready")
            self._checkpoint()
            fut = asyncio.ensure_future(self._run_one(task))
            in_flight[fut] = task

    async def _run_one(self, task: Task) -> Action:
        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            self.metrics.running_tasks += 1
            self.metrics.peak_concurrency = max(self.metrics.peak_concurrency, self.metrics.running_tasks)
            started = time.perf_counter()
            self._emit("TASK_STARTED", f"Task '{task.id}' started", task_id=task.id, agent_id=task.agent_id, status="running", retry_count=task.retry_count)
            try:
                action = await self.execute_task(task)
            finally:
                self.metrics.running_tasks -= 1
            duration_ms = (time.perf_counter() - started) * 1000

            if action == Action.CONTINUE:
                self.metrics.completed_tasks += 1
                self._emit("TASK_COMPLETED", f"Task '{task.id}' completed", task_id=task.id, agent_id=task.agent_id, status="succeeded", duration_ms=round(duration_ms, 2))
            elif action == Action.REPLAN:
                self._emit("TASK_FAILED", f"Task '{task.id}' failed - replanning", task_id=task.id, agent_id=task.agent_id, status="replan_required", duration_ms=round(duration_ms, 2))
                replanned = await self.on_replan_needed(task) if self.on_replan_needed else False
                if not replanned:
                    self.metrics.failed_tasks += 1
            else:  # ABORT
                self.metrics.failed_tasks += 1
                self._emit("TASK_FAILED", f"Task '{task.id}' failed permanently", task_id=task.id, agent_id=task.agent_id, status="failed", duration_ms=round(duration_ms, 2))
            self._checkpoint()
            return action

    async def _handle_cancellation(self, in_flight: dict[asyncio.Task, Task]) -> SchedulerResult:
        for fut in in_flight:
            fut.cancel()
        if in_flight:
            await asyncio.gather(*in_flight.keys(), return_exceptions=True)
        cancelled = self.graph.mark_remaining_as_cancelled()
        self.metrics.cancelled_tasks += len(cancelled)
        for task in cancelled:
            self._emit("TASK_CANCELLED", f"Task '{task.id}' cancelled", task_id=task.id, status="cancelled")
        self._checkpoint()
        self._emit("SCHEDULER_COMPLETED", "Scheduler cancelled", status="cancelled", extra=self.metrics.to_dict())
        return SchedulerResult.CANCELLED

    def _final_result(self) -> SchedulerResult:
        if self.graph.failed_tasks():
            return SchedulerResult.FAILED
        if self.graph.blocked_tasks():
            return SchedulerResult.BLOCKED
        return SchedulerResult.COMPLETED

    def _time_limit_exceeded(self) -> bool:
        limit = self.resource_limits.max_execution_time_seconds
        if limit is None or self.metrics.started_at is None:
            return False
        return (time.time() - self.metrics.started_at) > limit

    def _checkpoint(self) -> None:
        if self.checkpoint:
            self.checkpoint()

    def _emit(self, tag: str, message: str, **fields: Any) -> None:
        if not self.event_log:
            return
        extra = fields.pop("extra", {}) or {}
        if self.execution_id:
            extra = {"execution_id": self.execution_id, **extra}
        self.event_log.emit(tag, message, extra=extra, **fields)
