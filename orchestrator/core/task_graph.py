"""Task representation and dependency graph / scheduler.

A ``Task`` is a unit of planned work targeting a capability (not a
hardcoded agent class). ``TaskGraph`` is a DAG over tasks that exposes
which tasks are currently ready to run (all dependencies satisfied) so the
scheduler can execute independent branches in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from orchestrator.agents.base import AgentOutput


class TaskStatus(str, Enum):
    PENDING = "pending"        # dependencies not yet all satisfied, or not yet claimed
    READY = "ready"             # dependencies satisfied, claimed by the scheduler, about to run
    RUNNING = "running"
    WAITING = "waiting"         # reserved: waiting on something external (e.g. human input) - not yet used
    RETRYING = "retrying"       # failed, backing off before another attempt
    SUCCEEDED = "succeeded"
    FAILED = "failed"           # this task itself permanently failed
    BLOCKED = "blocked"         # a dependency permanently failed - this task can never run as planned
    CANCELLED = "cancelled"     # execution was cancelled before this task ran
    SKIPPED = "skipped"         # legacy: cascaded during a full-graph abort (see mark_unreachable_as_skipped)


# Statuses that mean "this task will never run/change again".
TERMINAL_TASK_STATUSES = frozenset(
    {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED, TaskStatus.SKIPPED}
)


class CycleError(ValueError):
    pass


@dataclass
class Task:
    id: str
    objective: str
    capability: str
    dependencies: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    expected_output: str = ""
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 2
    result: AgentOutput | None = None
    error: str | None = None
    agent_id: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def attempt(self) -> int:
        """1-indexed attempt number, matching the persisted TaskState contract."""
        return self.retry_count + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "capability": self.capability,
            "dependencies": self.dependencies,
            "required_tools": self.required_tools,
            "expected_output": self.expected_output,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "agent_id": self.agent_id,
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Full round-trippable snapshot, used for state persistence/resume."""
        d = self.to_dict()
        d.update(
            {
                "max_retries": self.max_retries,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "error": self.error,
                "result": self.result.to_dict() if self.result is not None else None,
                "metadata": self.metadata,
            }
        )
        return d

    @classmethod
    def from_full_dict(cls, data: dict[str, Any]) -> Task:
        result_data = data.get("result")
        return cls(
            id=data["id"],
            objective=data["objective"],
            capability=data["capability"],
            dependencies=list(data.get("dependencies", [])),
            required_tools=list(data.get("required_tools", [])),
            expected_output=data.get("expected_output", ""),
            status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 2),
            result=AgentOutput.from_dict(result_data) if result_data is not None else None,
            error=data.get("error"),
            agent_id=data.get("agent_id"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}) or {},
        )


class TaskGraph:
    def __init__(self, tasks: list[Task] | None = None) -> None:
        self.tasks: dict[str, Task] = {}
        for task in tasks or []:
            self.add_task(task)
        if tasks:
            self._validate_acyclic()

    def add_task(self, task: Task) -> None:
        if task.id in self.tasks:
            raise ValueError(f"Duplicate task id: {task.id}")
        for dep in task.dependencies:
            if dep == task.id:
                raise ValueError(f"Task {task.id} cannot depend on itself")
        self.tasks[task.id] = task

    def remove_task(self, task_id: str) -> None:
        """Remove a task. Refuses to leave dangling dependencies - remove
        dependents first (or accept the ``force`` escape hatch)."""
        if task_id not in self.tasks:
            return
        dependents = self.get_dependents(task_id)
        if dependents:
            raise ValueError(
                f"Cannot remove task '{task_id}': still depended on by {[t.id for t in dependents]}"
            )
        del self.tasks[task_id]

    def add_dependency(self, task_id: str, dependency_id: str) -> None:
        """Add an edge ``task_id -> dependency_id``. Rejects unknown tasks,
        self-dependency, and anything that would introduce a cycle."""
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task '{task_id}'")
        if dependency_id not in self.tasks:
            raise ValueError(f"Unknown dependency task '{dependency_id}'")
        if task_id == dependency_id:
            raise ValueError(f"Task '{task_id}' cannot depend on itself")
        task = self.tasks[task_id]
        if dependency_id in task.dependencies:
            return
        task.dependencies.append(dependency_id)
        cycle = self.detect_cycles()
        if cycle:
            task.dependencies.remove(dependency_id)
            raise CycleError(f"Adding dependency '{task_id}' -> '{dependency_id}' would create a cycle: {' -> '.join(cycle)}")

    def get_task(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def get_dependencies(self, task_id: str) -> list[Task]:
        task = self.tasks[task_id]
        return [self.tasks[d] for d in task.dependencies if d in self.tasks]

    def get_dependents(self, task_id: str) -> list[Task]:
        """Tasks that directly depend on ``task_id``."""
        return [t for t in self.tasks.values() if task_id in t.dependencies]

    def finalize(self) -> None:
        """Validate the graph (unknown deps, cycles) once fully built, raising
        on the first problem found. Prefer ``validate()`` for a full,
        structured report instead of a single exception."""
        errors = self.validate()
        if errors:
            # Preserve the historical exception types callers already handle
            # (e.g. Planner catches CycleError specifically).
            if any("cycle" in e.lower() for e in errors):
                raise CycleError(errors[0])
            raise ValueError(errors[0])

    def validate(self, *, known_capabilities: set[str] | None = None, known_tools: set[str] | None = None) -> list[str]:
        """Non-raising structured validation: unique ids (guaranteed by
        ``add_task``, checked again defensively here), every dependency
        exists, no cycles, and - when the caller supplies what's actually
        registered - every task's capability has a matching agent and every
        required tool is known. Returns every problem found, not just the
        first."""
        errors: list[str] = []
        seen: set[str] = set()
        for task in self.tasks.values():
            if task.id in seen:
                errors.append(f"Duplicate task id: {task.id}")
            seen.add(task.id)
            for dep in task.dependencies:
                if dep not in self.tasks:
                    errors.append(f"Task '{task.id}' depends on unknown task '{dep}'")
            if known_capabilities is not None and task.capability not in known_capabilities:
                errors.append(f"Task '{task.id}' requires capability '{task.capability}' but no agent declares it")
            if known_tools is not None:
                missing = [t for t in task.required_tools if t not in known_tools]
                if missing:
                    errors.append(f"Task '{task.id}' requires unknown tool(s): {missing}")

        cycle = self.detect_cycles()
        if cycle:
            errors.append(f"Cycle detected: {' -> '.join(cycle)}")
        return errors

    def detect_cycles(self) -> list[str] | None:
        """Returns the cycle as a list of task ids (e.g. ``["A", "B", "C", "A"]``)
        if one exists, else ``None``. Never raises."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in self.tasks}

        def visit(tid: str, stack: list[str]) -> list[str] | None:
            color[tid] = GRAY
            for dep in self.tasks[tid].dependencies:
                if dep not in self.tasks:
                    continue
                if color[dep] == GRAY:
                    return stack + [dep]
                if color[dep] == WHITE:
                    found = visit(dep, stack + [dep])
                    if found:
                        return found
            color[tid] = BLACK
            return None

        for tid in self.tasks:
            if color[tid] == WHITE:
                found = visit(tid, [tid])
                if found:
                    return found
        return None

    def _validate_acyclic(self) -> None:
        cycle = self.detect_cycles()
        if cycle:
            raise CycleError(f"Cycle detected: {' -> '.join(cycle)}")

    def get_ready_tasks(self) -> list[Task]:
        """Tasks whose dependencies all succeeded and that haven't run."""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps = [self.tasks[d] for d in task.dependencies]
            if all(d.status == TaskStatus.SUCCEEDED for d in deps):
                ready.append(task)
        return ready

    def mark_unreachable_as_skipped(self) -> list[Task]:
        """Legacy full-graph-abort cascade: skip every pending task whose
        dependency chain contains a failed/skipped task. Used only when the
        whole execution is being aborted (not for isolated task failures -
        see ``mark_blocked_by_failed_dependencies`` for that)."""
        skipped = []
        changed = True
        while changed:
            changed = False
            for task in self.tasks.values():
                if task.status != TaskStatus.PENDING:
                    continue
                deps = [self.tasks[d] for d in task.dependencies]
                if any(d.status in (TaskStatus.FAILED, TaskStatus.SKIPPED) for d in deps):
                    task.status = TaskStatus.SKIPPED
                    skipped.append(task)
                    changed = True
        return skipped

    def mark_blocked_by_failed_dependencies(self) -> list[Task]:
        """Failure isolation: cascade BLOCKED to pending tasks whose
        dependency chain contains a permanently-failed (or already-blocked)
        task, WITHOUT touching independent branches - those keep running
        normally. This is what lets ``Task D`` (depends on healthy tasks)
        keep going while ``Task E`` (depends on a failed task) becomes
        BLOCKED instead of the whole execution aborting."""
        blocked = []
        changed = True
        while changed:
            changed = False
            for task in self.tasks.values():
                if task.status not in (TaskStatus.PENDING, TaskStatus.RETRYING):
                    continue
                deps = [self.tasks[d] for d in task.dependencies]
                if any(d.status in (TaskStatus.FAILED, TaskStatus.BLOCKED) for d in deps):
                    task.status = TaskStatus.BLOCKED
                    blocked.append(task)
                    changed = True
        return blocked

    def mark_remaining_as_cancelled(self) -> list[Task]:
        """Used by execution cancellation: every task not already in a
        terminal state is marked CANCELLED (never marked as completed)."""
        cancelled = []
        for task in self.tasks.values():
            if task.status not in TERMINAL_TASK_STATUSES:
                task.status = TaskStatus.CANCELLED
                cancelled.append(task)
        return cancelled

    def is_complete(self) -> bool:
        return all(t.status in TERMINAL_TASK_STATUSES for t in self.tasks.values())

    def has_pending_work(self) -> bool:
        return any(
            t.status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.RETRYING, TaskStatus.WAITING)
            for t in self.tasks.values()
        )

    def succeeded_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.SUCCEEDED]

    def failed_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]

    def blocked_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.BLOCKED]

    def cancelled_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.CANCELLED]

    def running_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]

    def to_dict(self) -> dict[str, Any]:
        """Full round-trippable snapshot of every task, in insertion order."""
        return {"tasks": [t.to_full_dict() for t in self.tasks.values()]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskGraph:
        graph = cls()
        for raw in data.get("tasks", []):
            graph.tasks[raw["id"]] = Task.from_full_dict(raw)
        return graph

    def reset_interrupted_tasks(self) -> list[Task]:
        """After a crash/resume, a task left RUNNING/READY/RETRYING never
        actually finished - reset it to PENDING so the scheduler picks it
        up again. Tasks that reached a terminal status (SUCCEEDED/FAILED/
        BLOCKED/CANCELLED/SKIPPED) before the crash are left untouched, so
        completed work is never re-run."""
        reset = []
        for task in self.tasks.values():
            if task.status in (TaskStatus.RUNNING, TaskStatus.READY, TaskStatus.RETRYING):
                task.status = TaskStatus.PENDING
                reset.append(task)
        return reset

    def topological_order(self) -> list[Task]:
        order: list[Task] = []
        visited: set[str] = set()

        def visit(tid: str) -> None:
            if tid in visited:
                return
            visited.add(tid)
            for dep in self.tasks[tid].dependencies:
                visit(dep)
            order.append(self.tasks[tid])

        for tid in self.tasks:
            visit(tid)
        return order
