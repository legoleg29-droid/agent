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
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


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

    def finalize(self) -> None:
        """Validate the graph (unknown deps, cycles) once fully built."""
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    raise ValueError(f"Task {task.id} depends on unknown task '{dep}'")
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in self.tasks}

        def visit(tid: str, stack: list[str]) -> None:
            color[tid] = GRAY
            for dep in self.tasks[tid].dependencies:
                if dep not in self.tasks:
                    continue
                if color[dep] == GRAY:
                    raise CycleError(f"Cycle detected: {' -> '.join(stack + [dep])}")
                if color[dep] == WHITE:
                    visit(dep, stack + [dep])
            color[tid] = BLACK

        for tid in self.tasks:
            if color[tid] == WHITE:
                visit(tid, [tid])

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
        """Skip pending tasks whose dependency chain contains a failed task."""
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

    def is_complete(self) -> bool:
        return all(
            t.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.SKIPPED)
            for t in self.tasks.values()
        )

    def has_pending_work(self) -> bool:
        return any(t.status in (TaskStatus.PENDING, TaskStatus.RUNNING) for t in self.tasks.values())

    def succeeded_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.SUCCEEDED]

    def failed_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]

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
        """After a crash/resume, a task left RUNNING never actually finished -
        reset it to PENDING so the scheduler picks it up again. Tasks that
        reached SUCCEEDED/FAILED/SKIPPED before the crash are left untouched,
        so completed work is never re-run."""
        reset = []
        for task in self.tasks.values():
            if task.status == TaskStatus.RUNNING:
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
