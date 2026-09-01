"""Plan patch operations.

Replanning should modify the graph minimally, not regenerate it wholesale.
A ``PlanPatchOp`` is one small, explicit edit; ``apply_plan_patch`` applies
a list of them to a live ``TaskGraph``. ``REPLACE_TASK`` is the operation
that matters most for recovery: it adds the replacement task and rewires
every dependent that pointed at the old (failed) task onto the new one -
so ``Task C`` that depended on failed ``Task B`` automatically depends on
``B2`` instead, and (if it had been marked BLOCKED) is reset to PENDING so
it can run again. Completed work is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from orchestrator.core.task_graph import Task, TaskGraph, TaskStatus


class PatchOpType(str, Enum):
    ADD_TASK = "add_task"
    REMOVE_TASK = "remove_task"
    REPLACE_TASK = "replace_task"
    MODIFY_TASK = "modify_task"
    ADD_DEPENDENCY = "add_dependency"
    REMOVE_DEPENDENCY = "remove_dependency"


# Fields a MODIFY_TASK patch is allowed to touch - never status/result/
# timestamps/retry-tracking, which are execution-owned, not plan-owned.
_MODIFIABLE_FIELDS = frozenset(
    {"objective", "expected_output", "required_tools", "acceptance_criteria", "max_retries", "max_repairs"}
)


@dataclass
class PlanPatchOp:
    op: PatchOpType
    task_id: str | None = None          # target for REMOVE/MODIFY/REPLACE/dependency ops
    task: dict[str, Any] | None = None  # new task fields for ADD_TASK/REPLACE_TASK
    dependency_id: str | None = None    # for ADD_DEPENDENCY/REMOVE_DEPENDENCY
    changes: dict[str, Any] | None = None  # for MODIFY_TASK
    reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanPatchOp:
        return cls(
            op=PatchOpType(data["op"]),
            task_id=data.get("task_id"),
            task=data.get("task"),
            dependency_id=data.get("dependency_id"),
            changes=data.get("changes"),
            reason=data.get("reason", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op.value,
            "task_id": self.task_id,
            "task": self.task,
            "dependency_id": self.dependency_id,
            "changes": self.changes,
            "reason": self.reason,
        }


def _unique_id(graph: TaskGraph, proposed_id: str) -> str:
    if proposed_id not in graph.tasks:
        return proposed_id
    suffix = 2
    while f"{proposed_id}_{suffix}" in graph.tasks:
        suffix += 1
    return f"{proposed_id}_{suffix}"


def _task_from_dict(graph: TaskGraph, data: dict[str, Any]) -> Task:
    task_id = _unique_id(graph, data["id"])
    return Task(
        id=task_id,
        objective=data["objective"],
        capability=data["capability"],
        dependencies=list(data.get("dependencies", []) or []),
        required_tools=list(data.get("required_tools", []) or []),
        expected_output=data.get("expected_output", ""),
        acceptance_criteria=list(data.get("acceptance_criteria", []) or []),
    )


def apply_plan_patch(graph: TaskGraph, ops: list[PlanPatchOp]) -> list[str]:
    """Applies each op in order; returns a list of error strings for any
    op that couldn't be applied (that op is skipped, the rest still run).
    Callers should treat a non-empty return as "replan only partially
    succeeded" and decide whether that's acceptable."""
    errors: list[str] = []

    for patch_op in ops:
        try:
            _apply_one(graph, patch_op)
        except Exception as exc:  # noqa: BLE001 - never let one bad op crash the whole replan
            errors.append(f"{patch_op.op.value} on '{patch_op.task_id}' failed: {exc}")

    return errors


def _apply_one(graph: TaskGraph, patch_op: PlanPatchOp) -> None:
    if patch_op.op == PatchOpType.ADD_TASK:
        if not patch_op.task:
            raise ValueError("ADD_TASK requires 'task'")
        graph.add_task(_task_from_dict(graph, patch_op.task))
        return

    if patch_op.op == PatchOpType.REMOVE_TASK:
        if not patch_op.task_id:
            raise ValueError("REMOVE_TASK requires 'task_id'")
        graph.remove_task(patch_op.task_id)
        return

    if patch_op.op == PatchOpType.REPLACE_TASK:
        if not patch_op.task_id or not patch_op.task:
            raise ValueError("REPLACE_TASK requires 'task_id' and 'task'")
        old_id = patch_op.task_id
        if old_id not in graph.tasks:
            raise ValueError(f"unknown task '{old_id}'")
        new_task = _task_from_dict(graph, patch_op.task)
        graph.add_task(new_task)
        _rewire_dependents(graph, old_id, new_task.id)
        # The old task is superseded, not "still failing" - see
        # TaskGraph.failed_tasks(), which excludes it on this basis so a
        # successful replan reads as a completed execution, not a failed one.
        graph.tasks[old_id].metadata["superseded_by"] = new_task.id
        return

    if patch_op.op == PatchOpType.MODIFY_TASK:
        if not patch_op.task_id or not patch_op.changes:
            raise ValueError("MODIFY_TASK requires 'task_id' and 'changes'")
        task = graph.tasks.get(patch_op.task_id)
        if task is None:
            raise ValueError(f"unknown task '{patch_op.task_id}'")
        if task.status not in (TaskStatus.PENDING, TaskStatus.BLOCKED, TaskStatus.READY):
            raise ValueError(f"cannot modify task '{task.id}' in status {task.status.value}")
        for field_name, value in patch_op.changes.items():
            if field_name not in _MODIFIABLE_FIELDS:
                raise ValueError(f"field '{field_name}' is not plan-modifiable")
            setattr(task, field_name, value)
        if task.status == TaskStatus.BLOCKED:
            task.status = TaskStatus.PENDING
        return

    if patch_op.op == PatchOpType.ADD_DEPENDENCY:
        if not patch_op.task_id or not patch_op.dependency_id:
            raise ValueError("ADD_DEPENDENCY requires 'task_id' and 'dependency_id'")
        graph.add_dependency(patch_op.task_id, patch_op.dependency_id)
        return

    if patch_op.op == PatchOpType.REMOVE_DEPENDENCY:
        if not patch_op.task_id or not patch_op.dependency_id:
            raise ValueError("REMOVE_DEPENDENCY requires 'task_id' and 'dependency_id'")
        graph.remove_dependency(patch_op.task_id, patch_op.dependency_id)
        return

    raise ValueError(f"unhandled patch op: {patch_op.op}")


def _rewire_dependents(graph: TaskGraph, old_id: str, new_id: str) -> None:
    """Every task that depended on ``old_id`` now depends on ``new_id``
    instead. A dependent that had been BLOCKED specifically because
    ``old_id`` failed is reset to PENDING so it gets a real chance to run
    against the replacement."""
    for task in graph.tasks.values():
        if task.id == new_id:
            continue
        if old_id in task.dependencies:
            task.dependencies = [new_id if d == old_id else d for d in task.dependencies]
            if task.status == TaskStatus.BLOCKED:
                task.status = TaskStatus.PENDING
