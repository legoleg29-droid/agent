"""Runtime state schemas.

Distinguishes execution/task/agent/tool state instead of one generic
"memory" blob:

    Runtime State
    ├── Execution State   - one run's overall status and shape
    ├── Task State        - persistent, per-task record (survives retries)
    ├── Agent State       - transient, per-(execution, task, agent) scratch state
    ├── Tool State        - last-known status of a tool for a given call site
    └── Context           - owned by ContextManager (orchestrator/core/context.py)

All of these are plain, JSON-serializable dataclasses so the orchestrator
can reconstruct its current execution state from a checkpoint without
depending on conversation history.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
)


@dataclass
class TaskState:
    """Persistent, per-task record - survives individual agent calls and
    process restarts. Mirrors (and is kept in sync with) a live
    ``orchestrator.core.task_graph.Task``, but is what actually gets
    checkpointed and reasoned about as "task state"."""

    task_id: str
    status: str
    attempt: int = 1
    agent_id: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "attempt": self.attempt,
            "agent_id": self.agent_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskState:
        return cls(**data)

    @classmethod
    def from_task(cls, task: Any) -> TaskState:
        """Build a TaskState snapshot from a live ``task_graph.Task``."""
        return cls(
            task_id=task.id,
            status=task.status.value,
            attempt=task.attempt,
            agent_id=task.agent_id,
            started_at=task.started_at,
            completed_at=task.completed_at,
            result=task.result.to_dict() if task.result is not None else None,
            error=task.error,
        )


@dataclass
class AgentState:
    """Transient execution state an agent may hold mid-task: current
    objective, reasoning context, an in-flight tool call, an intermediate
    result. Always scoped to one (execution_id, task_id, agent_id) tuple -
    never a module-level/global dict - so it can never leak between
    concurrent executions, tasks, or users. Discarded once the task
    finishes; not itself checkpointed (Task/ExecutionState already capture
    what needs to survive a restart)."""

    execution_id: str
    task_id: str
    agent_id: str
    current_objective: str | None = None
    reasoning_context: dict[str, Any] = field(default_factory=dict)
    active_tool_call: str | None = None
    intermediate_result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()


@dataclass
class ToolState:
    """Last-known status of a tool at a given call site (task/agent)."""

    tool_id: str
    task_id: str | None = None
    agent_id: str | None = None
    last_status: str | None = None
    last_called_at: float | None = None
    call_count: int = 0

    def record_call(self, status: str) -> None:
        self.last_status = status
        self.last_called_at = time.time()
        self.call_count += 1


@dataclass
class Artifact:
    """A generated file or other durable output. Only a reference (path/id)
    is tracked here - large content never lives inside memory or state."""

    artifact_id: str
    type: str
    path: str
    task_id: str | None = None
    agent_id: str | None = None
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "type": self.type,
            "path": self.path,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact:
        return cls(**data)


@dataclass
class ExecutionState:
    """The top-level, persisted record of one orchestrator run. Enough to
    reconstruct where a run is without replaying conversation history."""

    execution_id: str
    user_goal: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    current_plan: dict[str, Any] | None = None  # TaskGraph.to_dict() snapshot
    active_task: str | None = None
    completed_tasks: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    task_states: dict[str, TaskState] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)  # small, execution-scoped global context
    artifacts: list[str] = field(default_factory=list)  # artifact ids
    replans_used: int = 0
    max_replans: int = 2
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Phase 5: evaluation/self-repair/replanning history - append-only,
    # never overwritten, so a run's full evaluation/repair/replan story is
    # reconstructable from the checkpointed state alone.
    evaluation_history: list[dict[str, Any]] = field(default_factory=list)
    repair_history: list[dict[str, Any]] = field(default_factory=list)
    replan_history: list[dict[str, Any]] = field(default_factory=list)
    plan_version: int = 1
    plan_versions: list[dict[str, Any]] = field(default_factory=list)
    failure_signatures: list[str] = field(default_factory=list)
    execution_metrics: dict[str, Any] = field(default_factory=dict)

    def record_evaluation(self, task_id: str, evaluation: dict[str, Any]) -> None:
        self.evaluation_history.append({"task_id": task_id, "timestamp": time.time(), **evaluation})
        self.touch()

    def record_repair(self, task_id: str, *, attempt: int, outcome: str, details: dict[str, Any] | None = None) -> None:
        self.repair_history.append(
            {"task_id": task_id, "attempt": attempt, "outcome": outcome, "timestamp": time.time(), **(details or {})}
        )
        self.touch()

    def record_replan(self, *, failed_task_id: str, reason: str, plan_version: dict[str, Any]) -> None:
        self.replan_history.append(
            {"failed_task_id": failed_task_id, "reason": reason, "timestamp": time.time(), "plan_version": plan_version}
        )
        self.touch()

    def record_plan_version(self, plan_version: dict[str, Any]) -> None:
        self.plan_versions.append(plan_version)
        self.plan_version = plan_version.get("version", self.plan_version)
        self.touch()

    def record_failure_signature(self, signature: str) -> None:
        self.failure_signatures.append(signature)
        self.touch()

    @classmethod
    def create(cls, user_goal: str, *, execution_id: str | None = None, max_replans: int = 2) -> ExecutionState:
        now = time.time()
        return cls(
            execution_id=execution_id or f"exec_{uuid.uuid4().hex[:12]}",
            user_goal=user_goal,
            status=ExecutionStatus.PENDING,
            max_replans=max_replans,
            created_at=now,
            updated_at=now,
        )

    def touch(self) -> None:
        self.updated_at = time.time()

    def set_status(self, status: ExecutionStatus) -> None:
        self.status = status
        self.touch()

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def sync_task_state(self, task: Any) -> TaskState:
        """Update this execution's task_states/completed_tasks/failed_tasks
        from a live Task, keeping the persisted view in sync at each
        transition."""
        from orchestrator.core.task_graph import TaskStatus  # local import avoids a cycle at module load

        snapshot = TaskState.from_task(task)
        self.task_states[task.id] = snapshot

        if task.status == TaskStatus.SUCCEEDED and task.id not in self.completed_tasks:
            self.completed_tasks.append(task.id)
        if task.status == TaskStatus.FAILED and task.id not in self.failed_tasks:
            self.failed_tasks.append(task.id)
        if task.status == TaskStatus.RUNNING:
            self.active_task = task.id
        elif self.active_task == task.id:
            self.active_task = None

        self.touch()
        return snapshot

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact.artifact_id)
        self.metadata.setdefault("artifacts_detail", []).append(artifact.to_dict())
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "user_goal": self.user_goal,
            "status": self.status.value,
            "current_plan": self.current_plan,
            "active_task": self.active_task,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "task_states": {tid: ts.to_dict() for tid, ts in self.task_states.items()},
            "context": self.context,
            "artifacts": self.artifacts,
            "replans_used": self.replans_used,
            "max_replans": self.max_replans,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "evaluation_history": self.evaluation_history,
            "repair_history": self.repair_history,
            "replan_history": self.replan_history,
            "plan_version": self.plan_version,
            "plan_versions": self.plan_versions,
            "failure_signatures": self.failure_signatures,
            "execution_metrics": self.execution_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionState:
        return cls(
            execution_id=data["execution_id"],
            user_goal=data["user_goal"],
            status=ExecutionStatus(data.get("status", ExecutionStatus.PENDING.value)),
            current_plan=data.get("current_plan"),
            active_task=data.get("active_task"),
            completed_tasks=list(data.get("completed_tasks", [])),
            failed_tasks=list(data.get("failed_tasks", [])),
            task_states={
                tid: TaskState.from_dict(ts) for tid, ts in (data.get("task_states") or {}).items()
            },
            context=data.get("context", {}) or {},
            artifacts=list(data.get("artifacts", [])),
            replans_used=data.get("replans_used", 0),
            max_replans=data.get("max_replans", 2),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}) or {},
            evaluation_history=list(data.get("evaluation_history", []) or []),
            repair_history=list(data.get("repair_history", []) or []),
            replan_history=list(data.get("replan_history", []) or []),
            plan_version=data.get("plan_version", 1),
            plan_versions=list(data.get("plan_versions", []) or []),
            failure_signatures=list(data.get("failure_signatures", []) or []),
            execution_metrics=data.get("execution_metrics", {}) or {},
        )
