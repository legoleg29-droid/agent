"""Structured execution context.

Separates global (goal-level) context from per-task, per-agent, and tool
result context so agents receive only what's relevant to their task
instead of the full conversation history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orchestrator.agents.base import AgentOutput
from orchestrator.core.task_graph import Task, TaskGraph


@dataclass
class ExecutionMetadata:
    goal: str
    started_at: float
    plan_version: int = 1
    replans: int = 0


class ContextManager:
    """Owns global context and per-task outputs; builds narrow agent inputs."""

    def __init__(self, goal: str, started_at: float) -> None:
        self.global_context: dict[str, Any] = {}
        self.task_outputs: dict[str, AgentOutput] = {}
        self.metadata = ExecutionMetadata(goal=goal, started_at=started_at)

    def record_task_output(self, task_id: str, output: AgentOutput) -> None:
        self.task_outputs[task_id] = output

    def upstream_outputs_for(self, task: Task, graph: TaskGraph) -> dict[str, str]:
        """Only the direct dependency outputs, not the entire run history."""
        outputs: dict[str, str] = {}
        for dep_id in task.dependencies:
            dep_output = self.task_outputs.get(dep_id)
            if dep_output is not None:
                outputs[dep_id] = dep_output.content
        return outputs

    def completed_summary(self, graph: TaskGraph) -> dict[str, str]:
        return {
            t.id: (self.task_outputs[t.id].content if t.id in self.task_outputs else "")
            for t in graph.succeeded_tasks()
        }

    def set_global(self, key: str, value: Any) -> None:
        self.global_context[key] = value
