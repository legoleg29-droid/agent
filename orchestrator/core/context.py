"""Structured execution context.

Separates global (goal-level) context from per-task, per-agent, and tool
result context so agents receive only what's relevant to their task
instead of the full conversation history or the entire memory database.

``ContextManager`` owns the live, in-process execution context (task
outputs, tool results); ``build_agent_context`` is where that gets
combined with persisted ``ExecutionState`` and retrieved ``MemoryEntry``
items into a single, budgeted context for one agent call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orchestrator.agents.base import AgentOutput
from orchestrator.core.context_budget import BudgetedContext, ContextBudget, ContextSection
from orchestrator.core.task_graph import Task, TaskGraph

if TYPE_CHECKING:
    from orchestrator.agents.base import BaseAgent
    from orchestrator.memory.manager import MemoryManager
    from orchestrator.state.models import ExecutionState

DEFAULT_MAX_CONTEXT_TOKENS = 3000
_RECENT_TOOL_RESULTS_PER_TASK = 5
_MAX_RETRIEVED_MEMORIES = 8
_MIN_MEMORY_IMPORTANCE = 0.3


@dataclass
class ExecutionMetadata:
    goal: str
    started_at: float
    plan_version: int = 1
    replans: int = 0


@dataclass
class AgentContext:
    """What actually gets handed to one agent call, after budgeting.

    Composed from: (1) system instructions, (2) the current task, (3)
    relevant execution state, (4) relevant previous task outputs, (5)
    relevant tool results, (6) retrieved memory, (7) current constraints -
    never the entire memory database or full conversation history.
    """

    system_instructions: str
    dependency_outputs: dict[str, str]
    constraints: list[str]
    memory_snippets: list[dict[str, Any]]
    budget: BudgetedContext

    @property
    def rendered(self) -> str:
        return self.budget.render()


class ContextManager:
    """Owns global context and per-task outputs; builds narrow agent inputs."""

    def __init__(self, goal: str, started_at: float) -> None:
        self.global_context: dict[str, Any] = {}
        self.task_outputs: dict[str, AgentOutput] = {}
        # Structured tool-call history, e.g.
        # {"tool": "calculator", "status": "success", "result": ..., "task_id": ..., "timestamp": ...}
        # Never raw/unstructured tool output dumped into global_context.
        self.tool_results: list[dict[str, Any]] = []
        self.metadata = ExecutionMetadata(goal=goal, started_at=started_at)

    def record_task_output(self, task_id: str, output: AgentOutput) -> None:
        self.task_outputs[task_id] = output

    def record_tool_result(
        self,
        *,
        tool: str,
        status: str,
        task_id: str | None,
        timestamp: float,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {"tool": tool, "status": status, "task_id": task_id, "timestamp": timestamp}
        if status == "success":
            entry["result"] = result
        else:
            entry["error"] = error
        self.tool_results.append(entry)

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

    # -- Phase 3: budgeted agent context ---------------------------------

    def build_agent_context(
        self,
        *,
        task: Task,
        graph: TaskGraph,
        agent: "BaseAgent",
        execution_state: "ExecutionState | None" = None,
        memory_manager: "MemoryManager | None" = None,
        constraints: list[str] | None = None,
        max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ) -> AgentContext:
        """Compose and budget everything one agent call needs:

        1. system instructions (always, not budgeted - it's fixed per agent)
        2. current task (required, never dropped)
        3. relevant execution state
        4. relevant previous (dependency) task outputs
        5. relevant tool results
        6. retrieved memory
        7. current constraints (required - a constraint is never silently dropped)
        """
        sections: list[ContextSection] = [
            ContextSection(
                name="task",
                text=f"Objective: {task.objective}" + (f"\nExpected output: {task.expected_output}" if task.expected_output else ""),
                priority=0,
                required=True,
            )
        ]

        dependency_outputs = self.upstream_outputs_for(task, graph)
        for dep_id, content in dependency_outputs.items():
            sections.append(ContextSection(name=f"dependency:{dep_id}", text=f"[{dep_id}]: {content}", priority=1))

        if constraints:
            sections.append(
                ContextSection(
                    name="constraints",
                    text="Constraints:\n" + "\n".join(f"- {c}" for c in constraints),
                    priority=1,
                    required=True,
                )
            )

        relevant_tools = [r for r in self.tool_results if r.get("task_id") == task.id]
        if relevant_tools:
            tool_text = "\n".join(
                f"- {r['tool']}: {r.get('result') if r['status'] == 'success' else 'ERROR: ' + str(r.get('error'))}"
                for r in relevant_tools[-_RECENT_TOOL_RESULTS_PER_TASK:]
            )
            sections.append(ContextSection(name="tool_results", text="Recent tool results:\n" + tool_text, priority=2))

        retrieved_memories = []
        if memory_manager is not None:
            retrieved_memories = self._retrieve_relevant_memory(memory_manager)
            if retrieved_memories:
                mem_lines = []
                for m in retrieved_memories:
                    content = m.content if isinstance(m.content, str) else json.dumps(m.content, default=str)
                    mem_lines.append(f"- ({m.type.value}, importance={m.importance:.1f}) {content}")
                priority = 3 if any(m.importance >= 0.6 for m in retrieved_memories) else 4
                sections.append(ContextSection(name="memory", text="Retrieved memory:\n" + "\n".join(mem_lines), priority=priority))

        if execution_state is not None:
            progress = (
                f"Execution goal: {execution_state.user_goal}\n"
                f"Completed tasks: {len(execution_state.completed_tasks)}, "
                f"Failed tasks: {len(execution_state.failed_tasks)}"
            )
            sections.append(ContextSection(name="execution_state", text=progress, priority=5))

        budget = ContextBudget(max_tokens=max_tokens).assemble(sections)

        final_dependency_outputs = {
            dep_id: budget.sections[f"dependency:{dep_id}"].split(f"[{dep_id}]: ", 1)[-1]
            for dep_id in dependency_outputs
            if f"dependency:{dep_id}" in budget.sections
        }

        return AgentContext(
            system_instructions=agent.system_instructions,
            dependency_outputs=final_dependency_outputs,
            constraints=constraints or [],
            memory_snippets=[m.to_dict() for m in retrieved_memories],
            budget=budget,
        )

    def _retrieve_relevant_memory(self, memory_manager: "MemoryManager") -> list[Any]:
        from orchestrator.memory.models import MemoryQuery

        query = MemoryQuery(
            execution_id=memory_manager.execution_id,
            min_importance=_MIN_MEMORY_IMPORTANCE,
            limit=_MAX_RETRIEVED_MEMORIES,
        )
        try:
            return memory_manager.search(query)
        except Exception:  # noqa: BLE001 - memory retrieval must never break execution
            return []
