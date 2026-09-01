"""Planner: turns a high-level goal into a Task DAG, and turns execution
feedback into a minimal plan patch (not a wholesale new plan).

Uses the LLM provider to decompose the goal, informed by the *actual*
registered agent capabilities and tools (never hardcoded), and parses the
result into a validated ``TaskGraph``.
"""

from __future__ import annotations

import json
import re

from orchestrator.core.evaluation_models import EvaluationResult
from orchestrator.core.logging_utils import EventLog
from orchestrator.core.plan_patch import PatchOpType, PlanPatchOp
from orchestrator.core.task_graph import CycleError, Task, TaskGraph
from orchestrator.providers.base import LLMMessage, LLMProvider

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class PlanParseError(ValueError):
    pass


_PLANNER_SYSTEM = """You are the planning module of an AI agent orchestrator.
Given a user goal, decompose it into a minimal set of discrete tasks that
together achieve the goal. You may ONLY assign each task a "capability"
from the provided list of available capabilities - never invent one.
Respect true dependencies (a task that needs another task's output must
list it in "dependencies"); do not add dependencies that aren't needed,
since independent tasks run in parallel.

Where it's genuinely useful, give a task explicit "acceptance_criteria" -
a list of either plain strings (checked by a human/LLM reviewer) or
structured checks the system can verify automatically, e.g.
{"type": "file_exists", "path": "index.html"},
{"type": "contains", "text": "..."}, {"type": "min_length", "length": 200},
{"type": "tool_succeeded", "tool": "run_python_tests"}. Leave it empty ([])
when a task has no meaningful way to check completion beyond "it ran".

Respond with ONLY a JSON object of this exact shape, no prose, no markdown
fences:
{
  "tasks": [
    {
      "id": "short_snake_case_id",
      "objective": "specific, actionable description of what this task must produce",
      "capability": "one of the available capabilities",
      "dependencies": ["id", "..."],
      "required_tools": ["tool_name", "..."],
      "expected_output": "what a correct result looks like",
      "acceptance_criteria": []
    }
  ]
}
"""

_REPLAN_SYSTEM = """You are the replanning module of an AI agent orchestrator.
A task in the current plan failed (or produced a result that couldn't be
repaired) and execution cannot continue as originally planned. Propose the
SMALLEST set of patch operations that lets execution continue - do not
regenerate the whole plan, and never touch tasks that already succeeded.

Prefer REPLACE_TASK when the failed task needs a fundamentally different
approach: it adds your new task and automatically rewires every task that
depended on the old one onto the new one - you don't need to manually fix
up dependents' dependency lists.

Available operations (a "task" object has: id, objective, capability,
dependencies, required_tools, expected_output, acceptance_criteria):
- {"op": "add_task", "task": {...}}
- {"op": "remove_task", "task_id": "..."}
- {"op": "replace_task", "task_id": "<id of the task to replace>", "task": {...new task...}}
- {"op": "modify_task", "task_id": "...", "changes": {"objective": "...", "acceptance_criteria": [...]}}
- {"op": "add_dependency", "task_id": "...", "dependency_id": "..."}
- {"op": "remove_dependency", "task_id": "...", "dependency_id": "..."}

Respond with ONLY a JSON object, no prose, no markdown fences:
{
  "reason": "one sentence: what changed and why",
  "operations": [ ... ]
}
"""


class Planner:
    def __init__(self, provider: LLMProvider, event_log: EventLog | None = None) -> None:
        self.provider = provider
        self.event_log = event_log

    async def plan(
        self,
        goal: str,
        *,
        capabilities: list[str],
        tools: list[str],
    ) -> TaskGraph:
        prompt = self._build_prompt(goal, capabilities, tools)
        response = await self.provider.complete(system=_PLANNER_SYSTEM, messages=[LLMMessage("user", prompt)])
        graph = self._parse_plan(response.text, capabilities, tools)
        if self.event_log:
            self.event_log.emit(
                "PLANNER",
                f"Generated plan with {len(graph.tasks)} task(s) for goal: {goal!r}",
                model=response.model,
                tokens_used=response.total_tokens,
                extra={"task_ids": list(graph.tasks.keys())},
            )
        return graph

    async def replan(
        self,
        goal: str,
        *,
        graph: TaskGraph,
        capabilities: list[str],
        tools: list[str],
        failed_task: Task,
        failure_reason: str,
        evaluation: EvaluationResult | None = None,
        repair_attempts: int = 0,
        memory_snippets: list[dict] | None = None,
    ) -> tuple[list[PlanPatchOp], str]:
        """Produce a minimal plan PATCH (not a wholesale new plan) for the
        unfinished portion of the plan, given full execution feedback:
        completed/failed/blocked tasks, the evaluator's own findings, and
        how many repair attempts already failed. Returns
        ``(operations, reason)`` - the orchestrator applies the ops to the
        live graph and records ``reason`` on the new plan version.
        """
        prompt = self._build_replan_prompt(
            goal,
            graph=graph,
            failed_task=failed_task,
            failure_reason=failure_reason,
            evaluation=evaluation,
            repair_attempts=repair_attempts,
            memory_snippets=memory_snippets,
        )
        response = await self.provider.complete(system=_REPLAN_SYSTEM, messages=[LLMMessage("user", prompt)])
        ops, reason = self._parse_patch(response.text, capabilities, tools)

        if self.event_log:
            self.event_log.emit(
                "REPLAN",
                f"Generated {len(ops)} patch operation(s). Reason: {reason or failure_reason}",
                model=response.model,
                tokens_used=response.total_tokens,
                extra={"operations": [op.to_dict() for op in ops]},
            )
        return ops, (reason or failure_reason)

    def _build_prompt(self, goal: str, capabilities: list[str], tools: list[str]) -> str:
        return (
            f"User goal: {goal}\n\n"
            f"Available capabilities: {', '.join(capabilities)}\n"
            f"Available tools: {', '.join(tools) if tools else '(none)'}"
        )

    def _build_replan_prompt(
        self,
        goal: str,
        *,
        graph: TaskGraph,
        failed_task: Task,
        failure_reason: str,
        evaluation: EvaluationResult | None,
        repair_attempts: int,
        memory_snippets: list[dict] | None,
    ) -> str:
        completed = graph.succeeded_tasks()
        failed = graph.failed_tasks()
        blocked = graph.blocked_tasks()

        parts = [
            f"Original goal: {goal}",
            "Completed tasks (do NOT recreate or depend around these - depend ON them by id if needed):\n"
            + "\n".join(f"- [{t.id}] {t.objective}" for t in completed) or "(none yet)",
            f"Failed task: [{failed_task.id}] {failed_task.objective}\nFailure reason: {failure_reason}",
        ]
        if evaluation is not None:
            parts.append(
                "Evaluator findings:\n"
                + "\n".join(f"- {r}" for r in evaluation.failed_criteria + evaluation.reasons)
            )
        if repair_attempts:
            parts.append(f"Repair was already attempted {repair_attempts} time(s) on this task and did not succeed.")
        if failed:
            parts.append("All currently failed tasks: " + ", ".join(t.id for t in failed))
        if blocked:
            parts.append(
                "Currently blocked tasks (depend on a failed task - a REPLACE_TASK on the "
                "failed task will automatically unblock these onto the replacement): "
                + ", ".join(t.id for t in blocked)
            )
        if memory_snippets:
            parts.append(
                "Relevant memory:\n" + "\n".join(f"- {m.get('content')}" for m in memory_snippets)
            )
        return "\n\n".join(parts)

    def _parse_patch(
        self, text: str, capabilities: list[str], tools: list[str]
    ) -> tuple[list[PlanPatchOp], str]:
        text = text.strip()
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            raise PlanParseError(f"Replan response did not contain a JSON object: {text[:200]!r}")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Replan response was not valid JSON: {exc}") from exc

        raw_ops = payload.get("operations")
        if not isinstance(raw_ops, list):
            raise PlanParseError("Replan JSON must contain an 'operations' list")

        ops: list[PlanPatchOp] = []
        for raw in raw_ops:
            if "op" not in raw:
                raise PlanParseError(f"Patch operation missing 'op': {raw}")
            try:
                patch_op = PlanPatchOp.from_dict(raw)
            except ValueError as exc:
                raise PlanParseError(f"Unknown patch operation: {raw}") from exc

            task_payload = patch_op.task
            if task_payload is not None:
                capability = task_payload.get("capability")
                if capability is not None and capability not in capabilities:
                    raise PlanParseError(f"Patch task uses unknown capability '{capability}'. Available: {capabilities}")
                unknown_tools = [t for t in task_payload.get("required_tools", []) or [] if t not in tools]
                if unknown_tools:
                    raise PlanParseError(f"Patch task references unknown tools: {unknown_tools}")
                for key in ("id", "objective", "capability"):
                    if patch_op.op in (PatchOpType.ADD_TASK, PatchOpType.REPLACE_TASK) and key not in task_payload:
                        raise PlanParseError(f"Patch task missing required field '{key}': {task_payload}")
            ops.append(patch_op)

        return ops, str(payload.get("reason", ""))

    def _parse_plan(
        self,
        text: str,
        capabilities: list[str],
        tools: list[str],
        known_external_ids: set[str] | None = None,
    ) -> TaskGraph:
        text = text.strip()
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            raise PlanParseError(f"Planner response did not contain a JSON object: {text[:200]!r}")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Planner response was not valid JSON: {exc}") from exc

        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise PlanParseError("Plan JSON must contain a non-empty 'tasks' list")

        known_external_ids = known_external_ids or set()
        graph = TaskGraph()
        for raw in raw_tasks:
            for key in ("id", "objective", "capability"):
                if key not in raw:
                    raise PlanParseError(f"Task missing required field '{key}': {raw}")
            capability = raw["capability"]
            if capability not in capabilities:
                raise PlanParseError(
                    f"Task '{raw['id']}' uses unknown capability '{capability}'. "
                    f"Available: {capabilities}"
                )
            required_tools = raw.get("required_tools", []) or []
            unknown_tools = [t for t in required_tools if t not in tools]
            if unknown_tools:
                raise PlanParseError(f"Task '{raw['id']}' references unknown tools: {unknown_tools}")

            graph.add_task(
                Task(
                    id=raw["id"],
                    objective=raw["objective"],
                    capability=capability,
                    dependencies=raw.get("dependencies", []) or [],
                    required_tools=required_tools,
                    expected_output=raw.get("expected_output", ""),
                    acceptance_criteria=raw.get("acceptance_criteria", []) or [],
                )
            )

        try:
            for task in graph.tasks.values():
                for dep in task.dependencies:
                    if dep not in graph.tasks and dep not in known_external_ids:
                        raise PlanParseError(
                            f"Task '{task.id}' depends on unknown task '{dep}'"
                        )
            graph._validate_acyclic()
        except CycleError as exc:
            raise PlanParseError(str(exc)) from exc
        return graph
