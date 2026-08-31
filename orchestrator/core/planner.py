"""Planner: turns a high-level goal into a Task DAG.

Uses the LLM provider to decompose the goal, informed by the *actual*
registered agent capabilities and tools (never hardcoded), and parses the
result into a validated ``TaskGraph``.
"""

from __future__ import annotations

import json
import re
import uuid

from orchestrator.core.logging_utils import EventLog
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
      "expected_output": "what a correct result looks like"
    }
  ]
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
        capabilities: list[str],
        tools: list[str],
        completed_summary: dict[str, str],
        failure_reason: str,
    ) -> list[Task]:
        """Produce replacement tasks for the unfinished portion of the plan.

        Returns a flat list of new ``Task`` objects (ids guaranteed unique
        via a random suffix) whose dependencies may reference ids from
        ``completed_summary`` (already-succeeded tasks) in addition to each
        other. The orchestrator is responsible for splicing these into the
        live graph.
        """
        prompt = (
            f"Original goal: {goal}\n\n"
            f"Previously completed tasks and their outputs:\n"
            + "\n".join(f"- [{tid}]: {summary}" for tid, summary in completed_summary.items())
            + f"\n\nThe current plan failed and needs revision. Reason: {failure_reason}\n\n"
            "Produce a NEW plan for the remaining work needed to still achieve the "
            "goal, given what's already completed above. You may depend on the "
            "completed task ids listed above in addition to your new tasks."
        )
        response = await self.provider.complete(system=_PLANNER_SYSTEM, messages=[LLMMessage("user", prompt)])
        graph = self._parse_plan(response.text, capabilities, tools, known_external_ids=set(completed_summary))
        suffix = uuid.uuid4().hex[:6]
        remapped: dict[str, str] = {}
        new_tasks: list[Task] = []
        for task in graph.tasks.values():
            new_id = f"{task.id}_{suffix}"
            remapped[task.id] = new_id
        for task in graph.tasks.values():
            new_deps = [
                remapped[d] if d in remapped else d for d in task.dependencies
            ]
            new_tasks.append(
                Task(
                    id=remapped[task.id],
                    objective=task.objective,
                    capability=task.capability,
                    dependencies=new_deps,
                    required_tools=task.required_tools,
                    expected_output=task.expected_output,
                )
            )
        if self.event_log:
            self.event_log.emit(
                "REPLAN",
                f"Generated {len(new_tasks)} replacement task(s). Reason: {failure_reason}",
                model=response.model,
                tokens_used=response.total_tokens,
            )
        return new_tasks

    def _build_prompt(self, goal: str, capabilities: list[str], tools: list[str]) -> str:
        return (
            f"User goal: {goal}\n\n"
            f"Available capabilities: {', '.join(capabilities)}\n"
            f"Available tools: {', '.join(tools) if tools else '(none)'}"
        )

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
