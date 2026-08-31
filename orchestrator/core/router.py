"""Agent Router: maps a task's required capability to a concrete agent.

This is the seam that keeps the orchestrator free of hardcoded
"if task == research: use ResearchAgent" logic - it queries the registry
for capability matches and applies an explicit, inspectable selection rule.
"""

from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.logging_utils import EventLog
from orchestrator.core.task_graph import Task


class NoAgentForCapabilityError(LookupError):
    pass


class AgentRouter:
    def __init__(self, registry: AgentRegistry, event_log: EventLog | None = None) -> None:
        self.registry = registry
        self.event_log = event_log

    def route(self, task: Task) -> BaseAgent:
        candidates = self.registry.find_by_capability(task.capability)
        if not candidates:
            raise NoAgentForCapabilityError(
                f"No registered agent declares capability '{task.capability}' "
                f"(required by task '{task.id}')"
            )
        # Prefer a candidate that also covers all required tools, if any do.
        tool_covering = [
            a for a in candidates if set(task.required_tools).issubset(set(a.available_tools))
        ]
        chosen = (tool_covering or candidates)[0]
        if self.event_log:
            self.event_log.emit(
                "ROUTER",
                f"Routed task '{task.id}' (capability={task.capability}) to agent '{chosen.id}'",
                task_id=task.id,
                agent_id=chosen.id,
                extra={"candidates": [c.id for c in candidates]},
            )
        return chosen
