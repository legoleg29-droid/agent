"""Structured, tag-based observability for the orchestrator.

Every significant lifecycle event is emitted through :class:`EventLog`, which
both prints a human-readable ``[TAG] message`` line and appends a structured
record (dict) to an in-memory list. Tests and downstream tooling can inspect
``EventLog.events`` instead of scraping stdout.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any

VALID_TAGS = {
    "ORCHESTRATOR",
    "PLANNER",
    "ROUTER",
    "TASK",
    "AGENT",
    "EVALUATOR",
    "RETRY",
    "REPLAN",
    "COMPLETE",
    # Fine-grained tool runtime lifecycle (Phase 2).
    "TOOL_REQUEST",
    "TOOL_VALIDATION",
    "TOOL_PERMISSION",
    "TOOL_EXECUTION",
    "TOOL_RESULT",
    "TOOL_ERROR",
    # State/memory lifecycle (Phase 3).
    "STATE_CREATED",
    "STATE_UPDATED",
    "TASK_STATE_CHANGED",
    "MEMORY_STORED",
    "MEMORY_RETRIEVED",
    "MEMORY_SKIPPED",
    "CHECKPOINT",
    "RESUME",
    # DAG scheduler lifecycle (Phase 4).
    "SCHEDULER_STARTED",
    "SCHEDULER_WAITING",
    "SCHEDULER_RESUMED",
    "SCHEDULER_COMPLETED",
    "TASK_READY",
    "TASK_STARTED",
    "TASK_COMPLETED",
    "TASK_FAILED",
    "TASK_RETRYING",
    "TASK_BLOCKED",
    "TASK_CANCELLED",
    "EXECUTION_PAUSED",
    "EXECUTION_RESUMED",
    # Evaluation / self-repair / replanning lifecycle (Phase 5).
    "EVALUATION_STARTED",
    "EVALUATION_COMPLETED",
    "EVALUATION_FAILED",
    "REPAIR_STARTED",
    "REPAIR_COMPLETED",
    "REPAIR_FAILED",
    "REPLAN_STARTED",
    "REPLAN_COMPLETED",
    "TASK_INVALIDATED",
    "PLAN_VERSION_CREATED",
    "LOOP_LIMIT_REACHED",
}

_logger = logging.getLogger("orchestrator")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)


@dataclass
class Event:
    tag: str
    message: str
    timestamp: float = field(default_factory=time.time)
    task_id: str | None = None
    agent_id: str | None = None
    tool_id: str | None = None
    model: str | None = None
    duration_ms: float | None = None
    status: str | None = None
    retry_count: int | None = None
    tokens_used: int | None = None
    tool_calls: int | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "tag": self.tag,
            "message": self.message,
            "timestamp": self.timestamp,
        }
        for key in (
            "task_id",
            "agent_id",
            "tool_id",
            "model",
            "duration_ms",
            "status",
            "retry_count",
            "tokens_used",
            "tool_calls",
            "error",
        ):
            value = getattr(self, key)
            if value is not None:
                d[key] = value
        if self.extra:
            d.update(self.extra)
        return d


class EventLog:
    """Collects structured events and mirrors them to stdout logging."""

    def __init__(self, *, verbose: bool = True) -> None:
        self.events: list[Event] = []
        self.verbose = verbose

    def emit(self, tag: str, message: str, **fields: Any) -> Event:
        if tag not in VALID_TAGS:
            raise ValueError(f"Unknown observability tag: {tag}")
        event = Event(tag=tag, message=message, **fields)
        self.events.append(event)
        if self.verbose:
            suffix_bits = []
            for key in ("task_id", "agent_id", "tool_id", "status", "retry_count", "duration_ms"):
                value = getattr(event, key)
                if value is not None:
                    suffix_bits.append(f"{key}={value}")
            suffix = f" ({', '.join(suffix_bits)})" if suffix_bits else ""
            _logger.info("[%s] %s%s", tag, message, suffix)
        return event

    def events_for_task(self, task_id: str) -> list[Event]:
        return [e for e in self.events if e.task_id == task_id]
