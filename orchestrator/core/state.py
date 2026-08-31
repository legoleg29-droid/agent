"""Run-level state tracking, separate from execution context.

Tracks orchestrator-wide status and phase transitions for observability
and for deciding when to stop (max replans, no ready tasks, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RunStatus(str, Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    REPLANNING = "replanning"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StateManager:
    status: RunStatus = RunStatus.PLANNING
    replans_used: int = 0
    max_replans: int = 2
    phase_history: list[str] = field(default_factory=list)

    def transition(self, status: RunStatus) -> None:
        self.phase_history.append(status.value)
        self.status = status

    def can_replan(self) -> bool:
        return self.replans_used < self.max_replans

    def record_replan(self) -> None:
        self.replans_used += 1
