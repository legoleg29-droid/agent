"""Retry / recovery policy.

Translates an ``EvaluationResult`` + task/run state into a concrete
action. Budget gating (retry/repair exhausted -> escalate) already
happened inside ``Evaluator._finalize`` - this is a thin, final mapping
from the evaluator's own signals to an executable action, kept separate so
the escalation policy (evaluator) and the action vocabulary (this module)
can be tested independently.
"""

from __future__ import annotations

from enum import Enum

from orchestrator.core.evaluation_models import EvaluationResult, EvaluationStatus
from orchestrator.core.state import StateManager
from orchestrator.core.task_graph import Task


class Action(str, Enum):
    CONTINUE = "continue"
    RETRY = "retry"
    REPAIR = "repair"
    REPLAN = "replan"
    ABORT = "abort"


class RetryPolicy:
    def decide(self, task: Task, evaluation: EvaluationResult, state: StateManager) -> Action:
        if evaluation.status in (EvaluationStatus.PASS, EvaluationStatus.PARTIAL):
            return Action.CONTINUE

        if evaluation.status == EvaluationStatus.REPAIR_REQUIRED and evaluation.repair_possible:
            return Action.REPAIR

        if evaluation.retry_possible:
            return Action.RETRY

        if evaluation.replan_required or evaluation.status == EvaluationStatus.REPLAN_REQUIRED:
            return Action.REPLAN if state.can_replan() else Action.ABORT

        return Action.ABORT  # EvaluationStatus.FAIL / INVALID with no recovery signal: terminal
