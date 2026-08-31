"""Retry / recovery policy.

Translates an evaluator verdict + task/run state into a concrete action:
retry the same task, trigger a replan, continue, or abort safely.
"""

from __future__ import annotations

from enum import Enum

from orchestrator.core.evaluator import EvaluationResult, Verdict
from orchestrator.core.state import StateManager
from orchestrator.core.task_graph import Task


class Action(str, Enum):
    CONTINUE = "continue"
    RETRY = "retry"
    REPLAN = "replan"
    ABORT = "abort"


class RetryPolicy:
    def decide(self, task: Task, evaluation: EvaluationResult, state: StateManager) -> Action:
        if evaluation.verdict in (Verdict.SUCCESS, Verdict.PARTIAL_SUCCESS):
            return Action.CONTINUE

        if evaluation.verdict == Verdict.RETRY_REQUIRED:
            if task.retry_count < task.max_retries:
                return Action.RETRY
            return Action.REPLAN if state.can_replan() else Action.ABORT

        if evaluation.verdict == Verdict.REPLAN_REQUIRED:
            return Action.REPLAN if state.can_replan() else Action.ABORT

        return Action.ABORT  # Verdict.FAILURE: terminal, non-recoverable
