"""Evaluator: independently judges task results instead of trusting
an agent's own success flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from orchestrator.agents.base import AgentOutput
from orchestrator.core.task_graph import Task


class Verdict(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    RETRY_REQUIRED = "retry_required"
    REPLAN_REQUIRED = "replan_required"


@dataclass
class EvaluationResult:
    verdict: Verdict
    reason: str


_FAILURE_PHRASES = (
    "i cannot",
    "i can't",
    "i don't have access",
    "unable to complete",
    "as an ai",
    "i'm not able to",
)
_MIN_ACCEPTABLE_LENGTH = 20


class Evaluator:
    """Rule-based evaluator. Never trusts ``AgentOutput.success`` alone."""

    def evaluate(self, task: Task, output: AgentOutput) -> EvaluationResult:
        retriable = task.retry_count < task.max_retries

        if not output.success or output.error:
            verdict = Verdict.RETRY_REQUIRED if retriable else Verdict.REPLAN_REQUIRED
            return EvaluationResult(verdict, f"Agent reported failure: {output.error or 'unknown error'}")

        content = (output.content or "").strip()
        if not content:
            verdict = Verdict.RETRY_REQUIRED if retriable else Verdict.REPLAN_REQUIRED
            return EvaluationResult(verdict, "Agent produced empty output")

        lowered = content.lower()
        if any(phrase in lowered for phrase in _FAILURE_PHRASES):
            verdict = Verdict.RETRY_REQUIRED if retriable else Verdict.REPLAN_REQUIRED
            return EvaluationResult(verdict, "Output indicates the agent could not complete the objective")

        if len(content) < _MIN_ACCEPTABLE_LENGTH:
            return EvaluationResult(
                Verdict.PARTIAL_SUCCESS, "Output is unusually short for the expected deliverable"
            )

        return EvaluationResult(Verdict.SUCCESS, "Output satisfies basic completeness checks")
