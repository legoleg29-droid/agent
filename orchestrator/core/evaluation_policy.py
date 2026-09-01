"""EvaluationPolicy: what to check, how strict to be, and when an LLM
judge is worth the cost.

``DETERMINISTIC FIRST -> LLM EVALUATION ONLY WHEN NEEDED``: semantic
(LLM) evaluation only runs when deterministic checks can't settle the
question - free-text acceptance criteria exist, or the deterministic pass
was inconclusive (PARTIAL) - never for a clean deterministic PASS/FAIL
with concrete evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from orchestrator.core.evaluation_models import AcceptanceCriterion, EvaluationResult, EvaluationStatus

SemanticTrigger = Literal["never", "always", "on_free_text_criteria", "on_uncertain"]


@dataclass
class EvaluationPolicy:
    minimum_score: float = 0.75
    semantic_trigger: SemanticTrigger = "on_free_text_criteria"
    max_repair_attempts: int = 2
    max_retry_attempts: int = 2
    repair_allowed: bool = True
    # Below this deterministic score, don't even bother with a repair pass -
    # the result is too far off to be a "fix the last mile" job; go straight
    # to REPLAN_REQUIRED territory instead.
    min_score_for_repair: float = 0.15

    def requires_semantic_evaluation(
        self, criteria: list[AcceptanceCriterion], deterministic_result: EvaluationResult
    ) -> bool:
        if self.semantic_trigger == "never":
            return False
        if self.semantic_trigger == "always":
            return True
        has_free_text = any(c.type is None for c in criteria)
        if self.semantic_trigger == "on_free_text_criteria":
            return has_free_text
        # on_uncertain: only bother the model when the deterministic pass
        # didn't produce a confident verdict on its own.
        return deterministic_result.status == EvaluationStatus.PARTIAL or has_free_text

    def score_to_status(self, score: float, *, has_hard_failure: bool) -> EvaluationStatus:
        if has_hard_failure:
            return EvaluationStatus.FAIL
        if score >= self.minimum_score:
            return EvaluationStatus.PASS
        if score > 0:
            return EvaluationStatus.PARTIAL
        return EvaluationStatus.FAIL

    def repair_is_viable(self, score: float) -> bool:
        return self.repair_allowed and score >= self.min_score_for_repair
