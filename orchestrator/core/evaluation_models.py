"""Evaluation schemas.

Deliberately separate from the old Phase 1-4 pass/fail ``Verdict`` - this
is a structured, multi-signal result: a status, a score, independent
evidence, and explicit repair/retry/replan signals, so the orchestrator
never has to infer intent from a single boolean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvaluationStatus(str, Enum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    INVALID = "invalid"              # the output itself is malformed/unusable (structural failure)
    REPAIR_REQUIRED = "repair_required"
    REPLAN_REQUIRED = "replan_required"


class EvaluatorType(str, Enum):
    """Which layer produced (or most decisively contributed to) a result -
    deterministic evidence is trusted more than subjective model scoring."""

    STRUCTURAL = "structural"
    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"
    HOLISTIC = "holistic"
    COMBINED = "combined"


class CriterionType(str, Enum):
    """Structured (deterministic, tool-backed) acceptance criterion kinds.
    A plain string criterion (not one of these) is treated as free-text and
    deferred to the semantic (LLM) evaluator - see AcceptanceCriterion."""

    FILE_EXISTS = "file_exists"
    ARTIFACT_EXISTS = "artifact_exists"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    JSON_VALID = "json_valid"
    TOOL_SUCCEEDED = "tool_succeeded"
    MIN_LENGTH = "min_length"


@dataclass
class AcceptanceCriterion:
    """One acceptance criterion for a task. Either a structured,
    deterministically-checkable assertion (``type`` set) or free-text
    handed to the semantic evaluator (``type`` is ``None``)."""

    description: str
    type: CriterionType | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: Any) -> AcceptanceCriterion:
        if isinstance(value, AcceptanceCriterion):
            return value
        if isinstance(value, str):
            return cls(description=value)
        if isinstance(value, dict):
            raw_type = value.get("type")
            return cls(
                description=value.get("description", str(value)),
                type=CriterionType(raw_type) if raw_type else None,
                params={k: v for k, v in value.items() if k not in ("type", "description")},
            )
        raise TypeError(f"Unsupported acceptance criterion: {value!r}")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"description": self.description}
        if self.type is not None:
            d["type"] = self.type.value
            d.update(self.params)
        return d


@dataclass
class EvaluationContext:
    """Only what's relevant to evaluating *this* task - direct dependency
    outputs, this task's own tool results, and artifacts it produced.
    Never the full memory database or unrelated tasks' state."""

    dependency_outputs: dict[str, str] = field(default_factory=dict)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvaluationResult:
    status: EvaluationStatus
    score: float = 0.0                       # 0.0-1.0
    passed: bool = False
    reasons: list[str] = field(default_factory=list)
    failed_criteria: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repair_possible: bool = False
    retry_possible: bool = False
    replan_required: bool = False
    evaluator_type: EvaluatorType = EvaluatorType.DETERMINISTIC
    confidence: float = 1.0                  # 0.0-1.0 - deterministic checks default to full confidence
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "score": self.score,
            "passed": self.passed,
            "reasons": self.reasons,
            "failed_criteria": self.failed_criteria,
            "warnings": self.warnings,
            "repair_possible": self.repair_possible,
            "retry_possible": self.retry_possible,
            "replan_required": self.replan_required,
            "evaluator_type": self.evaluator_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }
