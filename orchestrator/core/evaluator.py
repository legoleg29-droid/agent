"""Layered Evaluator: independently judges task results instead of
trusting an agent's own success flag or textual claim.

    DETERMINISTIC FIRST -> LLM EVALUATION ONLY WHEN NEEDED

LEVEL 1 (structural)    - did the agent actually report success, is there
                          non-empty output, no refusal language.
LEVEL 2 (deterministic) - explicit, tool-backed acceptance criteria: files
                          exist, JSON parses, required tool succeeded, text
                          contains/excludes a string.
LEVEL 3/4 (semantic/holistic, combined) - only when free-text criteria
                          remain unverified or the deterministic pass was
                          inconclusive: a separate Claude call, with its
                          own strict system prompt, asked to check each
                          criterion concretely rather than "is this good?".

Deterministic evidence is always weighted above LLM scoring - see
``_combine``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from orchestrator.agents.base import AgentOutput
from orchestrator.core.error_classification import RecoveryAction, classify_error, default_recovery_action
from orchestrator.core.evaluation_models import (
    AcceptanceCriterion,
    CriterionType,
    EvaluationContext,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorType,
)
from orchestrator.core.evaluation_policy import EvaluationPolicy
from orchestrator.core.task_graph import Task
from orchestrator.providers.base import LLMMessage, LLMProvider
from orchestrator.tools.sandbox import FileSandbox

_FAILURE_PHRASES = (
    "i cannot",
    "i can't",
    "i don't have access",
    "unable to complete",
    "as an ai",
    "i'm not able to",
)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_JUDGE_SYSTEM = """You are an independent evaluator for an AI agent orchestrator.
You did NOT produce the output being evaluated - your job is to check it
critically against explicit acceptance criteria, not to rubber-stamp the
agent's own claims about what it did.

For each acceptance criterion, decide SATISFIED or VIOLATED based only on
concrete evidence in the actual output text given below. Do not assume a
claim is true because the output asserts it - look for the substance.
Distinguish facts you can verify in the text from assumptions you cannot.

Respond with ONLY a JSON object, no prose, no markdown fences:
{
  "criteria_results": [
    {"criterion": "<criterion text>", "satisfied": true, "evidence": "<concrete evidence or lack thereof>"}
  ],
  "overall_score": 0.0,
  "concerns": ["<any additional concerns, even if not tied to a specific criterion>"]
}
"""


class Evaluator:
    def __init__(
        self,
        *,
        provider: LLMProvider | None = None,
        policy: EvaluationPolicy | None = None,
        sandbox: FileSandbox | None = None,
    ) -> None:
        self.provider = provider
        self.policy = policy or EvaluationPolicy()
        self.sandbox = sandbox

    async def evaluate(
        self, task: Task, output: AgentOutput, *, context: EvaluationContext | None = None
    ) -> EvaluationResult:
        context = context or EvaluationContext()

        structural = self._structural_check(output)
        if structural.status != EvaluationStatus.PASS:
            return self._finalize(task, structural)

        criteria = [AcceptanceCriterion.from_any(c) for c in task.acceptance_criteria]
        result = self._deterministic_check(criteria, output, context)

        if self.provider is not None and self.policy.requires_semantic_evaluation(criteria, result):
            semantic = await self._semantic_check(task, output, criteria)
            result = self._combine(result, semantic)

        return self._finalize(task, result)

    # -- Level 1: structural ------------------------------------------------

    def _structural_check(self, output: AgentOutput) -> EvaluationResult:
        if not output.success or output.error:
            category = classify_error(error_text=output.error)
            action = default_recovery_action(category)
            return EvaluationResult(
                status=EvaluationStatus.REPLAN_REQUIRED if action == RecoveryAction.REPLAN else EvaluationStatus.FAIL,
                score=0.0,
                passed=False,
                reasons=[f"Agent reported failure: {output.error or 'unknown error'}"],
                failed_criteria=["task execution succeeded"],
                retry_possible=action == RecoveryAction.RETRY,
                replan_required=action == RecoveryAction.REPLAN,
                evaluator_type=EvaluatorType.STRUCTURAL,
                confidence=1.0,
                evidence=[f"output.success={output.success}", f"error_category={category.value}"],
                metadata={"error_category": category.value},
            )

        content = (output.content or "").strip()
        if not content:
            return EvaluationResult(
                status=EvaluationStatus.FAIL,
                passed=False,
                reasons=["Agent produced empty output"],
                failed_criteria=["non-empty output"],
                retry_possible=True,
                evaluator_type=EvaluatorType.STRUCTURAL,
                evidence=["output.content is empty"],
            )
        if any(phrase in content.lower() for phrase in _FAILURE_PHRASES):
            return EvaluationResult(
                status=EvaluationStatus.FAIL,
                passed=False,
                reasons=["Output indicates the agent could not complete the objective"],
                failed_criteria=["agent completed the objective"],
                retry_possible=True,
                evaluator_type=EvaluatorType.STRUCTURAL,
                evidence=["refusal/incapability phrase detected in output"],
            )
        return EvaluationResult(status=EvaluationStatus.PASS, score=1.0, passed=True, evaluator_type=EvaluatorType.STRUCTURAL)

    # -- Level 2: deterministic ----------------------------------------------

    def _deterministic_check(
        self, criteria: list[AcceptanceCriterion], output: AgentOutput, context: EvaluationContext
    ) -> EvaluationResult:
        structured = [c for c in criteria if c.type is not None]
        free_text_count = sum(1 for c in criteria if c.type is None)

        if not structured:
            content = (output.content or "").strip()
            if len(content) < 20:
                return EvaluationResult(
                    status=EvaluationStatus.PARTIAL,
                    score=0.4,
                    passed=False,
                    reasons=["Output is unusually short for the expected deliverable"],
                    evaluator_type=EvaluatorType.DETERMINISTIC,
                    evidence=[f"content_length={len(content)}"],
                )
            return EvaluationResult(
                status=EvaluationStatus.PASS,
                score=1.0,
                passed=free_text_count == 0,
                evaluator_type=EvaluatorType.DETERMINISTIC,
                evidence=[f"content_length={len(content)}", "no structured acceptance criteria defined"],
            )

        evidence: list[str] = []
        failed: list[str] = []
        for criterion in structured:
            ok, note = self._check_criterion(criterion, output, context)
            evidence.append(note)
            if not ok:
                failed.append(criterion.description)

        score = (len(structured) - len(failed)) / len(structured)
        if failed:
            status = EvaluationStatus.REPAIR_REQUIRED if score > 0 else EvaluationStatus.FAIL
        elif free_text_count:
            status = EvaluationStatus.PARTIAL  # deterministic side is clean, still need semantic confirmation
        else:
            status = EvaluationStatus.PASS

        return EvaluationResult(
            status=status,
            score=score,
            passed=(not failed) and free_text_count == 0,
            reasons=[f"{len(structured) - len(failed)}/{len(structured)} deterministic acceptance criteria satisfied"],
            failed_criteria=failed,
            evaluator_type=EvaluatorType.DETERMINISTIC,
            evidence=evidence,
            confidence=1.0,
            repair_possible=bool(failed) and score > 0,
        )

    def _check_criterion(
        self, criterion: AcceptanceCriterion, output: AgentOutput, context: EvaluationContext
    ) -> tuple[bool, str]:
        params = criterion.params
        content = output.content or ""

        if criterion.type == CriterionType.CONTAINS:
            text = str(params.get("text", ""))
            ok = text.lower() in content.lower()
            return ok, f"contains({text!r})={ok}"

        if criterion.type == CriterionType.NOT_CONTAINS:
            text = str(params.get("text", ""))
            ok = text.lower() not in content.lower()
            return ok, f"not_contains({text!r})={ok}"

        if criterion.type == CriterionType.MIN_LENGTH:
            n = int(params.get("length", 0))
            ok = len(content) >= n
            return ok, f"min_length({n})={ok} (actual={len(content)})"

        if criterion.type == CriterionType.JSON_VALID:
            try:
                json.loads(content)
                return True, "json_valid=True"
            except Exception as exc:  # noqa: BLE001
                return False, f"json_valid=False ({type(exc).__name__})"

        if criterion.type == CriterionType.FILE_EXISTS:
            path = params.get("path")
            if not path:
                return False, "file_exists: no path given"
            if self.sandbox is None:
                return False, f"file_exists({path})=unknown (no sandbox configured for this evaluator)"
            try:
                exists = self.sandbox.resolve(path).exists()
            except Exception:  # noqa: BLE001 - a sandbox violation means "does not exist" for eval purposes
                exists = False
            return exists, f"file_exists({path})={exists}"

        if criterion.type == CriterionType.ARTIFACT_EXISTS:
            artifact_type = params.get("artifact_type")
            matches = [a for a in context.artifacts if artifact_type is None or a.get("type") == artifact_type]
            ok = bool(matches)
            return ok, f"artifact_exists(type={artifact_type})={ok}"

        if criterion.type == CriterionType.TOOL_SUCCEEDED:
            tool_name = params.get("tool")
            matches = [r for r in context.tool_results if tool_name is None or r.get("tool") == tool_name]
            ok = any(r.get("status") == "success" for r in matches)
            return ok, f"tool_succeeded(tool={tool_name})={ok}"

        return False, f"unrecognized criterion type: {criterion.type}"

    # -- Level 3/4: semantic + holistic (combined into one judge call) ------

    async def _semantic_check(
        self, task: Task, output: AgentOutput, criteria: list[AcceptanceCriterion]
    ) -> EvaluationResult:
        free_text = [c for c in criteria if c.type is None]
        if not free_text:
            free_text = [
                AcceptanceCriterion(
                    description=task.expected_output or "The output correctly and completely satisfies the task objective."
                )
            ]

        prompt = (
            f"Task objective: {task.objective}\n"
            f"Expected output: {task.expected_output or '(not specified)'}\n\n"
            "Acceptance criteria to check:\n" + "\n".join(f"- {c.description}" for c in free_text) + "\n\n"
            f"Actual output to evaluate:\n{output.content}"
        )
        response = await self.provider.complete(system=_JUDGE_SYSTEM, messages=[LLMMessage("user", prompt)])

        match = _JSON_BLOCK_RE.search(response.text)
        if not match:
            return EvaluationResult(
                status=EvaluationStatus.PARTIAL,
                score=0.5,
                evaluator_type=EvaluatorType.SEMANTIC,
                warnings=["LLM judge response was not valid JSON - treated as inconclusive"],
                confidence=0.3,
            )
        try:
            payload: dict[str, Any] = json.loads(match.group(0))
        except json.JSONDecodeError:
            return EvaluationResult(
                status=EvaluationStatus.PARTIAL,
                score=0.5,
                evaluator_type=EvaluatorType.SEMANTIC,
                warnings=["LLM judge response was not valid JSON - treated as inconclusive"],
                confidence=0.3,
            )

        criteria_results = payload.get("criteria_results", [])
        failed = [r["criterion"] for r in criteria_results if not r.get("satisfied", False)]
        evidence = [f"{r.get('criterion')}: {r.get('evidence', '')}" for r in criteria_results]
        score = payload.get("overall_score")
        if not isinstance(score, (int, float)):
            score = (len(criteria_results) - len(failed)) / len(criteria_results) if criteria_results else 0.5
        score = max(0.0, min(1.0, float(score)))

        status = EvaluationStatus.PASS if not failed else (EvaluationStatus.REPAIR_REQUIRED if score > 0 else EvaluationStatus.FAIL)
        return EvaluationResult(
            status=status,
            score=score,
            passed=not failed,
            failed_criteria=failed,
            warnings=list(payload.get("concerns", [])),
            evaluator_type=EvaluatorType.SEMANTIC,
            evidence=evidence,
            confidence=0.7,  # subjective model scoring is trusted less than deterministic evidence
            repair_possible=bool(failed) and score > 0,
        )

    def _combine(self, deterministic: EvaluationResult, semantic: EvaluationResult) -> EvaluationResult:
        # Deterministic evidence is weighted higher - it's ground truth,
        # not a model's subjective read.
        combined_score = deterministic.score * 0.6 + semantic.score * 0.4
        failed = list(dict.fromkeys(deterministic.failed_criteria + semantic.failed_criteria))
        passed = deterministic.passed and semantic.passed and not failed
        if passed:
            status = EvaluationStatus.PASS
        elif combined_score > 0:
            status = EvaluationStatus.REPAIR_REQUIRED
        else:
            status = EvaluationStatus.FAIL
        return EvaluationResult(
            status=status,
            score=combined_score,
            passed=passed,
            reasons=deterministic.reasons + semantic.reasons,
            failed_criteria=failed,
            warnings=deterministic.warnings + semantic.warnings,
            evaluator_type=EvaluatorType.COMBINED,
            confidence=min(1.0, deterministic.confidence * 0.6 + semantic.confidence * 0.4),
            evidence=deterministic.evidence + semantic.evidence,
            repair_possible=bool(failed) and combined_score > 0,
        )

    # -- Finalization: apply retry/repair budgets ----------------------------

    def _finalize(self, task: Task, result: EvaluationResult) -> EvaluationResult:
        if result.retry_possible and task.retry_count >= task.max_retries:
            result.retry_possible = False
            result.replan_required = True
            result.status = EvaluationStatus.REPLAN_REQUIRED
            result.reasons.append(f"Retry budget exhausted ({task.retry_count}/{task.max_retries})")

        if result.status == EvaluationStatus.REPAIR_REQUIRED:
            if not self.policy.repair_is_viable(result.score) or task.repair_count >= task.max_repairs:
                result.repair_possible = False
                result.replan_required = True
                result.status = EvaluationStatus.REPLAN_REQUIRED
                result.reasons.append(f"Repair budget exhausted or not viable ({task.repair_count}/{task.max_repairs})")
            else:
                result.repair_possible = True

        result.passed = result.status == EvaluationStatus.PASS
        return result
