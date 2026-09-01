from orchestrator.core.evaluation_models import AcceptanceCriterion, CriterionType, EvaluationResult, EvaluationStatus
from orchestrator.core.evaluation_policy import EvaluationPolicy


def test_never_trigger_skips_semantic_even_with_free_text():
    policy = EvaluationPolicy(semantic_trigger="never")
    criteria = [AcceptanceCriterion(description="reads well")]
    assert policy.requires_semantic_evaluation(criteria, EvaluationResult(status=EvaluationStatus.PASS)) is False


def test_always_trigger_requires_semantic_even_with_no_criteria():
    policy = EvaluationPolicy(semantic_trigger="always")
    assert policy.requires_semantic_evaluation([], EvaluationResult(status=EvaluationStatus.PASS)) is True


def test_on_free_text_criteria_only_triggers_when_a_free_text_criterion_exists():
    policy = EvaluationPolicy(semantic_trigger="on_free_text_criteria")
    structured_only = [AcceptanceCriterion(description="x", type=CriterionType.MIN_LENGTH, params={"length": 5})]
    with_free_text = structured_only + [AcceptanceCriterion(description="reads well")]
    assert policy.requires_semantic_evaluation(structured_only, EvaluationResult(status=EvaluationStatus.PASS)) is False
    assert policy.requires_semantic_evaluation(with_free_text, EvaluationResult(status=EvaluationStatus.PASS)) is True


def test_on_uncertain_triggers_on_partial_even_without_free_text():
    policy = EvaluationPolicy(semantic_trigger="on_uncertain")
    structured_only = [AcceptanceCriterion(description="x", type=CriterionType.MIN_LENGTH, params={"length": 5})]
    assert policy.requires_semantic_evaluation(structured_only, EvaluationResult(status=EvaluationStatus.PASS)) is False
    assert policy.requires_semantic_evaluation(structured_only, EvaluationResult(status=EvaluationStatus.PARTIAL)) is True


def test_score_to_status_thresholds():
    policy = EvaluationPolicy(minimum_score=0.75)
    assert policy.score_to_status(1.0, has_hard_failure=False) == EvaluationStatus.PASS
    assert policy.score_to_status(0.75, has_hard_failure=False) == EvaluationStatus.PASS
    assert policy.score_to_status(0.5, has_hard_failure=False) == EvaluationStatus.PARTIAL
    assert policy.score_to_status(0.0, has_hard_failure=False) == EvaluationStatus.FAIL
    assert policy.score_to_status(1.0, has_hard_failure=True) == EvaluationStatus.FAIL


def test_repair_is_viable_respects_minimum_score_and_allowed_flag():
    policy = EvaluationPolicy(min_score_for_repair=0.15)
    assert policy.repair_is_viable(0.5) is True
    assert policy.repair_is_viable(0.1) is False
    assert EvaluationPolicy(repair_allowed=False).repair_is_viable(0.9) is False
