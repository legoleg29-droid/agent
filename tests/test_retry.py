from orchestrator.core.evaluation_models import EvaluationResult, EvaluationStatus
from orchestrator.core.retry import Action, RetryPolicy
from orchestrator.core.state import StateManager
from orchestrator.core.task_graph import Task


def make_task(retry_count=0, max_retries=2, repair_count=0, max_repairs=2) -> Task:
    return Task(
        id="t1",
        objective="obj",
        capability="research",
        retry_count=retry_count,
        max_retries=max_retries,
        repair_count=repair_count,
        max_repairs=max_repairs,
    )


def test_pass_continues():
    policy = RetryPolicy()
    action = policy.decide(make_task(), EvaluationResult(status=EvaluationStatus.PASS, passed=True), StateManager())
    assert action == Action.CONTINUE


def test_partial_continues():
    policy = RetryPolicy()
    action = policy.decide(make_task(), EvaluationResult(status=EvaluationStatus.PARTIAL), StateManager())
    assert action == Action.CONTINUE


def test_retry_possible_retries_while_budget_remains():
    policy = RetryPolicy()
    evaluation = EvaluationResult(status=EvaluationStatus.FAIL, retry_possible=True)
    action = policy.decide(make_task(retry_count=0, max_retries=2), evaluation, StateManager())
    assert action == Action.RETRY


def test_repair_required_and_possible_repairs():
    policy = RetryPolicy()
    evaluation = EvaluationResult(status=EvaluationStatus.REPAIR_REQUIRED, repair_possible=True)
    action = policy.decide(make_task(), evaluation, StateManager())
    assert action == Action.REPAIR


def test_repair_required_but_not_possible_falls_through_to_abort():
    policy = RetryPolicy()
    evaluation = EvaluationResult(status=EvaluationStatus.REPAIR_REQUIRED, repair_possible=False)
    action = policy.decide(make_task(), evaluation, StateManager())
    assert action == Action.ABORT


def test_replan_required_replans_when_budget_remains():
    policy = RetryPolicy()
    state = StateManager(max_replans=1)
    evaluation = EvaluationResult(status=EvaluationStatus.REPLAN_REQUIRED, replan_required=True)
    action = policy.decide(make_task(), evaluation, state)
    assert action == Action.REPLAN


def test_replan_required_aborts_when_replan_budget_exhausted():
    policy = RetryPolicy()
    state = StateManager(max_replans=0)
    evaluation = EvaluationResult(status=EvaluationStatus.REPLAN_REQUIRED, replan_required=True)
    action = policy.decide(make_task(), evaluation, state)
    assert action == Action.ABORT


def test_terminal_failure_aborts():
    policy = RetryPolicy()
    evaluation = EvaluationResult(status=EvaluationStatus.FAIL, retry_possible=False, replan_required=False)
    action = policy.decide(make_task(), evaluation, StateManager())
    assert action == Action.ABORT


def test_invalid_status_aborts():
    policy = RetryPolicy()
    evaluation = EvaluationResult(status=EvaluationStatus.INVALID, retry_possible=False, replan_required=False)
    action = policy.decide(make_task(), evaluation, StateManager())
    assert action == Action.ABORT
