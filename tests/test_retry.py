from orchestrator.core.evaluator import EvaluationResult, Verdict
from orchestrator.core.retry import Action, RetryPolicy
from orchestrator.core.state import StateManager
from orchestrator.core.task_graph import Task


def make_task(retry_count=0, max_retries=2) -> Task:
    return Task(id="t1", objective="obj", capability="research", retry_count=retry_count, max_retries=max_retries)


def test_success_continues():
    policy = RetryPolicy()
    action = policy.decide(make_task(), EvaluationResult(Verdict.SUCCESS, "ok"), StateManager())
    assert action == Action.CONTINUE


def test_partial_success_continues():
    policy = RetryPolicy()
    action = policy.decide(make_task(), EvaluationResult(Verdict.PARTIAL_SUCCESS, "meh"), StateManager())
    assert action == Action.CONTINUE


def test_retry_required_retries_while_budget_remains():
    policy = RetryPolicy()
    action = policy.decide(make_task(retry_count=0, max_retries=2), EvaluationResult(Verdict.RETRY_REQUIRED, "x"), StateManager())
    assert action == Action.RETRY


def test_retry_required_falls_back_to_replan_when_retries_exhausted():
    policy = RetryPolicy()
    action = policy.decide(make_task(retry_count=2, max_retries=2), EvaluationResult(Verdict.RETRY_REQUIRED, "x"), StateManager())
    assert action == Action.REPLAN


def test_replan_required_replans_when_budget_remains():
    policy = RetryPolicy()
    state = StateManager(max_replans=1)
    action = policy.decide(make_task(), EvaluationResult(Verdict.REPLAN_REQUIRED, "x"), state)
    assert action == Action.REPLAN


def test_replan_required_aborts_when_replan_budget_exhausted():
    policy = RetryPolicy()
    state = StateManager(max_replans=0)
    action = policy.decide(make_task(), EvaluationResult(Verdict.REPLAN_REQUIRED, "x"), state)
    assert action == Action.ABORT


def test_terminal_failure_aborts():
    policy = RetryPolicy()
    action = policy.decide(make_task(), EvaluationResult(Verdict.FAILURE, "x"), StateManager())
    assert action == Action.ABORT
