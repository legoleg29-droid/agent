from orchestrator.agents.base import AgentOutput
from orchestrator.core.evaluator import Evaluator, Verdict
from orchestrator.core.task_graph import Task


def make_task(retry_count=0, max_retries=2) -> Task:
    return Task(id="t1", objective="obj", capability="research", retry_count=retry_count, max_retries=max_retries)


def test_success_output():
    evaluator = Evaluator()
    result = evaluator.evaluate(make_task(), AgentOutput(success=True, content="A thorough and complete answer to the objective."))
    assert result.verdict == Verdict.SUCCESS


def test_agent_reported_failure_is_not_trusted_blindly_but_becomes_retry():
    evaluator = Evaluator()
    task = make_task(retry_count=0, max_retries=2)
    result = evaluator.evaluate(task, AgentOutput(success=False, error="boom"))
    assert result.verdict == Verdict.RETRY_REQUIRED


def test_failure_after_retries_exhausted_requires_replan():
    evaluator = Evaluator()
    task = make_task(retry_count=2, max_retries=2)
    result = evaluator.evaluate(task, AgentOutput(success=False, error="boom"))
    assert result.verdict == Verdict.REPLAN_REQUIRED


def test_empty_content_is_not_trusted_as_success_even_if_flag_is_true():
    evaluator = Evaluator()
    result = evaluator.evaluate(make_task(), AgentOutput(success=True, content="   "))
    assert result.verdict == Verdict.RETRY_REQUIRED


def test_refusal_language_flagged_despite_success_flag():
    evaluator = Evaluator()
    result = evaluator.evaluate(make_task(), AgentOutput(success=True, content="I cannot complete this objective."))
    assert result.verdict == Verdict.RETRY_REQUIRED


def test_short_output_is_partial_success():
    evaluator = Evaluator()
    result = evaluator.evaluate(make_task(), AgentOutput(success=True, content="Too short."))
    assert result.verdict == Verdict.PARTIAL_SUCCESS
