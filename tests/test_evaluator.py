import pytest

from orchestrator.agents.base import AgentOutput
from orchestrator.core.evaluation_models import EvaluationStatus
from orchestrator.core.evaluator import Evaluator
from orchestrator.core.task_graph import Task


def make_task(retry_count=0, max_retries=2, acceptance_criteria=None) -> Task:
    return Task(
        id="t1",
        objective="obj",
        capability="research",
        retry_count=retry_count,
        max_retries=max_retries,
        acceptance_criteria=acceptance_criteria or [],
    )


@pytest.mark.asyncio
async def test_success_output():
    evaluator = Evaluator()
    result = await evaluator.evaluate(
        make_task(), AgentOutput(success=True, content="A thorough and complete answer to the objective.")
    )
    assert result.status == EvaluationStatus.PASS
    assert result.passed is True


@pytest.mark.asyncio
async def test_agent_reported_failure_is_not_trusted_blindly_but_becomes_retryable():
    evaluator = Evaluator()
    task = make_task(retry_count=0, max_retries=2)
    result = await evaluator.evaluate(task, AgentOutput(success=False, error="boom"))
    assert result.status != EvaluationStatus.PASS
    assert result.retry_possible is True


@pytest.mark.asyncio
async def test_failure_after_retries_exhausted_requires_replan():
    evaluator = Evaluator()
    task = make_task(retry_count=2, max_retries=2)
    result = await evaluator.evaluate(task, AgentOutput(success=False, error="boom"))
    assert result.status == EvaluationStatus.REPLAN_REQUIRED
    assert result.replan_required is True


@pytest.mark.asyncio
async def test_empty_content_is_not_trusted_as_success_even_if_flag_is_true():
    evaluator = Evaluator()
    result = await evaluator.evaluate(make_task(), AgentOutput(success=True, content="   "))
    assert result.status != EvaluationStatus.PASS
    assert result.retry_possible is True


@pytest.mark.asyncio
async def test_refusal_language_flagged_despite_success_flag():
    evaluator = Evaluator()
    result = await evaluator.evaluate(make_task(), AgentOutput(success=True, content="I cannot complete this objective."))
    assert result.status != EvaluationStatus.PASS
    assert result.retry_possible is True


@pytest.mark.asyncio
async def test_short_output_with_no_criteria_is_partial():
    evaluator = Evaluator()
    result = await evaluator.evaluate(make_task(), AgentOutput(success=True, content="Too short."))
    assert result.status == EvaluationStatus.PARTIAL


@pytest.mark.asyncio
async def test_deterministic_criterion_failure_is_repair_required():
    evaluator = Evaluator()
    # One satisfied + one failed criterion gives partial credit (score > 0),
    # which is what makes a failure REPAIR_REQUIRED rather than a terminal FAIL.
    task = make_task(
        acceptance_criteria=[{"type": "contains", "text": "fibonacci"}, {"type": "min_length", "length": 5}]
    )
    result = await evaluator.evaluate(task, AgentOutput(success=True, content="A generic answer with no relevant terms."))
    assert result.status == EvaluationStatus.REPAIR_REQUIRED
    assert result.failed_criteria
    assert result.repair_possible is True


@pytest.mark.asyncio
async def test_deterministic_criteria_all_satisfied_pass():
    evaluator = Evaluator()
    task = make_task(acceptance_criteria=[{"type": "contains", "text": "fibonacci"}, {"type": "min_length", "length": 5}])
    result = await evaluator.evaluate(task, AgentOutput(success=True, content="Here is a fibonacci implementation."))
    assert result.status == EvaluationStatus.PASS
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_repair_exhausted_escalates_to_replan():
    evaluator = Evaluator()
    task = make_task(
        acceptance_criteria=[{"type": "contains", "text": "fibonacci"}, {"type": "min_length", "length": 5}]
    )
    task.repair_count = task.max_repairs
    result = await evaluator.evaluate(task, AgentOutput(success=True, content="unrelated content"))
    assert result.status == EvaluationStatus.REPLAN_REQUIRED
    assert result.repair_possible is False


@pytest.mark.asyncio
async def test_artifact_exists_criterion():
    from orchestrator.core.evaluation_models import EvaluationContext

    evaluator = Evaluator()
    task = make_task(acceptance_criteria=[{"type": "artifact_exists", "artifact_type": "file"}])
    context = EvaluationContext(artifacts=[{"type": "file", "path": "fib.py"}])
    result = await evaluator.evaluate(task, AgentOutput(success=True, content="Created the file."), context=context)
    assert result.status == EvaluationStatus.PASS


@pytest.mark.asyncio
async def test_tool_succeeded_criterion():
    from orchestrator.core.evaluation_models import EvaluationContext

    evaluator = Evaluator()
    task = make_task(acceptance_criteria=[{"type": "tool_succeeded", "tool": "run_python_tests"}])
    context = EvaluationContext(tool_results=[{"tool": "run_python_tests", "status": "success"}])
    result = await evaluator.evaluate(task, AgentOutput(success=True, content="Tests pass."), context=context)
    assert result.status == EvaluationStatus.PASS
