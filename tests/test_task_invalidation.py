import pytest

from orchestrator.core.task_graph import Task, TaskGraph, TaskStatus


def make_graph() -> TaskGraph:
    graph = TaskGraph()
    graph.add_task(Task(id="a", objective="a", capability="research", status=TaskStatus.SUCCEEDED))
    graph.add_task(Task(id="b", objective="b", capability="research", dependencies=["a"], status=TaskStatus.SUCCEEDED))
    return graph


def test_invalidate_task_requires_succeeded_status():
    graph = make_graph()
    graph.tasks["a"].status = TaskStatus.PENDING
    with pytest.raises(ValueError):
        graph.invalidate_task("a", reason="test")


def test_invalidate_task_sets_status_and_reason():
    graph = make_graph()
    task = graph.invalidate_task("a", reason="artifact deleted")
    assert task.status == TaskStatus.INVALIDATED
    assert task.metadata["invalidation_reason"] == "artifact deleted"


def test_invalidate_does_not_cascade_to_dependents():
    graph = make_graph()
    graph.invalidate_task("a", reason="stale")
    assert graph.tasks["b"].status == TaskStatus.SUCCEEDED  # untouched


def test_invalidated_tasks_lists_only_invalidated():
    graph = make_graph()
    graph.invalidate_task("a", reason="stale")
    assert [t.id for t in graph.invalidated_tasks()] == ["a"]


def test_reactivate_invalidated_tasks_resets_to_pending_and_clears_result():
    from orchestrator.agents.base import AgentOutput

    graph = make_graph()
    graph.tasks["a"].result = AgentOutput(success=True, content="stale result")
    graph.tasks["a"].completed_at = 12345.0
    graph.invalidate_task("a", reason="stale")

    reactivated = graph.reactivate_invalidated_tasks()

    assert [t.id for t in reactivated] == ["a"]
    assert graph.tasks["a"].status == TaskStatus.PENDING
    assert graph.tasks["a"].result is None
    assert graph.tasks["a"].completed_at is None


def test_invalidated_status_counts_as_pending_work():
    graph = make_graph()
    graph.invalidate_task("a", reason="stale")
    assert graph.has_pending_work() is True
