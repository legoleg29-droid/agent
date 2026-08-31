import pytest

from orchestrator.core.task_graph import CycleError, Task, TaskGraph, TaskStatus


def test_get_ready_tasks_respects_dependencies():
    graph = TaskGraph([
        Task(id="t1", objective="a", capability="research"),
        Task(id="t2", objective="b", capability="analysis", dependencies=["t1"]),
    ])
    assert [t.id for t in graph.get_ready_tasks()] == ["t1"]

    graph.tasks["t1"].status = TaskStatus.SUCCEEDED
    assert [t.id for t in graph.get_ready_tasks()] == ["t2"]


def test_independent_tasks_are_both_ready():
    graph = TaskGraph([
        Task(id="t1", objective="a", capability="research"),
        Task(id="t2", objective="b", capability="research"),
    ])
    assert {t.id for t in graph.get_ready_tasks()} == {"t1", "t2"}


def test_cycle_detection_raises():
    with pytest.raises(CycleError):
        TaskGraph([
            Task(id="t1", objective="a", capability="research", dependencies=["t2"]),
            Task(id="t2", objective="b", capability="research", dependencies=["t1"]),
        ])


def test_self_dependency_rejected():
    with pytest.raises(ValueError):
        TaskGraph([Task(id="t1", objective="a", capability="research", dependencies=["t1"])])


def test_mark_unreachable_as_skipped_cascades():
    graph = TaskGraph([
        Task(id="t1", objective="a", capability="research"),
        Task(id="t2", objective="b", capability="analysis", dependencies=["t1"]),
        Task(id="t3", objective="c", capability="writing", dependencies=["t2"]),
    ])
    graph.tasks["t1"].status = TaskStatus.FAILED
    skipped = graph.mark_unreachable_as_skipped()
    assert {t.id for t in skipped} == {"t2", "t3"}
    assert graph.is_complete()


def test_topological_order_respects_dependencies():
    graph = TaskGraph([
        Task(id="t2", objective="b", capability="analysis", dependencies=["t1"]),
        Task(id="t1", objective="a", capability="research"),
    ])
    order = [t.id for t in graph.topological_order()]
    assert order.index("t1") < order.index("t2")


def test_finalize_rejects_unknown_dependency():
    graph = TaskGraph()
    graph.add_task(Task(id="t1", objective="a", capability="research", dependencies=["ghost"]))
    with pytest.raises(ValueError):
        graph.finalize()
