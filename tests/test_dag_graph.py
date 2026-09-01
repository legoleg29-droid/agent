"""TaskGraph DAG API: creation, validation, cycle detection, dependency
queries, and the failure-isolation/cancellation cascades."""

import pytest

from orchestrator.core.task_graph import CycleError, Task, TaskGraph, TaskStatus


def make_task(**overrides) -> Task:
    defaults = dict(id="t1", objective="do it", capability="research")
    defaults.update(overrides)
    return Task(**defaults)


# -- Creation / duplicate ids -----------------------------------------------


def test_add_task_and_get_task():
    graph = TaskGraph()
    graph.add_task(make_task(id="a"))
    assert graph.get_task("a").id == "a"
    assert graph.get_task("missing") is None


def test_duplicate_task_id_rejected():
    graph = TaskGraph()
    graph.add_task(make_task(id="a"))
    with pytest.raises(ValueError):
        graph.add_task(make_task(id="a"))


def test_self_dependency_rejected_on_add_task():
    graph = TaskGraph()
    with pytest.raises(ValueError):
        graph.add_task(make_task(id="a", dependencies=["a"]))


# -- validate() / detect_cycles() (non-raising) ------------------------------


def test_validate_returns_empty_for_a_good_graph():
    graph = TaskGraph([make_task(id="a"), make_task(id="b", dependencies=["a"])])
    assert graph.validate() == []


def test_validate_reports_missing_dependency_without_raising():
    graph = TaskGraph()
    graph.add_task(make_task(id="a", dependencies=["ghost"]))
    errors = graph.validate()
    assert any("ghost" in e for e in errors)


def test_validate_reports_cycle():
    graph = TaskGraph()
    graph.add_task(make_task(id="a", dependencies=["c"]))
    graph.add_task(make_task(id="b", dependencies=["a"]))
    graph.add_task(make_task(id="c", dependencies=["b"]))
    errors = graph.validate()
    assert any("cycle" in e.lower() for e in errors)


def test_validate_reports_unknown_capability_and_tool_when_known_sets_given():
    graph = TaskGraph()
    graph.add_task(make_task(id="a", capability="mystery", required_tools=["nope"]))
    errors = graph.validate(known_capabilities={"research"}, known_tools={"web_search"})
    assert any("mystery" in e for e in errors)
    assert any("nope" in e for e in errors)


def test_detect_cycles_returns_none_for_acyclic_graph():
    graph = TaskGraph([make_task(id="a"), make_task(id="b", dependencies=["a"])])
    assert graph.detect_cycles() is None


def test_detect_cycles_returns_the_cycle_path():
    graph = TaskGraph()
    graph.add_task(make_task(id="a", dependencies=["b"]))
    graph.add_task(make_task(id="b", dependencies=["a"]))
    cycle = graph.detect_cycles()
    assert cycle is not None
    assert set(cycle) == {"a", "b"}


def test_finalize_raises_cycle_error_for_a_cycle():
    graph = TaskGraph()
    graph.add_task(make_task(id="a", dependencies=["b"]))
    graph.add_task(make_task(id="b", dependencies=["a"]))
    with pytest.raises(CycleError):
        graph.finalize()


def test_finalize_raises_value_error_for_missing_dependency():
    graph = TaskGraph()
    graph.add_task(make_task(id="a", dependencies=["ghost"]))
    with pytest.raises(ValueError):
        graph.finalize()


# -- remove_task / add_dependency / get_dependencies / get_dependents ------


def test_remove_task_with_no_dependents():
    graph = TaskGraph([make_task(id="a")])
    graph.remove_task("a")
    assert graph.get_task("a") is None


def test_remove_task_with_dependents_is_rejected():
    graph = TaskGraph([make_task(id="a"), make_task(id="b", dependencies=["a"])])
    with pytest.raises(ValueError):
        graph.remove_task("a")


def test_remove_missing_task_is_a_noop():
    graph = TaskGraph()
    graph.remove_task("does-not-exist")  # must not raise


def test_add_dependency_creates_edge():
    graph = TaskGraph([make_task(id="a"), make_task(id="b")])
    graph.add_dependency("b", "a")
    assert graph.get_task("b").dependencies == ["a"]


def test_add_dependency_rejects_unknown_tasks():
    graph = TaskGraph([make_task(id="a")])
    with pytest.raises(ValueError):
        graph.add_dependency("a", "ghost")
    with pytest.raises(ValueError):
        graph.add_dependency("ghost", "a")


def test_add_dependency_rejects_self_dependency():
    graph = TaskGraph([make_task(id="a")])
    with pytest.raises(ValueError):
        graph.add_dependency("a", "a")


def test_add_dependency_rejects_cycles_and_leaves_graph_unchanged():
    graph = TaskGraph([make_task(id="a"), make_task(id="b", dependencies=["a"])])
    with pytest.raises(CycleError):
        graph.add_dependency("a", "b")
    assert graph.get_task("a").dependencies == []  # rolled back


def test_get_dependencies_and_get_dependents():
    graph = TaskGraph([
        make_task(id="a"),
        make_task(id="b", dependencies=["a"]),
        make_task(id="c", dependencies=["a"]),
    ])
    assert {t.id for t in graph.get_dependencies("b")} == {"a"}
    assert {t.id for t in graph.get_dependents("a")} == {"b", "c"}
    assert graph.get_dependents("b") == []


# -- Ready detection ----------------------------------------------------


def test_ready_task_detection_matches_spec_example():
    graph = TaskGraph([
        make_task(id="A"),
        make_task(id="B", dependencies=["A"]),
        make_task(id="C", dependencies=["A"]),
        make_task(id="D", dependencies=["B", "C"]),
    ])
    assert [t.id for t in graph.get_ready_tasks()] == ["A"]

    graph.tasks["A"].status = TaskStatus.SUCCEEDED
    assert {t.id for t in graph.get_ready_tasks()} == {"B", "C"}

    graph.tasks["B"].status = TaskStatus.SUCCEEDED
    assert {t.id for t in graph.get_ready_tasks()} == {"C"}

    graph.tasks["C"].status = TaskStatus.SUCCEEDED
    assert {t.id for t in graph.get_ready_tasks()} == {"D"}


# -- Failure isolation (BLOCKED) vs full abort (SKIPPED/CANCELLED) ----------


def test_mark_blocked_by_failed_dependencies_isolates_independent_branches():
    graph = TaskGraph([
        make_task(id="A"),
        make_task(id="B"),  # independent of A
        make_task(id="C", dependencies=["A"]),
        make_task(id="D"),
    ])
    graph.tasks["A"].status = TaskStatus.FAILED
    blocked = graph.mark_blocked_by_failed_dependencies()

    assert {t.id for t in blocked} == {"C"}
    assert graph.tasks["B"].status == TaskStatus.PENDING  # untouched
    assert graph.tasks["D"].status == TaskStatus.PENDING  # untouched


def test_mark_blocked_by_failed_dependencies_cascades_transitively():
    graph = TaskGraph([
        make_task(id="A"),
        make_task(id="B", dependencies=["A"]),
        make_task(id="C", dependencies=["B"]),
    ])
    graph.tasks["A"].status = TaskStatus.FAILED
    blocked = graph.mark_blocked_by_failed_dependencies()
    assert {t.id for t in blocked} == {"B", "C"}


def test_mark_remaining_as_cancelled_never_marks_a_completed_task():
    graph = TaskGraph([make_task(id="a"), make_task(id="b", dependencies=["a"])])
    graph.tasks["a"].status = TaskStatus.SUCCEEDED
    cancelled = graph.mark_remaining_as_cancelled()
    assert {t.id for t in cancelled} == {"b"}
    assert graph.tasks["a"].status == TaskStatus.SUCCEEDED  # not touched


def test_is_complete_treats_blocked_and_cancelled_as_terminal():
    graph = TaskGraph([make_task(id="a"), make_task(id="b", dependencies=["a"])])
    graph.tasks["a"].status = TaskStatus.FAILED
    graph.mark_blocked_by_failed_dependencies()
    assert graph.is_complete()


def test_reset_interrupted_tasks_resets_running_ready_and_retrying():
    graph = TaskGraph([make_task(id="a"), make_task(id="b"), make_task(id="c"), make_task(id="d")])
    graph.tasks["a"].status = TaskStatus.RUNNING
    graph.tasks["b"].status = TaskStatus.READY
    graph.tasks["c"].status = TaskStatus.RETRYING
    graph.tasks["d"].status = TaskStatus.SUCCEEDED

    reset = graph.reset_interrupted_tasks()
    assert {t.id for t in reset} == {"a", "b", "c"}
    assert graph.tasks["d"].status == TaskStatus.SUCCEEDED  # never touched
