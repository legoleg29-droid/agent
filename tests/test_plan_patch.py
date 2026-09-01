from orchestrator.core.plan_patch import PatchOpType, PlanPatchOp, apply_plan_patch
from orchestrator.core.task_graph import Task, TaskGraph, TaskStatus


def make_graph() -> TaskGraph:
    graph = TaskGraph()
    graph.add_task(Task(id="a", objective="do a", capability="research", status=TaskStatus.SUCCEEDED))
    graph.add_task(Task(id="b", objective="do b", capability="research", dependencies=["a"], status=TaskStatus.FAILED))
    graph.add_task(Task(id="c", objective="do c", capability="research", dependencies=["b"], status=TaskStatus.BLOCKED))
    return graph


def test_add_task_op():
    graph = TaskGraph()
    graph.add_task(Task(id="a", objective="a", capability="research"))
    op = PlanPatchOp(op=PatchOpType.ADD_TASK, task={"id": "d", "objective": "new", "capability": "research", "dependencies": ["a"]})
    errors = apply_plan_patch(graph, [op])
    assert errors == []
    assert "d" in graph.tasks
    assert graph.tasks["d"].dependencies == ["a"]


def test_remove_task_op_refuses_when_dependents_exist():
    graph = make_graph()
    op = PlanPatchOp(op=PatchOpType.REMOVE_TASK, task_id="a")
    errors = apply_plan_patch(graph, [op])
    assert errors  # 'b' still depends on 'a' - the op should be reported as failed, not crash
    assert "a" in graph.tasks


def test_replace_task_rewires_dependents_and_unblocks_them():
    graph = make_graph()
    op = PlanPatchOp(
        op=PatchOpType.REPLACE_TASK,
        task_id="b",
        task={"id": "b2", "objective": "do b differently", "capability": "research", "dependencies": ["a"]},
    )
    errors = apply_plan_patch(graph, [op])
    assert errors == []
    assert "b2" in graph.tasks
    assert graph.tasks["c"].dependencies == ["b2"]
    assert graph.tasks["c"].status == TaskStatus.PENDING  # was BLOCKED, now given a real chance to run
    assert graph.tasks["b"].status == TaskStatus.FAILED  # the old failed task itself is left alone, not deleted


def test_replace_task_marks_the_old_task_as_superseded_not_still_failing():
    graph = make_graph()
    op = PlanPatchOp(
        op=PatchOpType.REPLACE_TASK,
        task_id="b",
        task={"id": "b2", "objective": "do b differently", "capability": "research", "dependencies": ["a"]},
    )
    apply_plan_patch(graph, [op])
    assert graph.tasks["b"].metadata["superseded_by"] == "b2"
    # a successful replacement means the old failure no longer counts
    # against the execution's overall success.
    assert "b" not in [t.id for t in graph.failed_tasks()]


def test_replace_task_auto_suffixes_id_collision():
    graph = make_graph()
    op = PlanPatchOp(
        op=PatchOpType.REPLACE_TASK,
        task_id="b",
        task={"id": "a", "objective": "collides with existing id", "capability": "research", "dependencies": []},
    )
    errors = apply_plan_patch(graph, [op])
    assert errors == []
    new_ids = set(graph.tasks) - {"a", "b", "c"}
    assert len(new_ids) == 1
    assert next(iter(new_ids)) != "a"


def test_modify_task_only_touches_allowed_fields():
    graph = TaskGraph()
    graph.add_task(Task(id="a", objective="orig", capability="research"))
    op = PlanPatchOp(op=PatchOpType.MODIFY_TASK, task_id="a", changes={"objective": "revised"})
    errors = apply_plan_patch(graph, [op])
    assert errors == []
    assert graph.tasks["a"].objective == "revised"


def test_modify_task_rejects_non_plan_owned_field():
    graph = TaskGraph()
    graph.add_task(Task(id="a", objective="orig", capability="research"))
    op = PlanPatchOp(op=PatchOpType.MODIFY_TASK, task_id="a", changes={"status": "succeeded"})
    errors = apply_plan_patch(graph, [op])
    assert errors  # rejected, not silently applied
    assert graph.tasks["a"].status == TaskStatus.PENDING


def test_add_and_remove_dependency_ops():
    graph = TaskGraph()
    graph.add_task(Task(id="a", objective="a", capability="research"))
    graph.add_task(Task(id="b", objective="b", capability="research"))
    errors = apply_plan_patch(graph, [PlanPatchOp(op=PatchOpType.ADD_DEPENDENCY, task_id="b", dependency_id="a")])
    assert errors == []
    assert graph.tasks["b"].dependencies == ["a"]

    errors = apply_plan_patch(graph, [PlanPatchOp(op=PatchOpType.REMOVE_DEPENDENCY, task_id="b", dependency_id="a")])
    assert errors == []
    assert graph.tasks["b"].dependencies == []


def test_one_bad_op_does_not_prevent_the_rest_from_applying():
    graph = make_graph()
    ops = [
        PlanPatchOp(op=PatchOpType.REMOVE_TASK, task_id="does_not_exist"),
        PlanPatchOp(op=PatchOpType.ADD_TASK, task={"id": "d", "objective": "new", "capability": "research", "dependencies": []}),
    ]
    errors = apply_plan_patch(graph, ops)
    assert len(errors) == 0 or "d" in graph.tasks
    assert "d" in graph.tasks
