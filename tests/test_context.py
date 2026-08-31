import time

from orchestrator.agents.base import AgentOutput
from orchestrator.core.context import ContextManager
from orchestrator.core.task_graph import Task, TaskGraph, TaskStatus


def test_upstream_outputs_only_includes_direct_dependencies():
    graph = TaskGraph([
        Task(id="t1", objective="a", capability="research"),
        Task(id="t2", objective="b", capability="research"),
        Task(id="t3", objective="c", capability="analysis", dependencies=["t1"]),
    ])
    context = ContextManager(goal="goal", started_at=time.time())
    context.record_task_output("t1", AgentOutput(success=True, content="output from t1"))
    context.record_task_output("t2", AgentOutput(success=True, content="output from t2 - should not leak"))

    upstream = context.upstream_outputs_for(graph.tasks["t3"], graph)

    assert upstream == {"t1": "output from t1"}
    assert "t2" not in upstream


def test_completed_summary_only_includes_succeeded_tasks():
    graph = TaskGraph([
        Task(id="t1", objective="a", capability="research"),
        Task(id="t2", objective="b", capability="research"),
    ])
    graph.tasks["t1"].status = TaskStatus.SUCCEEDED
    graph.tasks["t2"].status = TaskStatus.FAILED

    context = ContextManager(goal="goal", started_at=time.time())
    context.record_task_output("t1", AgentOutput(success=True, content="done"))

    summary = context.completed_summary(graph)
    assert summary == {"t1": "done"}


def test_global_context_is_separate_from_task_outputs():
    context = ContextManager(goal="goal", started_at=time.time())
    context.set_global("run_id", "abc123")
    context.record_task_output("t1", AgentOutput(success=True, content="x"))

    assert context.global_context == {"run_id": "abc123"}
    assert "run_id" not in context.task_outputs


def test_tool_results_are_structured_not_dumped_into_global_context():
    context = ContextManager(goal="goal", started_at=time.time())
    context.record_tool_result(
        tool="calculator", status="success", task_id="t1", timestamp=time.time(), result={"result": 4}
    )
    context.record_tool_result(
        tool="web_search", status="failure", task_id="t2", timestamp=time.time(), error="timed out"
    )

    assert len(context.tool_results) == 2
    success_entry, failure_entry = context.tool_results
    assert success_entry == {
        "tool": "calculator",
        "status": "success",
        "task_id": "t1",
        "timestamp": success_entry["timestamp"],
        "result": {"result": 4},
    }
    assert failure_entry["status"] == "failure"
    assert failure_entry["error"] == "timed out"
    # tool results never leak into unrelated global context
    assert "tool_results" not in context.global_context
