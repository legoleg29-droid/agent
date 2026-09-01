import time

from orchestrator.agents.base import AgentOutput
from orchestrator.core.task_graph import Task, TaskGraph, TaskStatus
from orchestrator.state.models import Artifact, ExecutionState, ExecutionStatus, TaskState


def test_execution_state_create_defaults():
    state = ExecutionState.create("Do the thing")
    assert state.execution_id.startswith("exec_")
    assert state.user_goal == "Do the thing"
    assert state.status == ExecutionStatus.PENDING
    assert state.completed_tasks == []
    assert state.failed_tasks == []
    assert state.active_task is None
    assert state.created_at == state.updated_at


def test_execution_state_create_accepts_explicit_id():
    state = ExecutionState.create("goal", execution_id="exec_fixed")
    assert state.execution_id == "exec_fixed"


def test_set_status_updates_timestamp_and_is_terminal():
    state = ExecutionState.create("goal")
    before = state.updated_at
    time.sleep(0.001)
    state.set_status(ExecutionStatus.RUNNING)
    assert state.status == ExecutionStatus.RUNNING
    assert state.updated_at > before
    assert not state.is_terminal()

    state.set_status(ExecutionStatus.COMPLETED)
    assert state.is_terminal()


def test_task_state_from_task_mirrors_attempt_and_fields():
    task = Task(id="t1", objective="do x", capability="research", retry_count=1, agent_id="research_agent")
    task.status = TaskStatus.RUNNING
    task.started_at = 123.0
    ts = TaskState.from_task(task)
    assert ts.task_id == "t1"
    assert ts.status == "running"
    assert ts.attempt == 2  # retry_count + 1
    assert ts.agent_id == "research_agent"
    assert ts.started_at == 123.0


def test_sync_task_state_tracks_completed_and_failed_and_active():
    state = ExecutionState.create("goal")
    graph = TaskGraph([
        Task(id="t1", objective="a", capability="research"),
        Task(id="t2", objective="b", capability="analysis"),
    ])

    graph.tasks["t1"].status = TaskStatus.RUNNING
    state.sync_task_state(graph.tasks["t1"])
    assert state.active_task == "t1"
    assert "t1" not in state.completed_tasks

    graph.tasks["t1"].status = TaskStatus.SUCCEEDED
    graph.tasks["t1"].result = AgentOutput(success=True, content="done")
    state.sync_task_state(graph.tasks["t1"])
    assert "t1" in state.completed_tasks
    assert state.active_task is None  # no longer running
    assert state.task_states["t1"].result == {"success": True, "content": "done", "data": {}, "error": None, "tool_calls": 0, "tokens_used": 0, "model": None}

    graph.tasks["t2"].status = TaskStatus.FAILED
    graph.tasks["t2"].error = "boom"
    state.sync_task_state(graph.tasks["t2"])
    assert "t2" in state.failed_tasks
    assert state.task_states["t2"].error == "boom"


def test_sync_task_state_does_not_duplicate_entries():
    state = ExecutionState.create("goal")
    task = Task(id="t1", objective="a", capability="research")
    task.status = TaskStatus.SUCCEEDED
    state.sync_task_state(task)
    state.sync_task_state(task)
    assert state.completed_tasks == ["t1"]


def test_execution_state_round_trips_through_dict():
    state = ExecutionState.create("goal", execution_id="exec_rt")
    task = Task(id="t1", objective="a", capability="research")
    task.status = TaskStatus.SUCCEEDED
    task.result = AgentOutput(success=True, content="hello")
    state.sync_task_state(task)
    state.set_status(ExecutionStatus.COMPLETED)

    restored = ExecutionState.from_dict(state.to_dict())
    assert restored.execution_id == state.execution_id
    assert restored.status == ExecutionStatus.COMPLETED
    assert restored.completed_tasks == ["t1"]
    assert restored.task_states["t1"].result["content"] == "hello"


def test_add_artifact_tracks_reference_not_content():
    state = ExecutionState.create("goal")
    artifact = Artifact(artifact_id="art_1", type="file", path="result.txt", task_id="t1", agent_id="writer_agent")
    state.add_artifact(artifact)
    assert state.artifacts == ["art_1"]
    assert state.metadata["artifacts_detail"][0]["path"] == "result.txt"
    # only a reference is stored - no "content" field on Artifact at all
    assert "content" not in state.metadata["artifacts_detail"][0]
