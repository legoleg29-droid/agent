import pytest

from orchestrator.core.logging_utils import EventLog
from orchestrator.state.models import ExecutionState, ExecutionStatus
from orchestrator.state.store import InMemoryStateStore, SQLiteStateStore


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryStateStore()
    return SQLiteStateStore(tmp_path / "state.db")


def test_save_and_load_round_trip(store):
    state = ExecutionState.create("goal", execution_id="exec_1")
    state.set_status(ExecutionStatus.RUNNING)
    store.save(state)

    loaded = store.load("exec_1")
    assert loaded is not None
    assert loaded.execution_id == "exec_1"
    assert loaded.status == ExecutionStatus.RUNNING
    assert loaded.user_goal == "goal"


def test_load_missing_returns_none(store):
    assert store.load("does_not_exist") is None


def test_save_overwrites_existing(store):
    state = ExecutionState.create("goal", execution_id="exec_1")
    store.save(state)
    state.set_status(ExecutionStatus.COMPLETED)
    store.save(state)

    loaded = store.load("exec_1")
    assert loaded.status == ExecutionStatus.COMPLETED


def test_list_executions(store):
    store.save(ExecutionState.create("a", execution_id="exec_a"))
    store.save(ExecutionState.create("b", execution_id="exec_b"))
    assert set(store.list_executions()) == {"exec_a", "exec_b"}


def test_delete(store):
    store.save(ExecutionState.create("goal", execution_id="exec_1"))
    store.delete("exec_1")
    assert store.load("exec_1") is None


def test_checkpoint_persists_and_emits_event(store):
    event_log = EventLog(verbose=False)
    state = ExecutionState.create("goal", execution_id="exec_1")
    store.checkpoint(state, event_log, "task start")

    assert store.load("exec_1") is not None
    checkpoint_events = [e for e in event_log.events if e.tag == "CHECKPOINT"]
    assert len(checkpoint_events) == 1
    assert checkpoint_events[0].extra["reason"] == "task start"


def test_checkpoint_without_event_log_still_persists(store):
    state = ExecutionState.create("goal", execution_id="exec_1")
    store.checkpoint(state, None, "task start")
    assert store.load("exec_1") is not None


def test_sensitive_data_is_redacted_before_persistence(store):
    state = ExecutionState.create("goal", execution_id="exec_1")
    state.metadata["api_key"] = "sk-ant-super-secret-value-12345"
    state.context["notes"] = "the token is sk-ant-super-secret-value-12345"
    store.save(state)

    loaded = store.load("exec_1")
    assert loaded.metadata["api_key"] == "<redacted>"
    assert "sk-ant-super-secret-value-12345" not in loaded.context["notes"]
