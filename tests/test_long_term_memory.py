import pytest

from orchestrator.memory.long_term import InMemoryLongTermMemory, ScopeViolationError, SQLiteLongTermMemory
from orchestrator.memory.models import MemoryEntry, MemoryQuery, MemoryScope, MemoryType


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryLongTermMemory()
    return SQLiteLongTermMemory(tmp_path / "memory.db")


def entry(**overrides):
    defaults = dict(
        type=MemoryType.FACT,
        content="Competitor X uses a subscription model.",
        source="task:t1",
        scope=MemoryScope.EXECUTION,
        execution_id="exec_1",
        importance=0.7,
    )
    defaults.update(overrides)
    return MemoryEntry.create(**defaults)


def test_store_and_get(store):
    e = entry()
    store.store(e)
    fetched = store.get(e.memory_id)
    assert fetched is not None
    assert fetched.content == "Competitor X uses a subscription model."


def test_get_missing_returns_none(store):
    assert store.get("mem_missing") is None


def test_delete(store):
    e = entry()
    store.store(e)
    store.delete(e.memory_id)
    assert store.get(e.memory_id) is None


def test_search_requires_a_scope_filter(store):
    with pytest.raises(ScopeViolationError):
        store.search(MemoryQuery())


def test_search_filters_by_execution_id(store):
    store.store(entry(execution_id="exec_1"))
    store.store(entry(execution_id="exec_2"))
    results = store.search(MemoryQuery(execution_id="exec_1"))
    assert all(r.execution_id == "exec_1" for r in results)
    assert len(results) == 1


def test_search_filters_by_type_and_importance(store):
    store.store(entry(type=MemoryType.FACT, importance=0.8))
    store.store(entry(type=MemoryType.OBSERVATION, importance=0.9))
    results = store.search(MemoryQuery(execution_id="exec_1", type=MemoryType.FACT))
    assert len(results) == 1 and results[0].type == MemoryType.FACT

    results = store.search(MemoryQuery(execution_id="exec_1", min_importance=0.85))
    assert len(results) == 1 and results[0].type == MemoryType.OBSERVATION


def test_search_orders_by_importance_desc(store):
    store.store(entry(importance=0.3))
    store.store(entry(importance=0.9))
    store.store(entry(importance=0.6))
    results = store.search(MemoryQuery(execution_id="exec_1", limit=10))
    assert [r.importance for r in results] == [0.9, 0.6, 0.3]


def test_cross_execution_isolation(store):
    store.store(entry(execution_id="exec_a", content="a's secret plan"))
    store.store(entry(execution_id="exec_b", content="b's secret plan"))

    a_results = store.search(MemoryQuery(execution_id="exec_a"))
    assert len(a_results) == 1
    assert a_results[0].content == "a's secret plan"


def test_cross_user_isolation(store):
    store.store(entry(execution_id=None, user_id="user_a", scope=MemoryScope.USER, content="user a's preference"))
    store.store(entry(execution_id=None, user_id="user_b", scope=MemoryScope.USER, content="user b's preference"))

    a_results = store.search(MemoryQuery(user_id="user_a"))
    assert len(a_results) == 1
    assert a_results[0].content == "user a's preference"

    # A query scoped to user_a must never return user_b's memory even if
    # other filters are absent.
    assert all(r.user_id == "user_a" for r in a_results)
