import pytest

from orchestrator.memory.models import MemoryEntry, MemoryType
from orchestrator.memory.short_term import ShortTermMemory


def make_entry(execution_id="exec_1", type=MemoryType.OBSERVATION, importance=0.5, task_id=None):
    return MemoryEntry.create(type=type, content="something happened", source="test", execution_id=execution_id, importance=importance, task_id=task_id)


def test_add_and_recent():
    stm = ShortTermMemory("exec_1")
    stm.add(make_entry())
    stm.add(make_entry(type=MemoryType.FACT))
    recent = stm.recent(limit=10)
    assert len(recent) == 2
    assert recent[0].type == MemoryType.FACT  # most recent first


def test_recent_filters_by_type():
    stm = ShortTermMemory("exec_1")
    stm.add(make_entry(type=MemoryType.FACT))
    stm.add(make_entry(type=MemoryType.OBSERVATION))
    facts = stm.recent(type=MemoryType.FACT)
    assert len(facts) == 1 and facts[0].type == MemoryType.FACT


def test_for_task_filters_by_task_id():
    stm = ShortTermMemory("exec_1")
    stm.add(make_entry(task_id="t1"))
    stm.add(make_entry(task_id="t2"))
    assert len(stm.for_task("t1")) == 1


def test_rejects_entry_from_a_different_execution():
    stm = ShortTermMemory("exec_1")
    with pytest.raises(ValueError):
        stm.add(make_entry(execution_id="exec_other"))


def test_eviction_keeps_high_importance_and_recency():
    stm = ShortTermMemory("exec_1", max_entries=3)
    stm.add(make_entry(importance=0.9))  # should survive - high importance
    stm.add(make_entry(importance=0.1))
    stm.add(make_entry(importance=0.1))
    stm.add(make_entry(importance=0.1))  # pushes one of the low-importance ones out

    all_entries = stm.all()
    assert len(all_entries) == 3
    assert any(e.importance == 0.9 for e in all_entries)


def test_clear():
    stm = ShortTermMemory("exec_1")
    stm.add(make_entry())
    stm.clear()
    assert stm.all() == []
