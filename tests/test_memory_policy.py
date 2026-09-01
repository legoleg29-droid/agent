from orchestrator.core.logging_utils import EventLog
from orchestrator.memory.long_term import InMemoryLongTermMemory
from orchestrator.memory.manager import MemoryManager
from orchestrator.memory.models import MemoryQuery, MemoryScope, MemoryType
from orchestrator.memory.policy import MemoryPolicy


def test_low_importance_content_is_not_persisted_long_term():
    policy = MemoryPolicy(min_importance_to_persist=0.5)
    decision = policy.evaluate(type=MemoryType.TOOL_RESULT, content="a minor tool call result")
    assert decision.should_store is False


def test_high_importance_content_is_persisted():
    policy = MemoryPolicy(min_importance_to_persist=0.5)
    decision = policy.evaluate(type=MemoryType.DECISION, content="We decided to target mid-market customers.")
    assert decision.should_store is True
    assert decision.scope == MemoryScope.PROJECT  # decisions default to project scope


def test_preference_defaults_to_user_scope():
    policy = MemoryPolicy()
    decision = policy.evaluate(type=MemoryType.PREFERENCE, content="Prefers concise reports.")
    assert decision.scope == MemoryScope.USER


def test_fact_defaults_to_execution_scope():
    policy = MemoryPolicy()
    decision = policy.evaluate(type=MemoryType.FACT, content="Competitor X uses a subscription model.")
    assert decision.scope == MemoryScope.EXECUTION


def test_sensitive_content_is_never_stored_regardless_of_importance():
    policy = MemoryPolicy(min_importance_to_persist=0.0)
    decision = policy.evaluate(
        type=MemoryType.OBSERVATION,
        content="The API key is sk-ant-abcdef1234567890abcdef",
        importance_hint=1.0,
    )
    assert decision.should_store is False
    assert decision.importance == 0.0
    assert "credential" in decision.reason


def test_sensitive_key_in_structured_content_is_never_stored():
    policy = MemoryPolicy(min_importance_to_persist=0.0)
    decision = policy.evaluate(
        type=MemoryType.TOOL_RESULT,
        content={"api_key": "some-value", "note": "irrelevant"},
        importance_hint=1.0,
    )
    assert decision.should_store is False


def test_explicit_importance_hint_overrides_default():
    policy = MemoryPolicy(min_importance_to_persist=0.5)
    decision = policy.evaluate(type=MemoryType.TOOL_RESULT, content="normally low importance", importance_hint=0.9)
    assert decision.should_store is True
    assert decision.importance == 0.9


def build_manager():
    event_log = EventLog(verbose=False)
    manager = MemoryManager("exec_1", InMemoryLongTermMemory(), policy=MemoryPolicy(), event_log=event_log)
    return manager, event_log


def test_manager_store_persists_when_policy_approves():
    manager, event_log = build_manager()
    entry = manager.store(type=MemoryType.DECISION, content="Target the mid-market segment.", source="task:t1")
    assert entry is not None
    results = manager.search(MemoryQuery(execution_id="exec_1"))
    assert len(results) == 1
    assert any(e.tag == "MEMORY_STORED" for e in event_log.events)


def test_manager_store_skips_low_importance_but_keeps_short_term():
    manager, event_log = build_manager()
    entry = manager.store(type=MemoryType.TOOL_RESULT, content="ephemeral result", source="tool:calculator")
    assert entry is not None  # kept in short-term
    assert manager.short_term.all() == [entry]
    assert manager.search(MemoryQuery(execution_id="exec_1")) == []  # never reached long-term
    assert any(e.tag == "MEMORY_SKIPPED" for e in event_log.events)


def test_manager_store_refuses_sensitive_content_entirely():
    manager, event_log = build_manager()
    entry = manager.store(
        type=MemoryType.OBSERVATION, content="token: sk-ant-abcdef1234567890abcdef", source="tool:some_api"
    )
    assert entry is None
    assert manager.short_term.all() == []  # not even kept short-term
    skipped_events = [e for e in event_log.events if e.tag == "MEMORY_SKIPPED"]
    assert skipped_events
    # never log the actual sensitive content
    assert "sk-ant-abcdef1234567890abcdef" not in skipped_events[0].message


def test_manager_never_logs_memory_content_on_store():
    manager, event_log = build_manager()
    manager.store(type=MemoryType.FACT, content="a very specific secret-shaped detail xyz123", source="task:t1")
    stored_events = [e for e in event_log.events if e.tag == "MEMORY_STORED"]
    assert stored_events
    assert "a very specific secret-shaped detail xyz123" not in stored_events[0].message
