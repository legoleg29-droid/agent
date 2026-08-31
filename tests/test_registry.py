import pytest

from orchestrator.agents.registry import AgentNotFoundError, AgentRegistry
from tests.doubles import StubAgent, always_succeeds


def test_register_and_get():
    registry = AgentRegistry()
    agent = StubAgent("a1", ["research"], always_succeeds())
    registry.register(agent)
    assert registry.get("a1") is agent


def test_get_missing_raises():
    registry = AgentRegistry()
    with pytest.raises(AgentNotFoundError):
        registry.get("nope")


def test_find_by_capability_returns_matches_only():
    registry = AgentRegistry()
    research = StubAgent("research_agent", ["research"], always_succeeds())
    writer = StubAgent("writer_agent", ["writing"], always_succeeds())
    registry.register(research)
    registry.register(writer)

    matches = registry.find_by_capability("research")
    assert [a.id for a in matches] == ["research_agent"]


def test_find_by_capability_ranks_specialists_first():
    registry = AgentRegistry()
    generalist = StubAgent("generalist", ["research", "analysis", "writing"], always_succeeds())
    specialist = StubAgent("specialist", ["research"], always_succeeds())
    registry.register(generalist)
    registry.register(specialist)

    matches = registry.find_by_capability("research")
    assert [a.id for a in matches] == ["specialist", "generalist"]


def test_all_capabilities_is_union_and_sorted():
    registry = AgentRegistry()
    registry.register(StubAgent("a", ["writing"], always_succeeds()))
    registry.register(StubAgent("b", ["research", "analysis"], always_succeeds()))
    assert registry.all_capabilities() == ["analysis", "research", "writing"]
