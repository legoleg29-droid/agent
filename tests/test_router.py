import pytest

from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.router import AgentRouter, NoAgentForCapabilityError
from orchestrator.core.task_graph import Task
from tests.doubles import StubAgent, always_succeeds


def make_task(**overrides) -> Task:
    defaults = dict(id="t1", objective="do it", capability="research", required_tools=[])
    defaults.update(overrides)
    return Task(**defaults)


def test_route_selects_capability_match():
    registry = AgentRegistry()
    registry.register(StubAgent("research_agent", ["research"], always_succeeds()))
    router = AgentRouter(registry)

    agent = router.route(make_task(capability="research"))
    assert agent.id == "research_agent"


def test_route_prefers_tool_covering_candidate():
    registry = AgentRegistry()
    no_tools = StubAgent("no_tools", ["research"], always_succeeds())
    with_tools = StubAgent("with_tools", ["research"], always_succeeds(), available_tools=["web_search"])
    registry.register(no_tools)
    registry.register(with_tools)
    router = AgentRouter(registry)

    agent = router.route(make_task(capability="research", required_tools=["web_search"]))
    assert agent.id == "with_tools"


def test_route_raises_when_no_capability_match():
    registry = AgentRegistry()
    registry.register(StubAgent("writer", ["writing"], always_succeeds()))
    router = AgentRouter(registry)

    with pytest.raises(NoAgentForCapabilityError):
        router.route(make_task(capability="research"))
