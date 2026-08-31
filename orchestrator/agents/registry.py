"""Capability-based agent registry.

The orchestrator never hardcodes "if task == research: use ResearchAgent".
Instead each agent declares ``capabilities`` and the registry answers
"which agents can handle capability X", letting the router pick.
"""

from __future__ import annotations

from orchestrator.agents.base import BaseAgent


class AgentNotFoundError(KeyError):
    pass


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.id] = agent

    def get(self, agent_id: str) -> BaseAgent:
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise AgentNotFoundError(f"No agent registered with id '{agent_id}'") from exc

    def all_agents(self) -> list[BaseAgent]:
        return list(self._agents.values())

    def find_by_capability(self, capability: str) -> list[BaseAgent]:
        """Return agents that declare ``capability``, ranked by specificity.

        Specificity = fewer total declared capabilities (a specialist beats
        a generalist offering the same capability) as a simple, explicit
        tie-break rule.
        """
        matches = [a for a in self._agents.values() if capability in a.capabilities]
        matches.sort(key=lambda a: len(a.capabilities))
        return matches

    def all_capabilities(self) -> list[str]:
        caps: set[str] = set()
        for agent in self._agents.values():
            caps.update(agent.capabilities)
        return sorted(caps)

    def describe_all(self) -> list[dict]:
        return [a.describe() for a in self._agents.values()]
