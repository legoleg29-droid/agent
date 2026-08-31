"""Lightweight test doubles shared across the test suite."""

from __future__ import annotations

from collections.abc import Callable

from orchestrator.agents.base import AgentInput, AgentOutput, BaseAgent


class StubAgent(BaseAgent):
    """A BaseAgent whose output is driven by a callable, not an LLM.

    ``behavior(call_index, agent_input) -> AgentOutput`` lets tests script
    per-call results (e.g. fail twice then succeed) to exercise retry and
    replan paths deterministically.
    """

    def __init__(
        self,
        agent_id: str,
        capabilities: list[str],
        behavior: Callable[[int, AgentInput], AgentOutput],
        *,
        available_tools: list[str] | None = None,
    ) -> None:
        self.id = agent_id
        self.name = agent_id
        self.description = f"Stub agent for tests ({agent_id})"
        self.capabilities = capabilities
        self.available_tools = available_tools or []
        self.system_instructions = "stub"
        super().__init__(provider=None, tool_runtime=None)  # type: ignore[arg-type]
        self._behavior = behavior
        self.call_count = 0
        self.received_inputs: list[AgentInput] = []

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        self.received_inputs.append(agent_input)
        result = self._behavior(self.call_count, agent_input)
        self.call_count += 1
        return result


def always_succeeds(content: str = "stub output that is definitely long enough") -> Callable:
    def _behavior(_call_index: int, _agent_input: AgentInput) -> AgentOutput:
        return AgentOutput(success=True, content=content)

    return _behavior


def fails_n_times_then_succeeds(n: int, success_content: str = "recovered output after retries") -> Callable:
    def _behavior(call_index: int, _agent_input: AgentInput) -> AgentOutput:
        if call_index < n:
            return AgentOutput(success=False, error=f"transient failure #{call_index}")
        return AgentOutput(success=True, content=success_content)

    return _behavior


def always_fails(error: str = "permanent failure") -> Callable:
    def _behavior(_call_index: int, _agent_input: AgentInput) -> AgentOutput:
        return AgentOutput(success=False, error=error)

    return _behavior
