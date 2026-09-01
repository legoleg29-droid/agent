"""End-to-end test of Phase 3's memory/state layer, matching the spec
scenario:

    User: "Research three competitors and summarize the findings."
    Execution created -> Plan created -> Task state persisted ->
    Research agent executes -> Results stored -> Relevant memory created ->
    Execution checkpointed -> Final result generated -> Execution marked
    completed.

Then: execution interrupted after Task 1, runtime restarted, and
resume_execution(execution_id) continues without re-running Task 1.

Uses the real ResearchAgent/WriterAgent and real tools against a scripted
MockProvider, the same pattern as tests/test_e2e.py from Phase 2.
"""

import json

import pytest

from orchestrator.agents.registry import AgentRegistry
from orchestrator.agents.research_agent import ResearchAgent
from orchestrator.agents.writer_agent import WriterAgent
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.core.task_graph import TaskStatus
from orchestrator.memory.models import MemoryQuery, MemoryType
from orchestrator.providers.mock_provider import MockProvider, ScriptedToolUse
from orchestrator.state.models import ExecutionStatus
from orchestrator.state.store import InMemoryStateStore
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.web_search_tool import WebSearchTool

PLAN = {
    "tasks": [
        {
            "id": "research_competitors",
            "objective": "Research three competitors in the market",
            "capability": "research",
            "dependencies": [],
            "required_tools": ["web_search"],
            "expected_output": "A summary of three competitors",
        },
        {
            "id": "summarize_findings",
            "objective": "Summarize the research findings",
            "capability": "writing",
            "dependencies": ["research_competitors"],
            "required_tools": [],
            "expected_output": "A concise summary",
        },
    ]
}


class SimulatedCrash(BaseException):
    """See tests/test_checkpoint_resume.py - deliberately not an Exception
    subclass so it escapes run() the way a killed process would."""


def build_registry(tool_registry, provider, orchestrator):
    registry = AgentRegistry()
    registry.register(ResearchAgent(provider, orchestrator.tool_runtime))
    registry.register(WriterAgent(provider, orchestrator.tool_runtime))
    return registry


def base_responder(system, messages, tools):
    if system.startswith("You are the planning module"):
        return json.dumps(PLAN)
    if system.startswith("You are a research analyst"):
        if len(messages) == 1:
            return ScriptedToolUse(name="web_search", arguments={"query": "top three competitors"})
        return (
            "Research findings: Acme, Globex, and Initech are the three main "
            "competitors, each with a distinct go-to-market strategy."
        )
    if system.startswith("You are a professional writer"):
        return "Summary: Acme, Globex, and Initech lead the market with distinct strategies worth tracking."
    if system.startswith("You are the final synthesis stage"):
        return "Final summary of the three competitors and their strategies."
    raise AssertionError(f"unexpected system prompt: {system[:60]!r}")


@pytest.mark.asyncio
async def test_research_and_summarize_full_lifecycle():
    tool_registry = ToolRegistry()
    tool_registry.register(WebSearchTool())
    provider = MockProvider(responder=base_responder)
    store = InMemoryStateStore()

    # Registry needs the orchestrator's tool_runtime, so build the
    # orchestrator first with an empty registry and register agents after -
    # same pattern used throughout this codebase.
    agent_registry = AgentRegistry()
    orchestrator = Orchestrator(provider, agent_registry, tool_registry, state_store=store, verbose_logging=False)
    agent_registry.register(ResearchAgent(provider, orchestrator.tool_runtime))
    agent_registry.register(WriterAgent(provider, orchestrator.tool_runtime))

    result = await orchestrator.run("Research three competitors and summarize the findings.")

    # Execution created, plan created, task state persisted, execution completed.
    assert result.execution_id is not None
    assert result.execution_state is not None
    assert result.execution_state.status == ExecutionStatus.COMPLETED
    assert result.execution_state.current_plan is not None
    assert set(result.execution_state.completed_tasks) == {"research_competitors", "summarize_findings"}
    assert result.execution_state.task_states["research_competitors"].status == "succeeded"
    assert result.execution_state.task_states["research_competitors"].result is not None

    # Research agent executed and produced results that were stored.
    assert result.graph.tasks["research_competitors"].status == TaskStatus.SUCCEEDED
    assert "Acme" in result.graph.tasks["research_competitors"].result.content

    # Relevant memory was created for at least the successful tasks.
    memories = orchestrator.long_term_memory.search(
        MemoryQuery(execution_id=result.execution_id, type=MemoryType.TASK_RESULT)
    )
    assert len(memories) >= 1
    assert any("research_competitors" in (m.source or "") for m in memories)

    # Execution was checkpointed at every required lifecycle transition.
    checkpoint_reasons = {e["reason"] for e in result.events if e["tag"] == "CHECKPOINT"}
    assert {"plan creation", "plan created", "task start", "task completion", "execution completion"}.issubset(
        checkpoint_reasons
    )

    # Final result generated.
    assert result.final_output.strip() != ""
    assert result.succeeded

    # The persisted record on disk/in the store agrees with the in-memory result.
    persisted = store.load(result.execution_id)
    assert persisted.status == ExecutionStatus.COMPLETED
    assert set(persisted.completed_tasks) == {"research_competitors", "summarize_findings"}


@pytest.mark.asyncio
async def test_interrupted_after_research_task_resumes_without_rerunning_it():
    tool_registry = ToolRegistry()
    tool_registry.register(WebSearchTool())
    store = InMemoryStateStore()

    research_call_count = {"n": 0}

    def crashing_responder(system, messages, tools):
        if system.startswith("You are a professional writer"):
            raise SimulatedCrash("simulated crash before the writer task runs")
        if system.startswith("You are a research analyst") and len(messages) == 1:
            research_call_count["n"] += 1
        return base_responder(system, messages, tools)

    provider = MockProvider(responder=crashing_responder)
    agent_registry = AgentRegistry()
    orchestrator_1 = Orchestrator(provider, agent_registry, tool_registry, state_store=store, verbose_logging=False)
    agent_registry.register(ResearchAgent(provider, orchestrator_1.tool_runtime))
    agent_registry.register(WriterAgent(provider, orchestrator_1.tool_runtime))

    with pytest.raises(SimulatedCrash):
        await orchestrator_1.run(
            "Research three competitors and summarize the findings.", execution_id="exec_phase3_e2e"
        )

    persisted = store.load("exec_phase3_e2e")
    assert "research_competitors" in persisted.completed_tasks
    assert "summarize_findings" not in persisted.completed_tasks
    assert research_call_count["n"] == 1

    # Restart: a fresh Orchestrator/provider/agents, same persisted store.
    # The crashing responder no longer crashes on this "new process".
    provider_2 = MockProvider(responder=base_responder)
    agent_registry_2 = AgentRegistry()
    orchestrator_2 = Orchestrator(provider_2, agent_registry_2, tool_registry, state_store=store, verbose_logging=False)
    agent_registry_2.register(ResearchAgent(provider_2, orchestrator_2.tool_runtime))
    agent_registry_2.register(WriterAgent(provider_2, orchestrator_2.tool_runtime))

    result = await orchestrator_2.resume_execution("exec_phase3_e2e")

    assert result.succeeded
    assert research_call_count["n"] == 1  # research task was NOT re-executed
    assert result.graph.tasks["research_competitors"].status == TaskStatus.SUCCEEDED
    assert result.graph.tasks["summarize_findings"].status == TaskStatus.SUCCEEDED
    assert result.execution_state.status == ExecutionStatus.COMPLETED
