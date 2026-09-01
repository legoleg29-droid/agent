import time

from orchestrator.agents.base import AgentOutput
from orchestrator.core.context import ContextManager
from orchestrator.core.context_budget import ContextBudget, ContextSection, estimate_tokens
from orchestrator.core.task_graph import Task, TaskGraph
from orchestrator.memory.long_term import InMemoryLongTermMemory
from orchestrator.memory.manager import MemoryManager
from orchestrator.memory.models import MemoryType
from orchestrator.state.models import ExecutionState
from tests.doubles import StubAgent, always_succeeds


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_required_sections_are_never_dropped_even_over_budget():
    budget = ContextBudget(max_tokens=1)
    sections = [ContextSection(name="task", text="x" * 4000, priority=0, required=True)]
    result = budget.assemble(sections)
    assert "task" in result.sections
    assert result.sections["task"] == "x" * 4000  # not summarized despite blowing the budget


def test_low_priority_sections_are_summarized_before_being_dropped():
    budget = ContextBudget(max_tokens=50, min_summary_tokens=10)
    sections = [
        ContextSection(name="task", text="short objective", priority=0, required=True),
        ContextSection(name="big", text="y" * 2000, priority=5),
    ]
    result = budget.assemble(sections)
    assert "big" in result.sections
    assert "summarized" in result.sections["big"]
    assert "big" in result.truncated


def test_sections_that_cannot_fit_even_a_summary_are_omitted_not_silently_lost():
    budget = ContextBudget(max_tokens=5, min_summary_tokens=50)
    sections = [
        ContextSection(name="task", text="x" * 20, priority=0, required=True),
        ContextSection(name="low_priority", text="y" * 2000, priority=9),
    ]
    result = budget.assemble(sections)
    assert "low_priority" not in result.sections
    assert "low_priority" in result.omitted  # explicitly recorded, not just vanished


def test_priority_ordering_fills_high_priority_first():
    budget = ContextBudget(max_tokens=12)
    sections = [
        ContextSection(name="low", text="a" * 40, priority=9),
        ContextSection(name="high", text="b" * 40, priority=1),
    ]
    result = budget.assemble(sections)
    assert "high" in result.sections
    assert result.sections["high"] == "b" * 40  # fully included, no summarization needed
    assert "low" not in result.sections or len(result.sections["low"]) < 40


def build_execution_context():
    graph = TaskGraph([
        Task(id="t1", objective="research", capability="research"),
        Task(id="t2", objective="analyze", capability="analysis", dependencies=["t1"]),
    ])
    context = ContextManager(goal="goal", started_at=time.time())
    context.record_task_output("t1", AgentOutput(success=True, content="Competitor research findings."))
    context.record_tool_result(tool="web_search", status="success", task_id="t2", timestamp=time.time(), result={"ok": True})
    agent = StubAgent("analysis_agent", ["analysis"], always_succeeds())
    return context, graph, agent


def test_build_agent_context_includes_dependency_and_tool_results():
    context, graph, agent = build_execution_context()
    agent_context = context.build_agent_context(task=graph.tasks["t2"], graph=graph, agent=agent)

    assert "t1" in agent_context.dependency_outputs
    assert "Competitor research findings." in agent_context.dependency_outputs["t1"]
    assert "tool_results" in agent_context.budget.sections


def test_build_agent_context_never_sends_the_entire_memory_database():
    context, graph, agent = build_execution_context()
    long_term = InMemoryLongTermMemory()
    memory_manager = MemoryManager("exec_1", long_term)
    for i in range(50):
        memory_manager.store(
            type=MemoryType.FACT, content=f"fact number {i}", source="test", importance_hint=0.9
        )

    agent_context = context.build_agent_context(
        task=graph.tasks["t2"], graph=graph, agent=agent, memory_manager=memory_manager
    )
    # only a bounded, budgeted slice of memory is retrieved - never all 50 entries
    assert len(agent_context.memory_snippets) < 50


def test_build_agent_context_respects_a_tight_budget_by_summarizing_not_dropping_the_task():
    context, graph, agent = build_execution_context()
    execution_state = ExecutionState.create("goal")
    agent_context = context.build_agent_context(
        task=graph.tasks["t2"], graph=graph, agent=agent, execution_state=execution_state, max_tokens=5
    )
    assert "task" in agent_context.budget.sections
    assert agent_context.budget.total_tokens >= 0


def test_constraints_are_included_and_required():
    context, graph, agent = build_execution_context()
    agent_context = context.build_agent_context(
        task=graph.tasks["t2"], graph=graph, agent=agent, constraints=["Never reveal internal pricing."], max_tokens=1
    )
    assert agent_context.constraints == ["Never reveal internal pricing."]
    assert "constraints" in agent_context.budget.sections
