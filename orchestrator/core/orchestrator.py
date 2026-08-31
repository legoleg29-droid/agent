"""Orchestrator: the execution loop.

INSPECT -> PLAN -> ROUTE -> EXECUTE -> OBSERVE -> EVALUATE
       -> REPLAN or CONTINUE -> COMPLETE

The scheduler respects task dependencies and runs independent ready tasks
concurrently via ``asyncio.gather``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from orchestrator.agents.base import AgentInput, AgentOutput, BaseAgent
from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.context import ContextManager
from orchestrator.core.evaluator import Evaluator
from orchestrator.core.logging_utils import EventLog
from orchestrator.core.planner import Planner
from orchestrator.core.retry import Action, RetryPolicy
from orchestrator.core.router import AgentRouter
from orchestrator.core.state import RunStatus, StateManager
from orchestrator.core.synthesizer import FinalResultSynthesizer
from orchestrator.core.task_graph import Task, TaskGraph, TaskStatus
from orchestrator.providers.base import LLMProvider
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.runtime import ToolRuntime


@dataclass
class OrchestrationResult:
    goal: str
    final_output: str
    graph: TaskGraph
    events: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    succeeded: bool = True


class Orchestrator:
    def __init__(
        self,
        provider: LLMProvider,
        agent_registry: AgentRegistry,
        tool_registry: ToolRegistry | None = None,
        *,
        max_retries_per_task: int = 2,
        max_replans: int = 2,
        verbose_logging: bool = True,
    ) -> None:
        self.provider = provider
        self.agent_registry = agent_registry
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_retries_per_task = max_retries_per_task

        self.event_log = EventLog(verbose=verbose_logging)
        self.tool_runtime = ToolRuntime(self.tool_registry, self.event_log)
        self.planner = Planner(provider, self.event_log)
        self.router = AgentRouter(agent_registry, self.event_log)
        self.evaluator = Evaluator()
        self.retry_policy = RetryPolicy()
        self.synthesizer = FinalResultSynthesizer(provider, self.event_log)
        self.state = StateManager(max_replans=max_replans)

    async def run(self, goal: str) -> OrchestrationResult:
        self.event_log.emit("ORCHESTRATOR", f"Received goal: {goal!r}")
        context = ContextManager(goal=goal, started_at=time.time())

        # PLAN
        self.state.transition(RunStatus.PLANNING)
        capabilities = self.agent_registry.all_capabilities()
        tools = [t.id for t in self.tool_registry.list_tools()]
        graph = await self.planner.plan(goal, capabilities=capabilities, tools=tools)
        for task in graph.tasks.values():
            task.max_retries = self.max_retries_per_task

        # EXECUTE loop (ROUTE -> EXECUTE -> OBSERVE -> EVALUATE -> REPLAN/CONTINUE)
        self.state.transition(RunStatus.EXECUTING)
        aborted = False
        while not graph.is_complete():
            ready = graph.get_ready_tasks()
            if not ready:
                if graph.has_pending_work():
                    # pending tasks exist but none are ready -> unresolved deps
                    skipped = graph.mark_unreachable_as_skipped()
                    if skipped:
                        continue
                break

            for task in ready:
                task.status = TaskStatus.RUNNING

            results = await asyncio.gather(*(self._run_task(task, graph, context) for task in ready))

            for task, action in zip(ready, results):
                if action == Action.REPLAN:
                    replanned = await self._replan(goal, graph, context, task)
                    if not replanned:
                        aborted = True
                elif action == Action.ABORT:
                    aborted = True

            if aborted:
                graph.mark_unreachable_as_skipped()
                break

        succeeded = not graph.failed_tasks() and not aborted
        final_output = await self.synthesizer.synthesize(goal, graph, context)
        self.state.transition(RunStatus.COMPLETED if succeeded else RunStatus.FAILED)

        return OrchestrationResult(
            goal=goal,
            final_output=final_output,
            graph=graph,
            events=[e.to_dict() for e in self.event_log.events],
            tool_results=context.tool_results,
            succeeded=succeeded,
        )

    async def _run_task(self, task: Task, graph: TaskGraph, context: ContextManager) -> Action:
        agent = self.router.route(task)
        task.agent_id = agent.id
        agent_input = AgentInput(
            objective=task.objective,
            expected_output=task.expected_output,
            task_context={"task_id": task.id},
            upstream_outputs=context.upstream_outputs_for(task, graph),
        )

        started = time.perf_counter()
        self.event_log.emit("TASK", f"Starting task '{task.id}'", task_id=task.id, agent_id=agent.id, status="running")

        availability_error = self._validate_tool_requirements(task, agent)
        events_before = len(self.event_log.events)
        if availability_error:
            self.event_log.emit(
                "TASK",
                f"Task '{task.id}' failed pre-execution validation: {availability_error}",
                task_id=task.id,
                agent_id=agent.id,
                status="failed",
                error=availability_error,
            )
            output = AgentOutput(success=False, error=availability_error)
        else:
            try:
                output = await agent.execute(agent_input)
            except Exception as exc:  # noqa: BLE001 - agent crashes are a failure mode to evaluate, not a crash
                output = AgentOutput(success=False, error=f"{type(exc).__name__}: {exc}")
        duration_ms = (time.perf_counter() - started) * 1000

        task.result = output
        task.error = output.error
        context.record_task_output(task.id, output)
        self._record_tool_results(task.id, events_before, context)

        self.event_log.emit(
            "AGENT",
            f"Agent '{agent.id}' finished task '{task.id}'",
            task_id=task.id,
            agent_id=agent.id,
            status="success" if output.success else "failure",
            duration_ms=round(duration_ms, 2),
            tokens_used=output.tokens_used,
            tool_calls=output.tool_calls,
            model=output.model,
        )

        evaluation = self.evaluator.evaluate(task, output)
        self.event_log.emit(
            "EVALUATOR",
            f"Task '{task.id}' evaluated as {evaluation.verdict.value}: {evaluation.reason}",
            task_id=task.id,
            status=evaluation.verdict.value,
        )

        action = self.retry_policy.decide(task, evaluation, self.state)

        if action == Action.CONTINUE:
            task.status = TaskStatus.SUCCEEDED
        elif action == Action.RETRY:
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            self.event_log.emit(
                "RETRY",
                f"Retrying task '{task.id}' (attempt {task.retry_count}/{task.max_retries})",
                task_id=task.id,
                retry_count=task.retry_count,
            )
            action = await self._run_task(task, graph, context)
        elif action == Action.REPLAN:
            task.status = TaskStatus.FAILED
        else:  # ABORT
            task.status = TaskStatus.FAILED
            self.event_log.emit("TASK", f"Task '{task.id}' aborted (non-recoverable)", task_id=task.id, status="aborted")

        return action

    def _validate_tool_requirements(self, task: Task, agent: BaseAgent) -> str | None:
        """Fail fast, before any LLM call, if the required tools aren't
        actually usable - unregistered, not declared on the routed agent,
        or the agent lacks a required permission. Mirrors what ToolRuntime
        would reject anyway, but avoids wasting a model call on a task
        that's guaranteed to fail at the tool-call step.
        """
        for tool_id in task.required_tools:
            if not self.tool_registry.is_available(tool_id):
                return f"Required tool '{tool_id}' is not registered in the ToolRegistry"
            if tool_id not in agent.available_tools:
                return f"Agent '{agent.id}' does not declare required tool '{tool_id}' in available_tools"
            tool = self.tool_registry.get(tool_id)
            missing = [p for p in tool.permissions if p not in agent.permissions]
            if missing:
                return f"Agent '{agent.id}' lacks permission(s) {missing} required by tool '{tool_id}'"
        return None

    def _record_tool_results(self, task_id: str, events_since_index: int, context: ContextManager) -> None:
        for event in self.event_log.events[events_since_index:]:
            if event.tag not in ("TOOL_RESULT", "TOOL_ERROR") or event.task_id != task_id:
                continue
            context.record_tool_result(
                tool=event.tool_id or "unknown",
                status="success" if event.tag == "TOOL_RESULT" and event.status == "success" else "failure",
                task_id=event.task_id,
                timestamp=event.timestamp,
                result=event.extra.get("output") if event.status == "success" else None,
                error=event.error,
            )

    async def _replan(self, goal: str, graph: TaskGraph, context: ContextManager, failed_task: Task) -> bool:
        if not self.state.can_replan():
            return False
        self.state.transition(RunStatus.REPLANNING)
        self.state.record_replan()

        capabilities = self.agent_registry.all_capabilities()
        tools = [t.id for t in self.tool_registry.list_tools()]
        completed_summary = context.completed_summary(graph)

        new_tasks = await self.planner.replan(
            goal,
            capabilities=capabilities,
            tools=tools,
            completed_summary=completed_summary,
            failure_reason=f"Task '{failed_task.id}' failed: {failed_task.error}",
        )
        for task in new_tasks:
            task.max_retries = self.max_retries_per_task
            graph.add_task(task)
        graph.finalize()
        self.state.transition(RunStatus.EXECUTING)
        return True
