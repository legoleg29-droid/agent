"""Orchestrator: the execution loop.

USER -> ORCHESTRATOR -> PLANNER -> TASK GRAPH -> SCHEDULER -> AGENT RUNTIME
     -> CONTEXT MANAGER -> MEMORY / STATE -> TOOL RUNTIME -> EVALUATOR
     -> CHECKPOINT -> REPLAN / COMPLETE

The scheduler respects task dependencies and runs independent ready tasks
concurrently via ``asyncio.gather``. Every important transition (plan
creation, task start, task completion, task failure, replan, execution
completion) is checkpointed through a pluggable ``StateStore`` so a run can
be reconstructed and resumed after a crash without re-executing completed
work - see ``resume_execution``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from orchestrator.agents.base import AgentInput, AgentOutput, BaseAgent
from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.context import ContextManager
from orchestrator.core.evaluator import Evaluator, Verdict
from orchestrator.core.logging_utils import EventLog
from orchestrator.core.planner import Planner
from orchestrator.core.retry import Action, RetryPolicy
from orchestrator.core.router import AgentRouter
from orchestrator.core.state import RunStatus, StateManager
from orchestrator.core.synthesizer import FinalResultSynthesizer
from orchestrator.core.task_graph import Task, TaskGraph, TaskStatus
from orchestrator.memory.long_term import InMemoryLongTermMemory, LongTermMemory
from orchestrator.memory.manager import MemoryManager
from orchestrator.memory.models import MemoryType
from orchestrator.memory.policy import MemoryPolicy
from orchestrator.providers.base import LLMProvider
from orchestrator.state.models import Artifact, ExecutionState, ExecutionStatus, AgentState
from orchestrator.state.store import ExecutionNotFoundError, InMemoryStateStore, StateStore
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.runtime import ToolRuntime

DEFAULT_CONTEXT_MAX_TOKENS = 3000


@dataclass
class OrchestrationResult:
    goal: str
    final_output: str
    graph: TaskGraph
    events: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    succeeded: bool = True
    execution_id: str | None = None
    execution_state: ExecutionState | None = None


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
        state_store: StateStore | None = None,
        long_term_memory: LongTermMemory | None = None,
        memory_policy: MemoryPolicy | None = None,
        context_max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
        session_id: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.agent_registry = agent_registry
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_retries_per_task = max_retries_per_task
        self.max_replans = max_replans
        self.context_max_tokens = context_max_tokens

        self.event_log = EventLog(verbose=verbose_logging)
        self.tool_runtime = ToolRuntime(self.tool_registry, self.event_log)
        self.planner = Planner(provider, self.event_log)
        self.router = AgentRouter(agent_registry, self.event_log)
        self.evaluator = Evaluator()
        self.retry_policy = RetryPolicy()
        self.synthesizer = FinalResultSynthesizer(provider, self.event_log)

        # Persistence: pluggable, local-development-friendly defaults.
        self.state_store: StateStore = state_store or InMemoryStateStore()
        self.long_term_memory: LongTermMemory = long_term_memory or InMemoryLongTermMemory()
        self.memory_policy = memory_policy or MemoryPolicy()
        self.session_id = session_id
        self.project_id = project_id
        self.user_id = user_id

        # Phase 1/2 run-status/replan-budget tracker, recreated per run/resume
        # so replan budget is scoped to one execution, not shared across
        # multiple run() calls on the same Orchestrator instance.
        self.state = StateManager(max_replans=max_replans)

    # -- Public entry points ---------------------------------------------

    async def run(self, goal: str, *, execution_id: str | None = None) -> OrchestrationResult:
        self.event_log.emit("ORCHESTRATOR", f"Received goal: {goal!r}")
        self.state = StateManager(max_replans=self.max_replans)

        execution_state = ExecutionState.create(goal, execution_id=execution_id, max_replans=self.max_replans)
        self.event_log.emit(
            "STATE_CREATED",
            f"Created execution '{execution_state.execution_id}'",
            extra={"execution_id": execution_state.execution_id, "goal": goal},
        )

        context = ContextManager(goal=goal, started_at=execution_state.created_at)
        memory_manager = self._build_memory_manager(execution_state.execution_id)

        # PLAN
        execution_state.set_status(ExecutionStatus.PLANNING)
        self.state.transition(RunStatus.PLANNING)
        self.state_store.checkpoint(execution_state, self.event_log, "plan creation")

        capabilities = self.agent_registry.all_capabilities()
        tools = [t.id for t in self.tool_registry.list_tools()]
        graph = await self.planner.plan(goal, capabilities=capabilities, tools=tools)
        for task in graph.tasks.values():
            task.max_retries = self.max_retries_per_task
        execution_state.current_plan = graph.to_dict()
        self.state_store.checkpoint(execution_state, self.event_log, "plan created")

        execution_state.set_status(ExecutionStatus.RUNNING)
        self.state.transition(RunStatus.EXECUTING)

        return await self._execute_loop(goal, graph, context, execution_state, memory_manager)

    async def resume_execution(self, execution_id: str) -> OrchestrationResult:
        """Load a persisted execution and continue it from where it left
        off. A task that was RUNNING when the process stopped is reset to
        PENDING and retried; anything already SUCCEEDED/FAILED/SKIPPED is
        never re-executed."""
        state = self.state_store.load(execution_id)
        if state is None:
            raise ExecutionNotFoundError(f"No persisted execution found for id '{execution_id}'")
        if state.current_plan is None:
            raise ValueError(f"Execution '{execution_id}' has no persisted plan to resume from")

        self.state = StateManager(max_replans=state.max_replans)
        self.state.replans_used = state.replans_used

        graph = TaskGraph.from_dict(state.current_plan)
        reset_tasks = graph.reset_interrupted_tasks()

        self.event_log.emit(
            "RESUME",
            f"Resuming execution '{execution_id}' from status={state.status.value}, "
            f"{len(state.completed_tasks)} task(s) already completed",
            extra={
                "execution_id": execution_id,
                "completed_tasks": list(state.completed_tasks),
                "reset_tasks": [t.id for t in reset_tasks],
            },
        )

        context = ContextManager(goal=state.user_goal, started_at=state.created_at)
        context.global_context.update(state.context)
        for task in graph.tasks.values():
            if task.status == TaskStatus.SUCCEEDED and task.result is not None:
                # Rehydrate prior outputs so downstream tasks see them
                # without re-running the tasks that produced them.
                context.record_task_output(task.id, task.result)

        memory_manager = self._build_memory_manager(execution_id)

        state.set_status(ExecutionStatus.RUNNING)
        self.state.transition(RunStatus.EXECUTING)
        self.state_store.checkpoint(state, self.event_log, "resume")

        return await self._execute_loop(state.user_goal, graph, context, state, memory_manager)

    def cancel_execution(self, execution_id: str) -> None:
        state = self.state_store.load(execution_id)
        if state is None:
            raise ExecutionNotFoundError(f"No persisted execution found for id '{execution_id}'")
        state.set_status(ExecutionStatus.CANCELLED)
        self.state_store.checkpoint(state, self.event_log, "cancelled")

    # -- Execution loop -----------------------------------------------------

    async def _execute_loop(
        self,
        goal: str,
        graph: TaskGraph,
        context: ContextManager,
        execution_state: ExecutionState,
        memory_manager: MemoryManager,
    ) -> OrchestrationResult:
        aborted = False
        while not graph.is_complete():
            ready = graph.get_ready_tasks()
            if not ready:
                if graph.has_pending_work():
                    skipped = graph.mark_unreachable_as_skipped()
                    if skipped:
                        for task in skipped:
                            execution_state.sync_task_state(task)
                        continue
                break

            for task in ready:
                task.status = TaskStatus.RUNNING

            results = await asyncio.gather(
                *(self._run_task(task, graph, context, execution_state, memory_manager) for task in ready)
            )

            for task, action in zip(ready, results):
                if action == Action.REPLAN:
                    replanned = await self._replan(goal, graph, context, task, execution_state)
                    if not replanned:
                        aborted = True
                elif action == Action.ABORT:
                    aborted = True

            if aborted:
                for task in graph.mark_unreachable_as_skipped():
                    execution_state.sync_task_state(task)
                break

        succeeded = not graph.failed_tasks() and not aborted
        final_output = await self.synthesizer.synthesize(goal, graph, context)

        execution_state.current_plan = graph.to_dict()
        execution_state.set_status(ExecutionStatus.COMPLETED if succeeded else ExecutionStatus.FAILED)
        self.state.transition(RunStatus.COMPLETED if succeeded else RunStatus.FAILED)
        self._checkpoint(execution_state, graph, "execution completion")

        return OrchestrationResult(
            goal=goal,
            final_output=final_output,
            graph=graph,
            events=[e.to_dict() for e in self.event_log.events],
            tool_results=context.tool_results,
            succeeded=succeeded,
            execution_id=execution_state.execution_id,
            execution_state=execution_state,
        )

    async def _run_task(
        self,
        task: Task,
        graph: TaskGraph,
        context: ContextManager,
        execution_state: ExecutionState,
        memory_manager: MemoryManager,
    ) -> Action:
        agent = self.router.route(task)
        task.agent_id = agent.id
        task.started_at = time.time()
        execution_state.sync_task_state(task)
        self.event_log.emit(
            "TASK_STATE_CHANGED", f"Task '{task.id}' -> {task.status.value}", task_id=task.id, status=task.status.value
        )
        self._checkpoint(execution_state, graph, "task start")

        # Agent state is created fresh per (execution, task, agent) call and
        # discarded at the end of it - never a module-level/global mutable
        # object, so it can never leak between tasks, agents, or executions.
        agent_state = AgentState(
            execution_id=execution_state.execution_id,
            task_id=task.id,
            agent_id=agent.id,
            current_objective=task.objective,
        )

        agent_context = context.build_agent_context(
            task=task,
            graph=graph,
            agent=agent,
            execution_state=execution_state,
            memory_manager=memory_manager,
            constraints=execution_state.metadata.get("constraints"),
            max_tokens=self.context_max_tokens,
        )
        agent_input = AgentInput(
            objective=task.objective,
            expected_output=task.expected_output,
            task_context={"task_id": task.id, "agent_state": agent_state},
            upstream_outputs=agent_context.dependency_outputs,
            memory_context=agent_context.memory_snippets,
            constraints=agent_context.constraints,
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
        self._track_artifacts(task, agent, context, execution_state, events_before)

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
            task.completed_at = time.time()
            execution_state.sync_task_state(task)
            self.event_log.emit(
                "TASK_STATE_CHANGED", f"Task '{task.id}' -> {task.status.value}", task_id=task.id, status=task.status.value
            )
            self._checkpoint(execution_state, graph, "task completion")
            if evaluation.verdict == Verdict.SUCCESS:
                self._extract_memory(task, output, memory_manager)
        elif action == Action.RETRY:
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            execution_state.sync_task_state(task)
            self.event_log.emit(
                "RETRY",
                f"Retrying task '{task.id}' (attempt {task.retry_count}/{task.max_retries})",
                task_id=task.id,
                retry_count=task.retry_count,
            )
            action = await self._run_task(task, graph, context, execution_state, memory_manager)
        elif action == Action.REPLAN:
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            execution_state.sync_task_state(task)
            self.event_log.emit(
                "TASK_STATE_CHANGED", f"Task '{task.id}' -> {task.status.value}", task_id=task.id, status=task.status.value
            )
            self._checkpoint(execution_state, graph, "task failure")
        else:  # ABORT
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            execution_state.sync_task_state(task)
            self.event_log.emit("TASK", f"Task '{task.id}' aborted (non-recoverable)", task_id=task.id, status="aborted")
            self._checkpoint(execution_state, graph, "task failure")

        return action

    def _checkpoint(self, execution_state: ExecutionState, graph: TaskGraph, reason: str) -> None:
        """Refresh the persisted plan snapshot from the live graph (so a
        resume always sees each task's current status/result, not a stale
        all-pending snapshot from plan creation) and checkpoint."""
        execution_state.current_plan = graph.to_dict()
        self.state_store.checkpoint(execution_state, self.event_log, reason)

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

    def _track_artifacts(
        self,
        task: Task,
        agent: BaseAgent,
        context: ContextManager,
        execution_state: ExecutionState,
        events_since_index: int,
    ) -> None:
        """Any successful file_write in this task becomes a tracked
        Artifact (path/reference only - never the file content itself)."""
        for event in self.event_log.events[events_since_index:]:
            if event.tag != "TOOL_RESULT" or event.tool_id != "file_write" or event.status != "success":
                continue
            output = event.extra.get("output") or {}
            path = output.get("path")
            if not path:
                continue
            artifact = Artifact(
                artifact_id=f"artifact_{task.id}_{len(execution_state.artifacts)}",
                type="file",
                path=path,
                task_id=task.id,
                agent_id=agent.id,
                metadata={"bytes_written": output.get("bytes_written")},
            )
            execution_state.add_artifact(artifact)

    def _extract_memory(self, task: Task, output: AgentOutput, memory_manager: MemoryManager) -> None:
        """After a task the evaluator judged successful, offer its result
        up for long-term memory extraction. MemoryPolicy - not this method -
        makes the actual store/skip/importance/scope decision."""
        memory_manager.store(
            type=MemoryType.TASK_RESULT,
            content={"objective": task.objective, "result": output.content},
            source=f"task:{task.id}",
            task_id=task.id,
            agent_id=task.agent_id,
        )

    def _build_memory_manager(self, execution_id: str) -> MemoryManager:
        return MemoryManager(
            execution_id,
            self.long_term_memory,
            policy=self.memory_policy,
            event_log=self.event_log,
            session_id=self.session_id,
            project_id=self.project_id,
            user_id=self.user_id,
        )

    async def _replan(
        self,
        goal: str,
        graph: TaskGraph,
        context: ContextManager,
        failed_task: Task,
        execution_state: ExecutionState,
    ) -> bool:
        if not self.state.can_replan():
            return False
        self.state.transition(RunStatus.REPLANNING)
        self.state.record_replan()
        execution_state.replans_used = self.state.replans_used

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
        execution_state.current_plan = graph.to_dict()
        self.state.transition(RunStatus.EXECUTING)
        self.state_store.checkpoint(execution_state, self.event_log, "replan")
        return True
