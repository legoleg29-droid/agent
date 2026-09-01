"""Orchestrator: the execution loop.

USER -> ORCHESTRATOR -> PLANNER -> TASK GRAPH -> SCHEDULER -> AGENT RUNTIME
     -> CONTEXT MANAGER -> MEMORY / STATE -> TOOL RUNTIME -> EVALUATOR
     -> CHECKPOINT -> REPLAN / COMPLETE

Dependency-aware, concurrency-bounded task dispatch is delegated to
``DAGScheduler`` (``orchestrator/core/scheduler.py``); this module owns
routing, agent execution, evaluation, retry/backoff, checkpointing, and
replanning for one task, and interprets the scheduler's overall result.
Every important transition (plan creation, task start, task completion,
task failure, retry, replan, pause, cancellation, execution completion) is
checkpointed through a pluggable ``StateStore`` so a run can be
reconstructed and resumed after a crash without re-executing completed
work - see ``resume_execution``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from orchestrator.agents.base import AgentInput, AgentOutput, BaseAgent
from orchestrator.agents.registry import AgentRegistry
from orchestrator.core.context import ContextManager
from orchestrator.core.evaluation_models import EvaluationContext, EvaluationResult, EvaluationStatus
from orchestrator.core.evaluation_policy import EvaluationPolicy
from orchestrator.core.evaluator import Evaluator
from orchestrator.core.logging_utils import EventLog
from orchestrator.core.plan_patch import apply_plan_patch
from orchestrator.core.plan_version import PlanVersion
from orchestrator.core.planner import Planner
from orchestrator.core.repair import RepairManager
from orchestrator.core.retry import Action, RetryPolicy
from orchestrator.core.router import AgentRouter
from orchestrator.core.scheduler import DAGScheduler, ResourceLimits, SchedulerResult
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
from orchestrator.tools.sandbox import FileSandbox

DEFAULT_CONTEXT_MAX_TOKENS = 3000

# Same failure signature recurring this many times during one execution
# means replanning isn't converging - stop instead of oscillating forever.
MAX_REPEATED_FAILURE_SIGNATURE = 2


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
        max_repairs_per_task: int = 2,
        max_replans: int = 2,
        verbose_logging: bool = True,
        state_store: StateStore | None = None,
        long_term_memory: LongTermMemory | None = None,
        memory_policy: MemoryPolicy | None = None,
        context_max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
        session_id: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
        max_concurrent_tasks: int = 5,
        max_execution_time_seconds: float | None = None,
        retry_backoff_base_seconds: float = 0.5,
        max_retry_backoff_seconds: float = 10.0,
        sandbox: FileSandbox | None = None,
        evaluation_policy: EvaluationPolicy | None = None,
    ) -> None:
        self.provider = provider
        self.agent_registry = agent_registry
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_retries_per_task = max_retries_per_task
        self.max_repairs_per_task = max_repairs_per_task
        self.max_replans = max_replans
        self.context_max_tokens = context_max_tokens
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_execution_time_seconds = max_execution_time_seconds
        self.retry_backoff_base_seconds = retry_backoff_base_seconds
        self.max_retry_backoff_seconds = max_retry_backoff_seconds

        self.event_log = EventLog(verbose=verbose_logging)
        self.tool_runtime = ToolRuntime(self.tool_registry, self.event_log)
        self.planner = Planner(provider, self.event_log)
        self.router = AgentRouter(agent_registry, self.event_log)
        self.evaluator = Evaluator(provider=provider, policy=evaluation_policy, sandbox=sandbox)
        self.retry_policy = RetryPolicy()
        self.repair_manager = RepairManager()
        self.synthesizer = FinalResultSynthesizer(provider, self.event_log)

        # Last computed EvaluationResult per task id, for as long as this
        # attempt is in flight - the scheduler's on_replan_needed hook only
        # gets the failed Task back, not the evaluation that triggered the
        # replan, so this closes that gap without threading one more
        # parameter through the scheduler's generic TaskExecutor contract.
        self._last_evaluations: dict[str, EvaluationResult] = {}

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

        # Live DAGScheduler instances, keyed by execution_id, for as long as
        # a run()/resume_execution() call is in flight on this Orchestrator -
        # lets pause_execution()/cancel_execution() signal a run that's
        # actually executing right now (from another coroutine/task), not
        # just a persisted-but-not-live one.
        self._active_schedulers: dict[str, DAGScheduler] = {}

        # Current PlanVersion per execution_id, so replanning can link a new
        # version back to its parent - see plan_version.py.
        self._plan_versions: dict[str, PlanVersion] = {}

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
            task.max_repairs = self.max_repairs_per_task
        execution_state.current_plan = graph.to_dict()

        initial_version = PlanVersion.initial(graph.to_dict(), change_reason="initial plan")
        self._plan_versions[execution_state.execution_id] = initial_version
        execution_state.record_plan_version(initial_version.to_dict())
        self.event_log.emit(
            "PLAN_VERSION_CREATED",
            f"Plan version {initial_version.version} created for '{execution_state.execution_id}'",
            extra={"execution_id": execution_state.execution_id, "plan_id": initial_version.plan_id, "version": initial_version.version},
        )
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

        if state.plan_versions:
            self._plan_versions[execution_id] = PlanVersion.from_dict(state.plan_versions[-1])
        else:
            restored = PlanVersion.initial(graph.to_dict(), change_reason="restored on resume")
            self._plan_versions[execution_id] = restored
            state.record_plan_version(restored.to_dict())

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
        self.event_log.emit("EXECUTION_RESUMED", f"Execution '{execution_id}' resumed", extra={"execution_id": execution_id})

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
        """Request cancellation. If this execution is actively running on
        this Orchestrator instance, signals the live scheduler (which stops
        starting new tasks, cancels in-flight ones, and marks every
        unfinished task CANCELLED - never marks anything unfinished as
        completed) - its own completion path persists the final state.
        Otherwise, best-effort: mark the persisted (not currently running)
        execution CANCELLED directly."""
        scheduler = self._active_schedulers.get(execution_id)
        if scheduler is not None:
            scheduler.request_cancel()
            return
        state = self.state_store.load(execution_id)
        if state is None:
            raise ExecutionNotFoundError(f"No persisted execution found for id '{execution_id}'")
        state.set_status(ExecutionStatus.CANCELLED)
        self.state_store.checkpoint(state, self.event_log, "cancelled")

    def pause_execution(self, execution_id: str) -> None:
        """Request a pause: stop scheduling new tasks, let currently running
        ones finish, then persist PAUSED. If not live on this instance,
        best-effort marks the persisted execution PAUSED directly."""
        scheduler = self._active_schedulers.get(execution_id)
        if scheduler is not None:
            scheduler.request_pause()
            return
        state = self.state_store.load(execution_id)
        if state is None:
            raise ExecutionNotFoundError(f"No persisted execution found for id '{execution_id}'")
        state.set_status(ExecutionStatus.PAUSED)
        self.state_store.checkpoint(state, self.event_log, "paused")

    # -- Execution loop -----------------------------------------------------

    async def _execute_loop(
        self,
        goal: str,
        graph: TaskGraph,
        context: ContextManager,
        execution_state: ExecutionState,
        memory_manager: MemoryManager,
    ) -> OrchestrationResult:
        """Delegates dependency-aware, concurrency-bounded dispatch to
        ``DAGScheduler`` - this method owns interpreting the scheduler's
        result into execution status/output, not the scheduling itself."""

        async def execute_task(task: Task) -> Action:
            return await self._run_task(task, graph, context, execution_state, memory_manager)

        async def on_replan_needed(task: Task) -> bool:
            return await self._replan(goal, graph, context, task, execution_state)

        def checkpoint() -> None:
            self._checkpoint(execution_state, graph, "scheduler")

        scheduler = DAGScheduler(
            graph,
            execute_task=execute_task,
            resource_limits=ResourceLimits(
                max_concurrent_tasks=self.max_concurrent_tasks,
                max_execution_time_seconds=self.max_execution_time_seconds,
            ),
            event_log=self.event_log,
            execution_id=execution_state.execution_id,
            checkpoint=checkpoint,
            on_replan_needed=on_replan_needed,
        )
        self._active_schedulers[execution_state.execution_id] = scheduler
        try:
            result = await scheduler.run()
        finally:
            self._active_schedulers.pop(execution_state.execution_id, None)

        scheduler.metrics.retried_tasks = sum(1 for t in graph.tasks.values() if t.retry_count > 0)
        execution_state.metadata["scheduler_metrics"] = scheduler.metrics.to_dict()
        execution_state.current_plan = graph.to_dict()

        if result == SchedulerResult.PAUSED:
            execution_state.set_status(ExecutionStatus.PAUSED)
            self.event_log.emit(
                "EXECUTION_PAUSED", f"Execution '{execution_state.execution_id}' paused", extra={"execution_id": execution_state.execution_id}
            )
            self._checkpoint(execution_state, graph, "execution paused")
            return OrchestrationResult(
                goal=goal,
                final_output="",
                graph=graph,
                events=[e.to_dict() for e in self.event_log.events],
                tool_results=context.tool_results,
                succeeded=False,
                execution_id=execution_state.execution_id,
                execution_state=execution_state,
            )

        succeeded = result == SchedulerResult.COMPLETED
        cancelled = result == SchedulerResult.CANCELLED
        final_output = (
            "Execution was cancelled." if cancelled else await self.synthesizer.synthesize(goal, graph, context)
        )

        new_status = (
            ExecutionStatus.CANCELLED
            if cancelled
            else (ExecutionStatus.COMPLETED if succeeded else ExecutionStatus.FAILED)
        )
        execution_state.set_status(new_status)
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

        return await self._evaluate_and_act(task, output, agent, agent_input, graph, context, execution_state, memory_manager)

    async def _evaluate_and_act(
        self,
        task: Task,
        output: AgentOutput,
        agent: BaseAgent,
        agent_input: AgentInput,
        graph: TaskGraph,
        context: ContextManager,
        execution_state: ExecutionState,
        memory_manager: MemoryManager,
    ) -> Action:
        """Independently judges ``output`` (never trusting the agent's own
        success claim) and drives the resulting CONTINUE/RETRY/REPAIR/
        REPLAN/ABORT action - including looping a REPAIR back through the
        same task without restarting the whole DAG."""
        eval_context = EvaluationContext(
            dependency_outputs=agent_input.upstream_outputs,
            tool_results=[r for r in context.tool_results if r.get("task_id") == task.id],
            artifacts=[a for a in execution_state.metadata.get("artifacts_detail", []) if a.get("task_id") == task.id],
        )
        self.event_log.emit("EVALUATION_STARTED", f"Evaluating task '{task.id}'", task_id=task.id, agent_id=agent.id)
        try:
            evaluation = await self.evaluator.evaluate(task, output, context=eval_context)
        except Exception as exc:  # noqa: BLE001 - an evaluator crash is a failure signal, not a hard crash
            self.event_log.emit(
                "EVALUATION_FAILED", f"Evaluator raised on task '{task.id}': {exc}", task_id=task.id, error=str(exc)
            )
            evaluation = EvaluationResult(
                status=EvaluationStatus.FAIL,
                passed=False,
                reasons=[f"Evaluator raised {type(exc).__name__}: {exc}"],
                retry_possible=True,
            )

        self._last_evaluations[task.id] = evaluation
        task.evaluation = {
            "status": evaluation.status.value,
            "score": evaluation.score,
            "failed_criteria": evaluation.failed_criteria,
            "reasons": evaluation.reasons,
            "attempt": task.attempt,
            "repair_attempt": task.repair_attempt,
        }
        execution_state.record_evaluation(task.id, evaluation.to_dict())
        self.event_log.emit(
            "EVALUATION_COMPLETED",
            f"Task '{task.id}' evaluated as {evaluation.status.value} (score={evaluation.score:.2f})",
            task_id=task.id,
            status=evaluation.status.value,
        )
        self.event_log.emit(
            "EVALUATOR",
            f"Task '{task.id}' evaluated as {evaluation.status.value}",
            task_id=task.id,
            status=evaluation.status.value,
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
            if evaluation.status == EvaluationStatus.PASS:
                self._extract_memory(task, output, memory_manager)
        elif action == Action.REPAIR:
            action = await self._repair_task(
                task, output, evaluation, agent, agent_input, graph, context, execution_state, memory_manager
            )
        elif action == Action.RETRY:
            task.retry_count += 1
            task.status = TaskStatus.RETRYING
            execution_state.sync_task_state(task)
            backoff_seconds = self._retry_backoff_seconds(task.retry_count)
            self.event_log.emit(
                "RETRY",
                f"Retrying task '{task.id}' (attempt {task.retry_count}/{task.max_retries})",
                task_id=task.id,
                retry_count=task.retry_count,
            )
            self.event_log.emit(
                "TASK_RETRYING",
                f"Task '{task.id}' backing off {backoff_seconds:.2f}s before attempt {task.retry_count + 1}",
                task_id=task.id,
                agent_id=agent.id,
                status="retrying",
                retry_count=task.retry_count,
                extra={"backoff_seconds": backoff_seconds},
            )
            self._checkpoint(execution_state, graph, "task retrying")
            if backoff_seconds > 0:
                await asyncio.sleep(backoff_seconds)
            task.status = TaskStatus.RUNNING
            execution_state.sync_task_state(task)
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

    async def _repair_task(
        self,
        task: Task,
        previous_output: AgentOutput,
        evaluation: EvaluationResult,
        agent: BaseAgent,
        agent_input: AgentInput,
        graph: TaskGraph,
        context: ContextManager,
        execution_state: ExecutionState,
        memory_manager: MemoryManager,
    ) -> Action:
        """Focused re-execution of ``task`` via ``RepairManager`` - the same
        agent, called again with the failure feedback attached, never a
        full task/DAG restart. Loops back through ``_evaluate_and_act`` so
        a second REPAIR (up to ``task.max_repairs``) or an escalation to
        REPLAN is handled the same way a fresh evaluation would be."""
        task.repair_count += 1
        self.event_log.emit(
            "REPAIR_STARTED",
            f"Repairing task '{task.id}' (attempt {task.repair_count}/{task.max_repairs})",
            task_id=task.id,
            agent_id=agent.id,
            retry_count=task.repair_count,
        )
        self._checkpoint(execution_state, graph, "repair started")

        events_before = len(self.event_log.events)
        try:
            new_output = await self.repair_manager.repair(
                agent=agent, task=task, previous_output=previous_output, evaluation=evaluation, base_input=agent_input
            )
        except Exception as exc:  # noqa: BLE001 - a repair-attempt crash is a failed repair, not a hard crash
            new_output = AgentOutput(success=False, error=f"{type(exc).__name__}: {exc}")

        task.result = new_output
        task.error = new_output.error
        context.record_task_output(task.id, new_output)
        # The repair attempt may itself call tools (e.g. re-running tests) -
        # those results/artifacts must be visible to the re-evaluation below,
        # exactly as they would be for a first attempt.
        self._record_tool_results(task.id, events_before, context)
        self._track_artifacts(task, agent, context, execution_state, events_before)

        outcome = "succeeded" if new_output.success else "failed"
        self.event_log.emit(
            "REPAIR_COMPLETED" if new_output.success else "REPAIR_FAILED",
            f"Repair attempt {task.repair_count} for task '{task.id}' {outcome}",
            task_id=task.id,
            agent_id=agent.id,
            status=outcome,
        )
        execution_state.record_repair(task.id, attempt=task.repair_count, outcome=outcome)
        self._checkpoint(execution_state, graph, "repair completed")

        return await self._evaluate_and_act(
            task, new_output, agent, agent_input, graph, context, execution_state, memory_manager
        )

    def _retry_backoff_seconds(self, attempt: int) -> float:
        """Exponential backoff, capped, so a flaky task doesn't hammer the
        agent/tool immediately: attempt 1 -> base, attempt 2 -> 2x base, ..."""
        return min(self.max_retry_backoff_seconds, self.retry_backoff_base_seconds * (2 ** max(0, attempt - 1)))

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
        """Turns a permanently-failed task into a minimal plan PATCH (never
        a wholesale new plan): completed work is left untouched, and a
        REPLACE_TASK op automatically rewires any BLOCKED dependents onto
        the replacement so they get a real chance to run. Returns False
        (task stays FAILED) if the replan budget is exhausted, the same
        failure keeps recurring (loop protection), or the patch itself
        can't be applied - never lets a bad patch crash the scheduler."""
        if not self.state.can_replan():
            return False

        failure_reason = f"Task '{failed_task.id}' failed: {failed_task.error or 'unspecified failure'}"
        signature = f"{failed_task.capability}::{failure_reason}"
        if execution_state.failure_signatures.count(signature) >= MAX_REPEATED_FAILURE_SIGNATURE:
            self.event_log.emit(
                "LOOP_LIMIT_REACHED",
                f"Failure signature repeated for task '{failed_task.id}' - stopping replanning to avoid an infinite loop",
                task_id=failed_task.id,
                extra={"signature": signature},
            )
            self._checkpoint(execution_state, graph, "loop limit reached")
            return False
        execution_state.record_failure_signature(signature)

        self.state.transition(RunStatus.REPLANNING)
        self.event_log.emit(
            "REPLAN_STARTED",
            f"Replanning around failed task '{failed_task.id}'",
            task_id=failed_task.id,
            extra={"repair_attempts": failed_task.repair_count},
        )
        self._checkpoint(execution_state, graph, "replan started")

        try:
            capabilities = self.agent_registry.all_capabilities()
            tools = [t.id for t in self.tool_registry.list_tools()]
            evaluation = self._last_evaluations.get(failed_task.id)

            ops, reason = await self.planner.replan(
                goal,
                graph=graph,
                capabilities=capabilities,
                tools=tools,
                failed_task=failed_task,
                failure_reason=failure_reason,
                evaluation=evaluation,
                repair_attempts=failed_task.repair_count,
            )

            existing_ids = set(graph.tasks.keys())
            errors = apply_plan_patch(graph, ops)
            for new_id in set(graph.tasks.keys()) - existing_ids:
                new_task = graph.tasks[new_id]
                new_task.max_retries = self.max_retries_per_task
                new_task.max_repairs = self.max_repairs_per_task
            graph.finalize()
        except Exception as exc:  # noqa: BLE001 - a bad replan must not crash the scheduler
            self.event_log.emit(
                "REPLAN_COMPLETED",
                f"Replan for task '{failed_task.id}' failed: {exc}",
                task_id=failed_task.id,
                status="failed",
                error=str(exc),
            )
            self.state.record_replan()
            execution_state.replans_used = self.state.replans_used
            self.state.transition(RunStatus.EXECUTING)
            self._checkpoint(execution_state, graph, "replan failed")
            return False

        self.state.record_replan()
        execution_state.replans_used = self.state.replans_used

        current_version = self._plan_versions.get(execution_state.execution_id) or PlanVersion.initial(graph.to_dict())
        new_version = current_version.next_version(
            graph.to_dict(), change_reason=reason, patch_ops=[op.to_dict() for op in ops]
        )
        self._plan_versions[execution_state.execution_id] = new_version
        execution_state.record_plan_version(new_version.to_dict())
        execution_state.record_replan(failed_task_id=failed_task.id, reason=reason, plan_version=new_version.to_dict())
        self.event_log.emit(
            "PLAN_VERSION_CREATED",
            f"Plan version {new_version.version} created (replan): {reason}",
            extra={"execution_id": execution_state.execution_id, "plan_id": new_version.plan_id, "version": new_version.version},
        )
        self.event_log.emit(
            "REPLAN_COMPLETED",
            f"Replan applied {len(ops) - len(errors)}/{len(ops)} operation(s): {reason}"
            + (f" (errors: {errors})" if errors else ""),
            task_id=failed_task.id,
            extra={"operations": [op.to_dict() for op in ops], "errors": errors},
        )

        execution_state.current_plan = graph.to_dict()
        self.state.transition(RunStatus.EXECUTING)
        self.state_store.checkpoint(execution_state, self.event_log, "replan")
        return True
