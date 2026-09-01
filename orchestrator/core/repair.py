"""RepairManager: focused re-execution of a task whose result doesn't
satisfy its acceptance criteria - without restarting the task, the DAG, or
the execution.

    TASK -> AGENT -> RESULT -> EVALUATOR -> REPAIR_REQUIRED
         -> REPAIR MANAGER -> AGENT -> NEW RESULT -> EVALUATOR

The repair agent is the *same* routed agent, called again through the
normal ``BaseAgent.execute`` interface - the only difference is the
``AgentInput`` it receives carries ``repair_feedback`` (previous output,
failed criteria, evaluator reasons/evidence) instead of a blank slate, so
the fix stays targeted at what's actually wrong.
"""

from __future__ import annotations

from orchestrator.agents.base import AgentInput, AgentOutput, BaseAgent
from orchestrator.core.evaluation_models import EvaluationResult
from orchestrator.core.task_graph import Task


class RepairManager:
    async def repair(
        self,
        *,
        agent: BaseAgent,
        task: Task,
        previous_output: AgentOutput,
        evaluation: EvaluationResult,
        base_input: AgentInput,
    ) -> AgentOutput:
        repair_input = AgentInput(
            objective=base_input.objective,
            expected_output=base_input.expected_output,
            task_context=base_input.task_context,
            upstream_outputs=base_input.upstream_outputs,
            memory_context=base_input.memory_context,
            constraints=base_input.constraints,
            repair_feedback={
                "attempt": task.repair_attempt,
                "previous_output": previous_output.content,
                "failed_criteria": evaluation.failed_criteria,
                "reasons": evaluation.reasons,
                "evidence": evaluation.evidence,
            },
        )
        return await agent.execute(repair_input)
