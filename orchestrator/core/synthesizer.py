"""Final Result Synthesizer.

Combines outputs of all successful (leaf-ward) tasks into one coherent
answer to the original goal.
"""

from __future__ import annotations

from orchestrator.core.context import ContextManager
from orchestrator.core.logging_utils import EventLog
from orchestrator.core.task_graph import TaskGraph
from orchestrator.providers.base import LLMMessage, LLMProvider

_SYNTHESIS_SYSTEM = (
    "You are the final synthesis stage of an AI agent orchestrator. Combine the "
    "given task outputs into a single, coherent response that directly satisfies "
    "the user's original goal. Do not repeat yourself; do not include process "
    "commentary; produce only the final deliverable."
)


class FinalResultSynthesizer:
    def __init__(self, provider: LLMProvider, event_log: EventLog | None = None) -> None:
        self.provider = provider
        self.event_log = event_log

    async def synthesize(self, goal: str, graph: TaskGraph, context: ContextManager) -> str:
        succeeded = graph.succeeded_tasks()
        if not succeeded:
            result = "The orchestrator could not produce a result: no tasks completed successfully."
            if self.event_log:
                self.event_log.emit("COMPLETE", result, status="failed")
            return result

        sections = []
        for task in succeeded:
            output = context.task_outputs.get(task.id)
            if output:
                sections.append(f"### {task.objective}\n{output.content}")
        combined = "\n\n".join(sections)

        prompt = f"Original goal: {goal}\n\nCompleted task outputs:\n\n{combined}\n\nProduce the final result."
        response = await self.provider.complete(system=_SYNTHESIS_SYSTEM, messages=[LLMMessage("user", prompt)])
        final_text = response.text.strip() or combined

        if self.event_log:
            self.event_log.emit(
                "COMPLETE",
                "Final result synthesized",
                status="succeeded" if not graph.failed_tasks() else "partial",
                model=response.model,
                tokens_used=response.total_tokens,
            )
        return final_text
