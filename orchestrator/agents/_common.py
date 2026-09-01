"""Shared helpers for LLM-backed agents.

Not part of the public BaseAgent contract - just a convenience base class
so the example agents don't duplicate prompt plumbing and the tool-use
loop.

Tool use goes through the provider's native structured mechanism (Claude's
``tool_use``/``tool_result`` blocks) rather than parsing free text: the
agent hands the provider a set of tool schemas, and if the model decides
to use one, the response carries a structured ``ToolCallRequest`` instead
of text. This class drives that loop, executing each requested tool via
``ToolRuntime`` (which enforces this agent's permissions) and feeding the
result back until the model produces a final text answer.
"""

from __future__ import annotations

import json

from orchestrator.agents.base import AgentInput, AgentOutput, BaseAgent
from orchestrator.providers.base import LLMMessage

MAX_TOOL_ROUNDS = 4


class LLMAgent(BaseAgent):
    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        messages: list[LLMMessage] = [LLMMessage(role="user", content=self._build_prompt(agent_input))]
        tool_schemas = self._tool_schemas()
        tool_calls_made = 0
        tokens_used = 0
        model: str | None = None

        for _round in range(MAX_TOOL_ROUNDS):
            response = await self.provider.complete(
                system=self.system_instructions,
                messages=messages,
                tools=tool_schemas or None,
            )
            tokens_used += response.total_tokens
            model = response.model

            if not response.tool_calls:
                content = response.text.strip()
                if not content:
                    return AgentOutput(
                        success=False,
                        error="Agent produced empty output",
                        tool_calls=tool_calls_made,
                        tokens_used=tokens_used,
                        model=model,
                    )
                return AgentOutput(
                    success=True,
                    content=content,
                    tool_calls=tool_calls_made,
                    tokens_used=tokens_used,
                    model=model,
                )

            if not self.tool_runtime:
                return AgentOutput(
                    success=False,
                    error="Agent requested a tool but has no ToolRuntime configured",
                    tool_calls=tool_calls_made,
                    tokens_used=tokens_used,
                    model=model,
                )

            # Replay the assistant's turn verbatim (including tool_use
            # blocks) so the next request carries full tool-use context.
            messages.append(LLMMessage(role="assistant", content=response.content_blocks or response.text))

            tool_result_blocks = []
            for call in response.tool_calls:
                if call.name not in self.available_tools:
                    payload = {
                        "success": False,
                        "error": f"Tool '{call.name}' is not in this agent's available_tools",
                        "error_code": "permission_denied",
                    }
                else:
                    tool_result = await self.tool_runtime.call(
                        call.name,
                        task_id=agent_input.task_context.get("task_id"),
                        agent_id=self.id,
                        agent_permissions=self.permissions,
                        **call.arguments,
                    )
                    tool_calls_made += 1
                    payload = tool_result.to_public_dict()
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(payload),
                        "is_error": not payload.get("success", False),
                    }
                )
            messages.append(LLMMessage(role="user", content=tool_result_blocks))

        return AgentOutput(
            success=False,
            error=f"Agent exceeded the maximum number of tool-use rounds ({MAX_TOOL_ROUNDS})",
            tool_calls=tool_calls_made,
            tokens_used=tokens_used,
            model=model,
        )

    def _tool_schemas(self) -> list[dict]:
        if not self.tool_runtime or not self.available_tools:
            return []
        return self.tool_runtime.registry.claude_schemas(self.available_tools)

    def _build_prompt(self, agent_input: AgentInput) -> str:
        if agent_input.repair_feedback:
            return self._build_repair_prompt(agent_input)
        parts = [f"Objective: {agent_input.objective}"]
        if agent_input.expected_output:
            parts.append(f"Expected output: {agent_input.expected_output}")
        if agent_input.upstream_outputs:
            parts.append("Relevant prior task outputs:")
            for task_id, output in agent_input.upstream_outputs.items():
                parts.append(f"- [{task_id}]: {output}")
        if agent_input.constraints:
            parts.append("Constraints:")
            for constraint in agent_input.constraints:
                parts.append(f"- {constraint}")
        if agent_input.memory_context:
            parts.append("Retrieved memory (for context - verify before relying on it):")
            for entry in agent_input.memory_context:
                content = entry.get("content")
                parts.append(f"- ({entry.get('type')}) {content}")
        return "\n\n".join(parts)

    def _build_repair_prompt(self, agent_input: AgentInput) -> str:
        """A focused repair prompt: what needs fixing, not a fresh brief.
        The agent gets its own previous output back and is told exactly
        which acceptance criteria it failed, so it can target the fix
        instead of redoing everything from scratch."""
        fb = agent_input.repair_feedback or {}
        parts = [
            f"Objective (unchanged): {agent_input.objective}",
            f"REPAIR REQUEST (attempt {fb.get('attempt', 1)}): your previous attempt at this "
            "objective did not satisfy all acceptance criteria. Fix ONLY what's identified "
            "below - do not discard work that already satisfies other criteria.",
        ]
        if fb.get("failed_criteria"):
            parts.append("Failed criteria:\n" + "\n".join(f"- {c}" for c in fb["failed_criteria"]))
        if fb.get("reasons"):
            parts.append("Evaluator reasons:\n" + "\n".join(f"- {r}" for r in fb["reasons"]))
        if fb.get("evidence"):
            parts.append("Evaluator evidence:\n" + "\n".join(f"- {e}" for e in fb["evidence"]))
        if fb.get("previous_output"):
            parts.append(f"Your previous output was:\n{fb['previous_output']}")
        if agent_input.upstream_outputs:
            parts.append("Relevant prior task outputs:")
            for task_id, output in agent_input.upstream_outputs.items():
                parts.append(f"- [{task_id}]: {output}")
        return "\n\n".join(parts)
