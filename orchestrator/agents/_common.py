"""Shared helpers for LLM-backed agents.

Not part of the public BaseAgent contract - just a convenience base class
so the four example agents don't duplicate prompt plumbing and the
tool-request protocol.
"""

from __future__ import annotations

import json
import re

from orchestrator.agents.base import AgentInput, AgentOutput, BaseAgent
from orchestrator.providers.base import LLMMessage

_TOOL_CALL_RE = re.compile(r"TOOL_CALL:\s*(\w+)\((\{.*?\})\)", re.DOTALL)


class LLMAgent(BaseAgent):
    """Base class that wires an agent's system instructions + objective
    through the provider, with one optional round of tool use.

    Protocol: the agent may respond with a line like
    ``TOOL_CALL: web_search({"query": "..."})`` to request a tool. The
    runtime executes it, the result is fed back, and the agent produces
    its final answer. This keeps tool use explicit and provider-agnostic
    instead of depending on a specific vendor's function-calling schema.
    """

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        prompt = self._build_prompt(agent_input)
        messages = [LLMMessage(role="user", content=prompt)]
        tool_calls_made = 0
        tokens_used = 0

        response = await self.provider.complete(
            system=self._system_prompt(), messages=messages
        )
        tokens_used += response.total_tokens

        match = _TOOL_CALL_RE.search(response.text)
        if match and self.tool_runtime and self.available_tools:
            tool_name, raw_args = match.group(1), match.group(2)
            if tool_name in self.available_tools:
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    return AgentOutput(
                        success=False,
                        error=f"Agent requested tool '{tool_name}' with invalid JSON args: {exc}",
                        tokens_used=tokens_used,
                        model=response.model,
                    )
                tool_result = await self.tool_runtime.call(
                    tool_name, task_id=agent_input.task_context.get("task_id"), agent_id=self.id, **args
                )
                tool_calls_made += 1
                follow_up = (
                    f"Tool `{tool_name}` returned: "
                    f"{tool_result.output if tool_result.success else 'ERROR: ' + str(tool_result.error)}\n\n"
                    "Now produce your final answer for the objective, incorporating this result. "
                    "Do not request another tool."
                )
                messages.append(LLMMessage(role="assistant", content=response.text))
                messages.append(LLMMessage(role="user", content=follow_up))
                response = await self.provider.complete(
                    system=self._system_prompt(), messages=messages
                )
                tokens_used += response.total_tokens

        content = response.text.strip()
        if not content:
            return AgentOutput(
                success=False,
                error="Agent produced empty output",
                tool_calls=tool_calls_made,
                tokens_used=tokens_used,
                model=response.model,
            )
        return AgentOutput(
            success=True,
            content=content,
            tool_calls=tool_calls_made,
            tokens_used=tokens_used,
            model=response.model,
        )

    def _system_prompt(self) -> str:
        tools_note = ""
        if self.available_tools:
            tools_note = (
                "\n\nIf you need external information to complete the objective, respond with "
                "exactly one line: TOOL_CALL: <tool_name>({\"arg\": \"value\"})\n"
                f"Available tools: {', '.join(self.available_tools)}."
            )
        return f"{self.system_instructions}{tools_note}"

    def _build_prompt(self, agent_input: AgentInput) -> str:
        parts = [f"Objective: {agent_input.objective}"]
        if agent_input.expected_output:
            parts.append(f"Expected output: {agent_input.expected_output}")
        if agent_input.upstream_outputs:
            parts.append("Relevant prior task outputs:")
            for task_id, output in agent_input.upstream_outputs.items():
                parts.append(f"- [{task_id}]: {output}")
        return "\n\n".join(parts)
