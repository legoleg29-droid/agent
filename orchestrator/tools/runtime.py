"""Tool execution runtime.

Agents call ``ToolRuntime.call(name, **kwargs)`` instead of importing tool
implementations directly. This keeps agents decoupled from concrete tool
implementations and gives us one place to log tool invocations and, later,
route certain tool names to an MCP client transparently.
"""

from __future__ import annotations

import time
from typing import Any

from orchestrator.core.logging_utils import EventLog
from orchestrator.tools.base import ToolResult
from orchestrator.tools.registry import ToolRegistry


class ToolRuntime:
    def __init__(self, registry: ToolRegistry, event_log: EventLog | None = None) -> None:
        self.registry = registry
        self.event_log = event_log

    async def call(
        self,
        name: str,
        *,
        task_id: str | None = None,
        agent_id: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        tool = self.registry.get(name)
        started = time.perf_counter()
        try:
            result = await tool.execute(**kwargs)
        except Exception as exc:  # noqa: BLE001 - convert to structured failure
            result = ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")
        duration_ms = (time.perf_counter() - started) * 1000
        if self.event_log:
            self.event_log.emit(
                "TOOL",
                f"{name} invoked",
                task_id=task_id,
                agent_id=agent_id,
                status="success" if result.success else "failure",
                duration_ms=round(duration_ms, 2),
                error=result.error,
                extra={"tool": name, "args": kwargs},
            )
        return result
