"""Tool execution runtime.

This is the enforcement point for the whole tool layer. Agents never call
tool implementations directly - they call ``ToolRuntime.call(...)``, which
runs the full flow:

    Tool Request -> Permission Check -> Input Validation
        -> Tool Execution (with timeout) -> Output Validation -> Tool Result

Every step is observed via one of the ``TOOL_REQUEST`` / ``TOOL_VALIDATION``
/ ``TOOL_PERMISSION`` / ``TOOL_EXECUTION`` / ``TOOL_RESULT`` / ``TOOL_ERROR``
tags, and no step is ever silently skipped or swallowed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from orchestrator.core.logging_utils import EventLog
from orchestrator.tools.base import ToolErrorCode, ToolResult
from orchestrator.tools.permissions import missing_permissions
from orchestrator.tools.registry import ToolNotFoundError, ToolRegistry
from orchestrator.tools.schema_validation import validate_against_schema

_SENSITIVE_KEY_MARKERS = ("key", "token", "secret", "password", "credential", "authorization")


def _summarize_args(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Argument *metadata* only (keys, types, lengths) - never raw values,
    so logs can never leak secrets or large payloads."""
    summary: dict[str, Any] = {}
    for key, value in kwargs.items():
        if any(marker in key.lower() for marker in _SENSITIVE_KEY_MARKERS):
            summary[key] = "<redacted>"
            continue
        if isinstance(value, str):
            summary[key] = f"str(len={len(value)})"
        elif isinstance(value, (int, float, bool)) or value is None:
            summary[key] = value
        elif isinstance(value, (list, tuple)):
            summary[key] = f"{type(value).__name__}(len={len(value)})"
        elif isinstance(value, dict):
            summary[key] = f"dict(keys={len(value)})"
        else:
            summary[key] = type(value).__name__
    return summary


class ToolRuntime:
    def __init__(
        self,
        registry: ToolRegistry,
        event_log: EventLog | None = None,
        *,
        default_timeout_seconds: float = 30.0,
    ) -> None:
        self.registry = registry
        self.event_log = event_log
        self.default_timeout_seconds = default_timeout_seconds

    async def call(
        self,
        tool_id: str,
        *,
        task_id: str | None = None,
        agent_id: str | None = None,
        agent_permissions: list[str] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        log_ctx = dict(task_id=task_id, agent_id=agent_id, tool_id=tool_id)
        arg_summary = _summarize_args(kwargs)

        self._emit("TOOL_REQUEST", f"Requested tool '{tool_id}'", log_ctx, extra={"args": arg_summary})

        # 1. Availability
        try:
            tool = self.registry.get(tool_id)
        except ToolNotFoundError:
            return self._fail(
                log_ctx,
                ToolErrorCode.UNAVAILABLE,
                f"Tool '{tool_id}' is not registered/available",
                tag="TOOL_REQUEST",
            )

        # 2. Permission check (enforced here, not just in the LLM prompt)
        missing = missing_permissions(tool.permissions, agent_permissions or [])
        if missing:
            self._emit(
                "TOOL_PERMISSION",
                f"Permission denied for tool '{tool_id}'",
                log_ctx,
                status="denied",
                extra={"missing_permissions": missing, "required": tool.permissions},
            )
            return self._fail(
                log_ctx,
                ToolErrorCode.PERMISSION_DENIED,
                f"Agent lacks required permission(s) for '{tool_id}': {missing}",
                tag=None,
            )
        self._emit(
            "TOOL_PERMISSION",
            f"Permission granted for tool '{tool_id}'",
            log_ctx,
            status="granted",
            extra={"required": tool.permissions},
        )

        # 3. Input validation
        validation_errors = validate_against_schema(kwargs, tool.input_schema)
        if validation_errors:
            self._emit(
                "TOOL_VALIDATION",
                f"Invalid arguments for tool '{tool_id}'",
                log_ctx,
                status="failure",
                extra={"errors": validation_errors},
            )
            return self._fail(
                log_ctx,
                ToolErrorCode.INVALID_ARGUMENTS,
                f"Invalid arguments for '{tool_id}': {'; '.join(validation_errors)}",
                tag=None,
            )
        self._emit("TOOL_VALIDATION", f"Arguments valid for tool '{tool_id}'", log_ctx, status="success")

        # 4. Execution (with timeout)
        effective_timeout = timeout or tool.timeout_seconds or self.default_timeout_seconds
        self._emit(
            "TOOL_EXECUTION",
            f"Executing tool '{tool_id}'",
            log_ctx,
            extra={"timeout_seconds": effective_timeout},
        )
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(tool.execute(**kwargs), timeout=effective_timeout)
        except TimeoutError:
            duration_ms = (time.perf_counter() - started) * 1000
            return self._fail(
                log_ctx,
                ToolErrorCode.TIMEOUT,
                f"Tool '{tool_id}' timed out after {effective_timeout}s",
                tag="TOOL_RESULT",
                duration_ms=duration_ms,
            )
        except Exception as exc:  # noqa: BLE001 - convert to structured failure, never swallow
            duration_ms = (time.perf_counter() - started) * 1000
            return self._fail(
                log_ctx,
                ToolErrorCode.EXECUTION_ERROR,
                f"Tool '{tool_id}' raised {type(exc).__name__}: {exc}",
                tag="TOOL_RESULT",
                duration_ms=duration_ms,
            )
        duration_ms = (time.perf_counter() - started) * 1000

        if not result.success:
            return self._fail(
                log_ctx,
                result.error_code or ToolErrorCode.EXECUTION_ERROR,
                result.error or f"Tool '{tool_id}' reported failure",
                tag="TOOL_RESULT",
                duration_ms=duration_ms,
            )

        # 5. Output validation
        output_errors = validate_against_schema(result.output, tool.output_schema)
        if output_errors:
            return self._fail(
                log_ctx,
                ToolErrorCode.MALFORMED_OUTPUT,
                f"Tool '{tool_id}' produced output that fails its schema: {'; '.join(output_errors)}",
                tag="TOOL_RESULT",
                duration_ms=duration_ms,
            )

        self._emit(
            "TOOL_RESULT",
            f"Tool '{tool_id}' succeeded",
            log_ctx,
            status="success",
            duration_ms=round(duration_ms, 2),
            extra={"output": result.output},
        )
        return result

    def _fail(
        self,
        log_ctx: dict[str, Any],
        error_code: ToolErrorCode,
        message: str,
        *,
        tag: str | None,
        duration_ms: float | None = None,
    ) -> ToolResult:
        if tag:
            self._emit(tag, message, log_ctx, status="failure", error=message)
        self._emit(
            "TOOL_ERROR",
            message,
            log_ctx,
            status=error_code.value,
            error=message,
            duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
        )
        return ToolResult(success=False, error=message, error_code=error_code)

    def _emit(
        self,
        tag: str,
        message: str,
        log_ctx: dict[str, Any],
        *,
        status: str | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.event_log:
            return
        self.event_log.emit(
            tag,
            message,
            task_id=log_ctx.get("task_id"),
            agent_id=log_ctx.get("agent_id"),
            tool_id=log_ctx.get("tool_id"),
            status=status,
            error=error,
            duration_ms=duration_ms,
            extra=extra or {},
        )
