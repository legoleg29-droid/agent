"""Narrowly-scoped test runner used to independently verify coding-agent
output, instead of trusting the agent's own claim that "the tests pass".

Runs ``python -m pytest`` inside the sandbox root only, via
``asyncio.create_subprocess_exec`` (argv list, never a shell string - no
shell interpolation, no arbitrary command execution). The only user-facing
input is a sandbox-relative ``path`` to test; everything else about the
invocation is fixed.
"""

from __future__ import annotations

import asyncio
import sys

from orchestrator.tools.base import BaseTool, ToolErrorCode, ToolResult
from orchestrator.tools.permissions import CODE_EXECUTION, FILESYSTEM_READ
from orchestrator.tools.sandbox import FileSandbox, SandboxViolationError

_MAX_OUTPUT_CHARS = 8000


class RunPythonTestsTool(BaseTool):
    id = "run_python_tests"
    name = "Run Python Tests"
    description = (
        "Runs pytest against a file or directory inside the sandboxed working "
        "directory and reports the real exit code and output - used to "
        "independently verify code the coding agent wrote, never to run "
        "arbitrary shell commands."
    )
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Sandbox-relative file or directory to test."}},
        "required": ["path"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "exit_code": {"type": "integer"},
            "passed": {"type": "boolean"},
            "output": {"type": "string"},
        },
        "required": ["path", "exit_code", "passed", "output"],
    }
    permissions = [CODE_EXECUTION, FILESYSTEM_READ]
    capabilities = ["code_execution"]
    timeout_seconds = 30.0

    def __init__(self, sandbox: FileSandbox, *, timeout_seconds: float | None = None) -> None:
        self.sandbox = sandbox
        if timeout_seconds is not None:
            self.timeout_seconds = timeout_seconds

    async def execute(self, *, path: str) -> ToolResult:
        try:
            target = self.sandbox.resolve(path)
        except SandboxViolationError as exc:
            return ToolResult(success=False, error=str(exc), error_code=ToolErrorCode.PERMISSION_DENIED)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc), error_code=ToolErrorCode.INVALID_ARGUMENTS)
        if not target.exists():
            return ToolResult(
                success=False,
                error=f"Path not found: {path}",
                error_code=ToolErrorCode.INVALID_ARGUMENTS,
            )

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pytest",
                str(target),
                "-q",
                "--no-header",
                cwd=str(self.sandbox.root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=self.timeout_seconds)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    error=f"Test run exceeded {self.timeout_seconds}s timeout",
                    error_code=ToolErrorCode.TIMEOUT,
                )
        except FileNotFoundError as exc:
            return ToolResult(success=False, error=f"pytest is not available: {exc}", error_code=ToolErrorCode.EXECUTION_ERROR)

        output = stdout.decode("utf-8", errors="replace")[-_MAX_OUTPUT_CHARS:]
        exit_code = process.returncode or 0
        passed = exit_code == 0

        # A tool "succeeding" here means the tests passed - not merely that
        # pytest ran without crashing - so a TOOL_SUCCEEDED acceptance
        # criterion means what it says: the tests actually pass, never just
        # that the subprocess started. A failing test run is reported as a
        # tool failure (with the real output attached) rather than success.
        if passed:
            return ToolResult(success=True, output={"path": path, "exit_code": exit_code, "passed": True, "output": output})
        return ToolResult(
            success=False,
            error=f"pytest failed (exit code {exit_code}):\n{output}",
            error_code=ToolErrorCode.EXECUTION_ERROR,
        )
