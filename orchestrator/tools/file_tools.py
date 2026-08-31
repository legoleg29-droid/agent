"""Native filesystem tools, all sandboxed through ``FileSandbox``.

No tool here ever touches the filesystem directly with a caller-supplied
path - every operation goes through ``FileSandbox.resolve()`` (or its
``read_text``/``write_text``/``list_dir`` wrappers), which rejects any
path that would escape the configured sandbox root. There is no shell or
arbitrary command execution here or anywhere else in this codebase.
"""

from __future__ import annotations

from orchestrator.tools.base import BaseTool, ToolErrorCode, ToolResult
from orchestrator.tools.permissions import FILESYSTEM_READ, FILESYSTEM_WRITE
from orchestrator.tools.sandbox import FileSandbox, SandboxViolationError


class FileReadTool(BaseTool):
    id = "file_read"
    name = "File Read"
    description = "Reads the text content of a file inside the sandboxed working directory."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    output_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }
    permissions = [FILESYSTEM_READ]
    capabilities = ["filesystem"]
    timeout_seconds = 10.0

    def __init__(self, sandbox: FileSandbox) -> None:
        self.sandbox = sandbox

    async def execute(self, *, path: str) -> ToolResult:
        try:
            content = self.sandbox.read_text(path)
        except SandboxViolationError as exc:
            return ToolResult(success=False, error=str(exc), error_code=ToolErrorCode.PERMISSION_DENIED)
        except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
            return ToolResult(success=False, error=str(exc), error_code=ToolErrorCode.INVALID_ARGUMENTS)
        return ToolResult(success=True, output={"path": path, "content": content})


class FileWriteTool(BaseTool):
    id = "file_write"
    name = "File Write"
    description = "Writes text content to a file inside the sandboxed working directory, creating it if needed."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }
    output_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "bytes_written": {"type": "integer"}},
        "required": ["path", "bytes_written"],
    }
    permissions = [FILESYSTEM_WRITE]
    capabilities = ["filesystem"]
    timeout_seconds = 10.0

    def __init__(self, sandbox: FileSandbox) -> None:
        self.sandbox = sandbox

    async def execute(self, *, path: str, content: str) -> ToolResult:
        try:
            bytes_written = self.sandbox.write_text(path, content)
        except SandboxViolationError as exc:
            return ToolResult(success=False, error=str(exc), error_code=ToolErrorCode.PERMISSION_DENIED)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc), error_code=ToolErrorCode.INVALID_ARGUMENTS)
        return ToolResult(success=True, output={"path": path, "bytes_written": bytes_written})


class ListFilesTool(BaseTool):
    id = "list_files"
    name = "List Files"
    description = "Lists files and directories at a path inside the sandboxed working directory."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }
    output_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "entries": {"type": "array"}},
        "required": ["path", "entries"],
    }
    permissions = [FILESYSTEM_READ]
    capabilities = ["filesystem"]
    timeout_seconds = 10.0

    def __init__(self, sandbox: FileSandbox) -> None:
        self.sandbox = sandbox

    async def execute(self, *, path: str = ".") -> ToolResult:
        try:
            entries = self.sandbox.list_dir(path)
        except SandboxViolationError as exc:
            return ToolResult(success=False, error=str(exc), error_code=ToolErrorCode.PERMISSION_DENIED)
        except (FileNotFoundError, NotADirectoryError) as exc:
            return ToolResult(success=False, error=str(exc), error_code=ToolErrorCode.INVALID_ARGUMENTS)
        return ToolResult(success=True, output={"path": path, "entries": entries})
