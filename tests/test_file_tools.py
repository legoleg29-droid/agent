import pytest

from orchestrator.core.logging_utils import EventLog
from orchestrator.tools.file_tools import FileReadTool, FileWriteTool, ListFilesTool
from orchestrator.tools.permissions import FILESYSTEM_READ, FILESYSTEM_WRITE
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.runtime import ToolRuntime
from orchestrator.tools.sandbox import FileSandbox


@pytest.fixture
def wired_runtime(tmp_path):
    sandbox = FileSandbox(tmp_path / "sandbox_root")
    registry = ToolRegistry()
    registry.register(FileReadTool(sandbox))
    registry.register(FileWriteTool(sandbox))
    registry.register(ListFilesTool(sandbox))
    runtime = ToolRuntime(registry, EventLog(verbose=False))
    return runtime, sandbox


@pytest.mark.asyncio
async def test_write_then_read_via_tool_runtime(wired_runtime):
    runtime, _sandbox = wired_runtime
    write_result = await runtime.call(
        "file_write", path="result.txt", content="42", agent_permissions=[FILESYSTEM_WRITE]
    )
    assert write_result.success
    assert write_result.output["bytes_written"] == 2

    read_result = await runtime.call("file_read", path="result.txt", agent_permissions=[FILESYSTEM_READ])
    assert read_result.success
    assert read_result.output["content"] == "42"


@pytest.mark.asyncio
async def test_write_denied_without_permission(wired_runtime):
    runtime, _sandbox = wired_runtime
    result = await runtime.call("file_write", path="x.txt", content="y", agent_permissions=[])
    assert not result.success
    assert result.error_code.value == "permission_denied"


@pytest.mark.asyncio
async def test_read_denied_without_permission(wired_runtime):
    runtime, sandbox = wired_runtime
    sandbox.write_text("visible.txt", "content")
    result = await runtime.call("file_read", path="visible.txt", agent_permissions=[])
    assert not result.success
    assert result.error_code.value == "permission_denied"


@pytest.mark.asyncio
async def test_path_traversal_is_rejected_even_with_full_permissions(wired_runtime, tmp_path):
    runtime, _sandbox = wired_runtime
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("outside the sandbox")

    result = await runtime.call(
        "file_read", path="../secret.txt", agent_permissions=[FILESYSTEM_READ, FILESYSTEM_WRITE]
    )
    assert not result.success
    # Sandbox violations surface as permission_denied - having a permission
    # never grants escaping the sandbox root.
    assert result.error_code.value == "permission_denied"


@pytest.mark.asyncio
async def test_list_files_reports_directory_contents(wired_runtime):
    runtime, sandbox = wired_runtime
    sandbox.write_text("a.txt", "1")
    sandbox.write_text("b.txt", "2")
    result = await runtime.call("list_files", path=".", agent_permissions=[FILESYSTEM_READ])
    assert result.success
    names = {e["name"] for e in result.output["entries"]}
    assert names == {"a.txt", "b.txt"}
