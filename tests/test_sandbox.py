import pytest

from orchestrator.tools.sandbox import FileSandbox, SandboxViolationError


@pytest.fixture
def sandbox(tmp_path):
    return FileSandbox(tmp_path / "sandbox_root")


def test_resolve_simple_relative_path_stays_inside_root(sandbox):
    resolved = sandbox.resolve("notes.txt")
    assert resolved.parent == sandbox.root


def test_resolve_nested_relative_path(sandbox):
    resolved = sandbox.resolve("subdir/notes.txt")
    assert resolved == sandbox.root / "subdir" / "notes.txt"


def test_dotdot_traversal_is_rejected(sandbox):
    with pytest.raises(SandboxViolationError):
        sandbox.resolve("../outside.txt")


def test_deep_dotdot_traversal_is_rejected(sandbox):
    with pytest.raises(SandboxViolationError):
        sandbox.resolve("a/b/../../../etc/passwd")


def test_absolute_path_cannot_override_the_sandbox_root(sandbox):
    # A naive os.path.join("/sandbox", "/etc/passwd") would resolve to
    # /etc/passwd - this must not escape the sandbox.
    resolved = sandbox.resolve("/etc/passwd")
    assert resolved == sandbox.root / "etc" / "passwd"


def test_write_then_read_round_trip(sandbox):
    sandbox.write_text("greeting.txt", "hello sandbox")
    assert sandbox.read_text("greeting.txt") == "hello sandbox"


def test_write_creates_parent_directories_inside_root(sandbox):
    sandbox.write_text("a/b/c.txt", "nested")
    assert (sandbox.root / "a" / "b" / "c.txt").exists()


def test_read_missing_file_raises_file_not_found(sandbox):
    with pytest.raises(FileNotFoundError):
        sandbox.read_text("missing.txt")


def test_read_directory_traversal_via_symlink_is_rejected(sandbox, tmp_path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("top secret")

    link = sandbox.root / "escape"
    link.symlink_to(outside_dir)

    with pytest.raises(SandboxViolationError):
        sandbox.resolve("escape/secret.txt")


def test_write_rejects_oversized_content(tmp_path):
    small_sandbox = FileSandbox(tmp_path / "small", max_file_size_bytes=10)
    with pytest.raises(ValueError):
        small_sandbox.write_text("big.txt", "this content is definitely more than ten bytes")


def test_list_dir_reports_entries(sandbox):
    sandbox.write_text("one.txt", "1")
    sandbox.write_text("nested/two.txt", "2")
    entries = sandbox.list_dir(".")
    names = {e["name"] for e in entries}
    assert names == {"one.txt", "nested"}
