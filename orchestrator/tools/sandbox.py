"""Filesystem sandbox.

Every filesystem tool resolves paths through a single ``FileSandbox``
instance so there is exactly one place that enforces the boundary: all
paths are treated as relative to a configured root directory, and any
path that would resolve outside that root (via ``..``, an absolute path,
or a symlink) is rejected. There is no unrestricted filesystem access
anywhere in this codebase.
"""

from __future__ import annotations

from pathlib import Path


class SandboxViolationError(PermissionError):
    """Raised when a requested path would escape the sandbox root."""


class FileSandbox:
    def __init__(self, root: str | Path, *, max_file_size_bytes: int = 5_000_000) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file_size_bytes = max_file_size_bytes

    def resolve(self, relative_path: str) -> Path:
        """Resolve ``relative_path`` against the sandbox root, rejecting any
        path (absolute, ``..``-traversal, or symlink) that would escape it.
        """
        if not relative_path or not relative_path.strip():
            raise ValueError("path must be non-empty")

        # An absolute input path must not be able to override the sandbox
        # root when joined - strip any leading separators so it's always
        # treated as relative to the root.
        cleaned = relative_path.lstrip("/\\")
        candidate = (self.root / cleaned).resolve()

        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise SandboxViolationError(
                f"Path '{relative_path}' resolves outside the sandbox root '{self.root}'"
            ) from exc
        return candidate

    def list_dir(self, relative_path: str = ".") -> list[dict]:
        target = self.resolve(relative_path)
        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {relative_path}")
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {relative_path}")
        entries = []
        for entry in sorted(target.iterdir()):
            entries.append(
                {
                    "name": entry.name,
                    "type": "directory" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else None,
                }
            )
        return entries

    def read_text(self, relative_path: str) -> str:
        target = self.resolve(relative_path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")
        if not target.is_file():
            raise IsADirectoryError(f"Not a file: {relative_path}")
        size = target.stat().st_size
        if size > self.max_file_size_bytes:
            raise ValueError(f"File too large ({size} bytes > {self.max_file_size_bytes} byte limit)")
        return target.read_text(encoding="utf-8")

    def write_text(self, relative_path: str, content: str) -> int:
        encoded_size = len(content.encode("utf-8"))
        if encoded_size > self.max_file_size_bytes:
            raise ValueError(f"Content too large ({encoded_size} bytes > {self.max_file_size_bytes} byte limit)")
        target = self.resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return encoded_size
