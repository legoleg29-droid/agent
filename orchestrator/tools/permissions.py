"""Tool permission model.

Permissions are plain strings (e.g. ``"filesystem.read"``,
``"external_network"``, ``"database.delete"``). Tools declare which
permissions they require; agents declare which permissions they hold. The
``ToolRuntime`` is the enforcement point - see orchestrator/tools/runtime.py -
never the LLM prompt.
"""

from __future__ import annotations

# Well-known permission strings. Not exhaustive - a tool or agent may
# declare any string - but centralized here so callers share the same
# vocabulary for the common cases instead of inventing near-duplicates.
FILESYSTEM_READ = "filesystem.read"
FILESYSTEM_WRITE = "filesystem.write"
EXTERNAL_NETWORK = "external_network"
COMPUTE = "compute"
DATABASE_READ = "database.read"
DATABASE_WRITE = "database.write"
DATABASE_DELETE = "database.delete"


def missing_permissions(required: list[str], granted: list[str]) -> list[str]:
    """Permissions in ``required`` that are absent from ``granted``."""
    granted_set = set(granted)
    return [p for p in required if p not in granted_set]
