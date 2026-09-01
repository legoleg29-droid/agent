"""Plan versioning.

Every plan (initial or replanned) gets a ``PlanVersion`` record: never
overwritten, only appended to ``ExecutionState.plan_versions``. Replanning
bumps the version and links back to its parent so the full history of how
a plan evolved is reconstructable.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanVersion:
    plan_id: str
    version: int
    parent_plan_id: str | None
    created_at: float
    change_reason: str
    graph_snapshot: dict[str, Any]
    patch_ops: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def initial(cls, graph_snapshot: dict[str, Any], *, change_reason: str = "initial plan") -> PlanVersion:
        return cls(
            plan_id=f"plan_{uuid.uuid4().hex[:10]}",
            version=1,
            parent_plan_id=None,
            created_at=time.time(),
            change_reason=change_reason,
            graph_snapshot=graph_snapshot,
        )

    def next_version(
        self, graph_snapshot: dict[str, Any], *, change_reason: str, patch_ops: list[dict[str, Any]] | None = None
    ) -> PlanVersion:
        return PlanVersion(
            plan_id=f"plan_{uuid.uuid4().hex[:10]}",
            version=self.version + 1,
            parent_plan_id=self.plan_id,
            created_at=time.time(),
            change_reason=change_reason,
            graph_snapshot=graph_snapshot,
            patch_ops=patch_ops or [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "version": self.version,
            "parent_plan_id": self.parent_plan_id,
            "created_at": self.created_at,
            "change_reason": self.change_reason,
            "graph_snapshot": self.graph_snapshot,
            "patch_ops": self.patch_ops,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanVersion:
        return cls(
            plan_id=data["plan_id"],
            version=data["version"],
            parent_plan_id=data.get("parent_plan_id"),
            created_at=data["created_at"],
            change_reason=data.get("change_reason", ""),
            graph_snapshot=data["graph_snapshot"],
            patch_ops=list(data.get("patch_ops", []) or []),
        )
