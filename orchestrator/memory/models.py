"""Memory schemas: explicit types, scopes, and structured entries.

Nothing here is "plain text dumped into a blob" - every stored item has a
declared ``MemoryType`` and ``MemoryScope``, plus enough provenance
(source/task_id/agent_id/execution_id) to filter it back out later without
scanning everything.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    FACT = "fact"
    DECISION = "decision"
    PREFERENCE = "preference"
    TASK_RESULT = "task_result"
    TOOL_RESULT = "tool_result"
    SUMMARY = "summary"
    ARTIFACT = "artifact"
    ERROR = "error"
    OBSERVATION = "observation"


class MemoryScope(str, Enum):
    """Isolation boundary for a memory entry. Search always requires at
    least one scope-identifying filter (see ``MemoryQuery``) so one
    user's/execution's memory can never be returned for another's query
    by accident."""

    EXECUTION = "execution"
    SESSION = "session"
    PROJECT = "project"
    USER = "user"


@dataclass
class MemoryEntry:
    memory_id: str
    type: MemoryType
    content: Any  # structured payload - a dict for most types, short str for summary/fact text
    source: str  # e.g. "task:research_competitors", "agent:analysis_agent", "user"
    scope: MemoryScope = MemoryScope.EXECUTION
    execution_id: str | None = None
    session_id: str | None = None
    project_id: str | None = None
    user_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    importance: float = 0.5  # 0.0-1.0
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        type: MemoryType,
        content: Any,
        source: str,
        scope: MemoryScope = MemoryScope.EXECUTION,
        importance: float = 0.5,
        **scope_ids: Any,
    ) -> MemoryEntry:
        return cls(
            memory_id=f"mem_{uuid.uuid4().hex[:12]}",
            type=type,
            content=content,
            source=source,
            scope=scope,
            importance=importance,
            **scope_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "type": self.type.value,
            "content": self.content,
            "source": self.source,
            "scope": self.scope.value,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(
            memory_id=data["memory_id"],
            type=MemoryType(data["type"]),
            content=data["content"],
            source=data["source"],
            scope=MemoryScope(data.get("scope", MemoryScope.EXECUTION.value)),
            execution_id=data.get("execution_id"),
            session_id=data.get("session_id"),
            project_id=data.get("project_id"),
            user_id=data.get("user_id"),
            task_id=data.get("task_id"),
            agent_id=data.get("agent_id"),
            importance=data.get("importance", 0.5),
            timestamp=data.get("timestamp", time.time()),
            tags=list(data.get("tags", [])),
        )


@dataclass
class MemoryQuery:
    """Search filter. At least one scope-identifying field is required -
    enforced by ``LongTermMemory.search`` implementations and by
    ``MemoryManager`` - so a query can never silently span scopes/users."""

    execution_id: str | None = None
    session_id: str | None = None
    project_id: str | None = None
    user_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    type: MemoryType | None = None
    scope: MemoryScope | None = None
    min_importance: float | None = None
    since: float | None = None
    until: float | None = None
    limit: int = 20

    def has_scope_filter(self) -> bool:
        return any((self.execution_id, self.session_id, self.project_id, self.user_id))
