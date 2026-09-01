"""MemoryManager: the facade the rest of the orchestrator talks to.

Combines short-term memory (this execution only), long-term memory
(pluggable backend), and the memory policy (what's worth keeping) behind
four calls: ``store``, ``search``, ``get``, ``delete``. Nothing is stored
automatically - every call to ``store`` runs through ``MemoryPolicy``
first, and every ``search`` requires a scope filter.
"""

from __future__ import annotations

from orchestrator.core.logging_utils import EventLog
from orchestrator.memory.long_term import LongTermMemory
from orchestrator.memory.models import MemoryEntry, MemoryQuery, MemoryScope, MemoryType
from orchestrator.memory.policy import MemoryPolicy
from orchestrator.memory.short_term import ShortTermMemory


class MemoryManager:
    def __init__(
        self,
        execution_id: str,
        long_term: LongTermMemory,
        *,
        policy: MemoryPolicy | None = None,
        short_term: ShortTermMemory | None = None,
        event_log: EventLog | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self.execution_id = execution_id
        self.long_term = long_term
        self.policy = policy or MemoryPolicy()
        self.short_term = short_term or ShortTermMemory(execution_id)
        self.event_log = event_log
        # Default scope identifiers stamped onto entries created by this
        # manager - keeps a single execution from ever writing another
        # user's/session's/project's memory by omission.
        self.session_id = session_id
        self.project_id = project_id
        self.user_id = user_id

    def store(
        self,
        *,
        type: MemoryType,
        content: object,
        source: str,
        task_id: str | None = None,
        agent_id: str | None = None,
        scope_hint: MemoryScope | None = None,
        importance_hint: float | None = None,
        short_term_only: bool = False,
    ) -> MemoryEntry | None:
        decision = self.policy.evaluate(
            type=type, content=content, scope_hint=scope_hint, importance_hint=importance_hint
        )
        if decision.importance == 0.0 and "credential" in decision.reason:
            self._emit("MEMORY_SKIPPED", type, source, decision.reason)
            return None

        entry = MemoryEntry.create(
            type=type,
            content=content,
            source=source,
            scope=decision.scope,
            importance=decision.importance,
            execution_id=self.execution_id,
            session_id=self.session_id,
            project_id=self.project_id,
            user_id=self.user_id,
            task_id=task_id,
            agent_id=agent_id,
        )
        self.short_term.add(entry)

        if short_term_only or not decision.should_store:
            self._emit("MEMORY_SKIPPED", type, source, decision.reason + " (kept in short-term only)")
            return entry

        self.long_term.store(entry)
        self._emit("MEMORY_STORED", type, source, decision.reason, memory_id=entry.memory_id, scope=entry.scope.value)
        return entry

    def search(self, query: MemoryQuery) -> list[MemoryEntry]:
        results = self.long_term.search(query)
        if self.event_log:
            self.event_log.emit(
                "MEMORY_RETRIEVED",
                f"Retrieved {len(results)} memory entr{'y' if len(results) == 1 else 'ies'}",
                extra={
                    "count": len(results),
                    "filters": {
                        k: v
                        for k, v in {
                            "execution_id": query.execution_id,
                            "session_id": query.session_id,
                            "project_id": query.project_id,
                            "user_id": query.user_id,
                            "task_id": query.task_id,
                            "type": query.type.value if query.type else None,
                        }.items()
                        if v is not None
                    },
                },
            )
        return results

    def get(self, memory_id: str) -> MemoryEntry | None:
        return self.long_term.get(memory_id)

    def delete(self, memory_id: str) -> None:
        self.long_term.delete(memory_id)

    def _emit(self, tag: str, type: MemoryType, source: str, reason: str, **extra) -> None:
        if not self.event_log:
            return
        # Never log memory *content* - only its type/source/decision, per
        # the "never log sensitive memory contents" observability rule.
        self.event_log.emit(
            tag,
            f"Memory[{type.value}] from '{source}': {reason}",
            extra={"memory_type": type.value, "source": source, **extra},
        )
