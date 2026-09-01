"""Short-term memory for the current execution.

An in-process, execution-scoped, size-bounded buffer of structured
entries - recent task/tool results, decisions, constraints. It is never
persisted on its own (it dies with the process, by design - durable items
graduate to long-term memory via ``MemoryPolicy``) and it never stores raw
conversation history; only explicit, typed ``MemoryEntry`` records.
"""

from __future__ import annotations

from orchestrator.memory.models import MemoryEntry, MemoryType


class ShortTermMemory:
    def __init__(self, execution_id: str, *, max_entries: int = 200) -> None:
        self.execution_id = execution_id
        self.max_entries = max_entries
        self._entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        if entry.execution_id and entry.execution_id != self.execution_id:
            raise ValueError(
                f"Refusing to add entry scoped to execution '{entry.execution_id}' "
                f"into short-term memory for execution '{self.execution_id}'"
            )
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            # Drop the oldest, lowest-importance entries first rather than
            # blindly truncating from the front.
            self._entries.sort(key=lambda e: (e.importance, e.timestamp))
            self._entries = self._entries[len(self._entries) - self.max_entries :]
            self._entries.sort(key=lambda e: e.timestamp)

    def recent(self, *, limit: int = 20, type: MemoryType | None = None) -> list[MemoryEntry]:
        entries = self._entries if type is None else [e for e in self._entries if e.type == type]
        return list(reversed(entries[-limit:]))

    def for_task(self, task_id: str) -> list[MemoryEntry]:
        return [e for e in self._entries if e.task_id == task_id]

    def all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
