"""Pluggable long-term memory backends.

    LongTermMemory
    ├── InMemoryLongTermMemory   - tests / ephemeral runs
    ├── SQLiteLongTermMemory     - default local-development backend
    └── (future) VectorStoreLongTermMemory - same interface, semantic search

The core engine only ever depends on the ``LongTermMemory`` interface, so a
vector database can be added later purely as a new implementation of
``store``/``search``/``get``/``delete`` - no orchestrator changes required.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from orchestrator.memory.models import MemoryEntry, MemoryQuery
from orchestrator.security.redaction import redact_sensitive


class ScopeViolationError(ValueError):
    """Raised when a search/store would cross a scope/user boundary."""


class LongTermMemory(ABC):
    @abstractmethod
    def store(self, entry: MemoryEntry) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: MemoryQuery) -> list[MemoryEntry]:
        raise NotImplementedError

    @abstractmethod
    def get(self, memory_id: str) -> MemoryEntry | None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        raise NotImplementedError

    @staticmethod
    def _require_scope(query: MemoryQuery) -> None:
        if not query.has_scope_filter():
            raise ScopeViolationError(
                "MemoryQuery must specify at least one of execution_id/session_id/"
                "project_id/user_id - unscoped searches are refused to prevent "
                "cross-execution or cross-user memory leakage."
            )

    @staticmethod
    def _matches(entry: MemoryEntry, query: MemoryQuery) -> bool:
        if query.execution_id is not None and entry.execution_id != query.execution_id:
            return False
        if query.session_id is not None and entry.session_id != query.session_id:
            return False
        if query.project_id is not None and entry.project_id != query.project_id:
            return False
        if query.user_id is not None and entry.user_id != query.user_id:
            return False
        if query.task_id is not None and entry.task_id != query.task_id:
            return False
        if query.agent_id is not None and entry.agent_id != query.agent_id:
            return False
        if query.type is not None and entry.type != query.type:
            return False
        if query.scope is not None and entry.scope != query.scope:
            return False
        if query.min_importance is not None and entry.importance < query.min_importance:
            return False
        if query.since is not None and entry.timestamp < query.since:
            return False
        if query.until is not None and entry.timestamp > query.until:
            return False
        return True


class InMemoryLongTermMemory(LongTermMemory):
    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    def store(self, entry: MemoryEntry) -> None:
        self._entries[entry.memory_id] = MemoryEntry.from_dict(redact_sensitive(entry.to_dict()))

    def search(self, query: MemoryQuery) -> list[MemoryEntry]:
        self._require_scope(query)
        matches = [e for e in self._entries.values() if self._matches(e, query)]
        matches.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
        return matches[: query.limit]

    def get(self, memory_id: str) -> MemoryEntry | None:
        return self._entries.get(memory_id)

    def delete(self, memory_id: str) -> None:
        self._entries.pop(memory_id, None)


class SQLiteLongTermMemory(LongTermMemory):
    """Default local-development persistence backend for long-term memory.

    A `future vector store <VectorStoreLongTermMemory>` would implement the
    same four methods, likely backed by an embedding index for ``search``,
    without changing anything upstream of this interface.
    """

    def __init__(self, db_path: str | Path = "./orchestrator_state.db") -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    memory_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    execution_id TEXT,
                    session_id TEXT,
                    project_id TEXT,
                    user_id TEXT,
                    task_id TEXT,
                    agent_id TEXT,
                    importance REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    data TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_execution ON memory_entries(execution_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_session ON memory_entries(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_project ON memory_entries(project_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_user ON memory_entries(user_id)")

    def store(self, entry: MemoryEntry) -> None:
        redacted = redact_sensitive(entry.to_dict())
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_entries
                    (memory_id, type, scope, execution_id, session_id, project_id, user_id,
                     task_id, agent_id, importance, timestamp, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    redacted["memory_id"],
                    redacted["type"],
                    redacted["scope"],
                    redacted["execution_id"],
                    redacted["session_id"],
                    redacted["project_id"],
                    redacted["user_id"],
                    redacted["task_id"],
                    redacted["agent_id"],
                    redacted["importance"],
                    redacted["timestamp"],
                    json.dumps(redacted),
                ),
            )

    def search(self, query: MemoryQuery) -> list[MemoryEntry]:
        self._require_scope(query)
        clauses: list[str] = []
        params: list[object] = []
        for column, value in (
            ("execution_id", query.execution_id),
            ("session_id", query.session_id),
            ("project_id", query.project_id),
            ("user_id", query.user_id),
            ("task_id", query.task_id),
            ("agent_id", query.agent_id),
            ("type", query.type.value if query.type else None),
            ("scope", query.scope.value if query.scope else None),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if query.min_importance is not None:
            clauses.append("importance >= ?")
            params.append(query.min_importance)
        if query.since is not None:
            clauses.append("timestamp >= ?")
            params.append(query.since)
        if query.until is not None:
            clauses.append("timestamp <= ?")
            params.append(query.until)

        sql = "SELECT data FROM memory_entries"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
        params.append(query.limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [MemoryEntry.from_dict(json.loads(r[0])) for r in rows]

    def get(self, memory_id: str) -> MemoryEntry | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM memory_entries WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return MemoryEntry.from_dict(json.loads(row[0])) if row else None

    def delete(self, memory_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM memory_entries WHERE memory_id = ?", (memory_id,))
