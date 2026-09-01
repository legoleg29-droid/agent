"""Execution state persistence.

Pluggable, like everything else in this codebase: the orchestrator depends
on the small ``StateStore`` interface, not a concrete database, so SQLite
(the local-dev default here) can be swapped for Postgres or anything else
without touching orchestration code.

``START -> SAVE STATE -> EXECUTE -> SAVE STATE -> CRASH/STOP -> RESTART ->
RESUME`` only works if every save is a complete, atomic snapshot - each
``save()`` call here runs inside a SQLite transaction so a crash mid-write
can never leave a half-written execution on disk.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from orchestrator.core.logging_utils import EventLog
from orchestrator.security.redaction import redact_sensitive
from orchestrator.state.models import ExecutionState


class ExecutionNotFoundError(KeyError):
    pass


class StateStore(ABC):
    @abstractmethod
    def save(self, state: ExecutionState) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, execution_id: str) -> ExecutionState | None:
        raise NotImplementedError

    @abstractmethod
    def list_executions(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, execution_id: str) -> None:
        raise NotImplementedError

    def checkpoint(self, state: ExecutionState, event_log: EventLog | None, reason: str) -> None:
        """Persist ``state`` and emit a single structured CHECKPOINT event.
        This is the one call site the orchestrator uses at every lifecycle
        transition it must survive a crash across."""
        self.save(state)
        if event_log:
            event_log.emit(
                "CHECKPOINT",
                f"Checkpointed execution '{state.execution_id}' ({reason})",
                task_id=state.active_task,
                status=state.status.value,
                extra={"reason": reason, "execution_id": state.execution_id},
            )


class InMemoryStateStore(StateStore):
    """Process-local store - no durability across restarts. Useful for
    tests and for callers that don't need crash recovery."""

    def __init__(self) -> None:
        self._states: dict[str, ExecutionState] = {}

    def save(self, state: ExecutionState) -> None:
        # Store a snapshot (round-tripped through to/from_dict) so mutating
        # the live object afterwards can't silently corrupt the "saved" copy.
        self._states[state.execution_id] = ExecutionState.from_dict(redact_sensitive(state.to_dict()))

    def load(self, execution_id: str) -> ExecutionState | None:
        state = self._states.get(execution_id)
        return ExecutionState.from_dict(state.to_dict()) if state else None

    def list_executions(self) -> list[str]:
        return list(self._states.keys())

    def delete(self, execution_id: str) -> None:
        self._states.pop(execution_id, None)


class SQLiteStateStore(StateStore):
    """Default local-development persistence backend.

    One row per execution, the full state as a redacted JSON blob plus a
    few indexed columns for quick lookups. A per-instance lock serializes
    writes from this process; the transaction itself protects against a
    torn write on crash.
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
                CREATE TABLE IF NOT EXISTS execution_state (
                    execution_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    user_goal TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def save(self, state: ExecutionState) -> None:
        payload = json.dumps(redact_sensitive(state.to_dict()))
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO execution_state (execution_id, status, user_goal, data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    status=excluded.status, user_goal=excluded.user_goal,
                    data=excluded.data, updated_at=excluded.updated_at
                """,
                (state.execution_id, state.status.value, state.user_goal, payload, state.created_at, state.updated_at),
            )

    def load(self, execution_id: str) -> ExecutionState | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM execution_state WHERE execution_id = ?", (execution_id,)
            ).fetchone()
        if row is None:
            return None
        return ExecutionState.from_dict(json.loads(row[0]))

    def list_executions(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT execution_id FROM execution_state ORDER BY updated_at DESC").fetchall()
        return [r[0] for r in rows]

    def delete(self, execution_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM execution_state WHERE execution_id = ?", (execution_id,))
