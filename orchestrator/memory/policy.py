"""Memory policy: the explicit gate between "something happened" and
"something gets remembered".

Nothing is stored automatically just because a task produced output -
``MemoryPolicy.evaluate()`` decides what should be stored, what shouldn't,
its importance, and its scope. Sensitive-looking content is refused
outright rather than merely redacted, matching the spec example: a
temporary tool output stays execution-scoped, an important project
decision graduates to project scope, and a sensitive credential is never
stored at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from orchestrator.memory.models import MemoryScope, MemoryType
from orchestrator.security.redaction import contains_sensitive_value

# Baseline importance per type - a human/agent-provided importance_hint
# always takes precedence when given.
_DEFAULT_IMPORTANCE: dict[MemoryType, float] = {
    MemoryType.FACT: 0.6,
    MemoryType.DECISION: 0.8,
    MemoryType.PREFERENCE: 0.7,
    MemoryType.TASK_RESULT: 0.4,
    MemoryType.TOOL_RESULT: 0.2,
    MemoryType.SUMMARY: 0.6,
    MemoryType.ARTIFACT: 0.5,
    MemoryType.ERROR: 0.3,
    MemoryType.OBSERVATION: 0.3,
}

# Types that represent durable, cross-run knowledge default to a wider
# scope than plain per-execution scratch data.
_DEFAULT_SCOPE: dict[MemoryType, MemoryScope] = {
    MemoryType.DECISION: MemoryScope.PROJECT,
    MemoryType.PREFERENCE: MemoryScope.USER,
}

# Below this importance, an item is judged not durable enough for
# long-term memory (it can still live in short-term memory for the
# current execution).
DEFAULT_MIN_IMPORTANCE_TO_PERSIST = 0.4


@dataclass
class MemoryDecision:
    should_store: bool
    scope: MemoryScope
    importance: float
    reason: str


class MemoryPolicy:
    def __init__(self, *, min_importance_to_persist: float = DEFAULT_MIN_IMPORTANCE_TO_PERSIST) -> None:
        self.min_importance_to_persist = min_importance_to_persist

    def evaluate(
        self,
        *,
        type: MemoryType,
        content: Any,
        scope_hint: MemoryScope | None = None,
        importance_hint: float | None = None,
    ) -> MemoryDecision:
        text = content if isinstance(content, str) else json.dumps(content, default=str)

        if contains_sensitive_value(content) or contains_sensitive_value(text):
            return MemoryDecision(
                should_store=False,
                scope=scope_hint or MemoryScope.EXECUTION,
                importance=0.0,
                reason="content looks like it contains a credential/secret - never stored",
            )

        importance = importance_hint if importance_hint is not None else _DEFAULT_IMPORTANCE.get(type, 0.5)
        importance = max(0.0, min(1.0, importance))
        scope = scope_hint or _DEFAULT_SCOPE.get(type, MemoryScope.EXECUTION)

        should_store = importance >= self.min_importance_to_persist
        reason = (
            f"importance {importance:.2f} >= threshold {self.min_importance_to_persist:.2f}"
            if should_store
            else f"importance {importance:.2f} below threshold {self.min_importance_to_persist:.2f} - not persisted long-term"
        )
        return MemoryDecision(should_store=should_store, scope=scope, importance=importance, reason=reason)
