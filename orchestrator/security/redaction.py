"""Sensitive-data filtering shared by state persistence and memory.

Applied before anything is written to a durable store (checkpoints,
short-term memory, long-term memory) - never after. The same key-name
heuristic used for tool-call log redaction in Phase 2
(``orchestrator/tools/runtime.py``) is centralized here so every
persistence path uses one definition of "looks sensitive".
"""

from __future__ import annotations

import re
from typing import Any

_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "apikey",
    "password",
    "passwd",
    "token",
    "secret",
    "credential",
    "authorization",
    "private_key",
    "access_key",
)
_REDACTED = "<redacted>"

# Heuristics for sensitive-looking *values* even under an innocuous key name,
# e.g. a stray "sk-ant-..." or "Bearer ..." token pasted into free text.
_VALUE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{10,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-.=]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id shape
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
]


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS)


def redact_text(text: str) -> str:
    redacted = text
    for pattern in _VALUE_PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    return redacted


def contains_sensitive_value(value: Any) -> bool:
    """True if ``value`` (str, dict, list, ...) appears to contain a
    credential/secret. Used by MemoryPolicy to refuse creating a memory
    entry at all - stronger than ``redact_sensitive``, which scrubs and
    keeps (used for state checkpoints, which must remain resumable)."""
    if isinstance(value, dict):
        return any(_is_sensitive_key(str(k)) or contains_sensitive_value(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_sensitive_value(v) for v in value)
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in _VALUE_PATTERNS)
    return False


def redact_sensitive(value: Any) -> Any:
    """Recursively redact a JSON-like structure (dict/list/str/scalars).

    Dict values whose key looks like a secret are fully redacted; string
    values (dict or not) are additionally scanned for secret-shaped
    substrings. Never mutates the input.
    """
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _is_sensitive_key(str(k)) else redact_sensitive(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(v) for v in value)
    if isinstance(value, str):
        return redact_text(value)
    return value
