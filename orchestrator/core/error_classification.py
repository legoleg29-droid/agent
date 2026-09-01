"""Structured error classification.

Maps whatever went wrong (an exception, an agent-reported error string, or
an evaluation failure) onto one of a fixed set of categories, and each
category onto a default recovery action. This is a *default* mapping - the
Evaluator/RetryPolicy still layer task-specific signals (retry/repair
budgets, deterministic evidence) on top; it exists so "why did this fail"
is always a structured label, never just a free-text string to eyeball.
"""

from __future__ import annotations

import re
from enum import Enum


class ErrorCategory(str, Enum):
    TRANSIENT = "transient"
    TOOL_ERROR = "tool_error"
    VALIDATION_ERROR = "validation_error"
    OUTPUT_ERROR = "output_error"
    PERMISSION_ERROR = "permission_error"
    DEPENDENCY_ERROR = "dependency_error"
    MODEL_ERROR = "model_error"
    ENVIRONMENT_ERROR = "environment_error"
    LOGIC_ERROR = "logic_error"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    RETRY = "retry"
    REPAIR = "repair"
    REPLAN = "replan"
    FAIL = "fail"


# Default category -> recovery action. TRANSIENT/MODEL_ERROR are usually
# fixed by simply trying again; VALIDATION/OUTPUT/LOGIC errors mean the
# agent produced *something* that just needs fixing (repair); PERMISSION/
# DEPENDENCY/ENVIRONMENT errors mean the current approach can't work at all
# (replan); UNKNOWN defaults to the safest option that still makes
# progress (retry), letting the retry-budget/evaluator escalate it.
DEFAULT_RECOVERY: dict[ErrorCategory, RecoveryAction] = {
    ErrorCategory.TRANSIENT: RecoveryAction.RETRY,
    ErrorCategory.MODEL_ERROR: RecoveryAction.RETRY,
    ErrorCategory.TOOL_ERROR: RecoveryAction.RETRY,
    ErrorCategory.VALIDATION_ERROR: RecoveryAction.REPAIR,
    ErrorCategory.OUTPUT_ERROR: RecoveryAction.REPAIR,
    ErrorCategory.LOGIC_ERROR: RecoveryAction.REPAIR,
    ErrorCategory.PERMISSION_ERROR: RecoveryAction.REPLAN,
    ErrorCategory.DEPENDENCY_ERROR: RecoveryAction.REPLAN,
    ErrorCategory.ENVIRONMENT_ERROR: RecoveryAction.REPLAN,
    ErrorCategory.UNKNOWN: RecoveryAction.RETRY,
}

_TRANSIENT_PATTERNS = re.compile(
    r"\b(timeout|timed out|connection (reset|refused|error)|temporarily unavailable|"
    r"rate limit|too many requests|429|503|overloaded|try again)\b",
    re.IGNORECASE,
)
_PERMISSION_PATTERNS = re.compile(r"\b(permission denied|lacks permission|unauthorized|forbidden|403)\b", re.IGNORECASE)
_TOOL_PATTERNS = re.compile(
    r"\b(tool '.*' (is not registered|failed)|not registered in the toolregistry|tool_error|tool call failed)\b",
    re.IGNORECASE,
)
_DEPENDENCY_PATTERNS = re.compile(
    r"\b(does not declare required tool|depends on unknown task|dependency|blocked)\b", re.IGNORECASE
)
_VALIDATION_PATTERNS = re.compile(
    r"\b(invalid (arguments|json|schema)|json.*decode|schema validation|malformed|parse error)\b", re.IGNORECASE
)
_ENVIRONMENT_PATTERNS = re.compile(
    r"\b(no such file or directory|command not found|module not found|environment|not available)\b", re.IGNORECASE
)
_OUTPUT_PATTERNS = re.compile(r"\b(empty output|missing required|no content|did not produce)\b", re.IGNORECASE)

_TRANSIENT_EXCEPTIONS = ("TimeoutError", "ConnectionError", "ConnectionResetError", "OSError")


def classify_error(
    *,
    error_text: str | None = None,
    exception_type: str | None = None,
    failed_criteria: list[str] | None = None,
) -> ErrorCategory:
    """Best-effort classification from whatever signals are available.
    Never raises - falls back to UNKNOWN rather than guessing wrong
    confidently."""
    if exception_type and exception_type in _TRANSIENT_EXCEPTIONS:
        return ErrorCategory.TRANSIENT

    text = " ".join(filter(None, [error_text, *(failed_criteria or [])]))
    if not text:
        return ErrorCategory.UNKNOWN

    if _TRANSIENT_PATTERNS.search(text):
        return ErrorCategory.TRANSIENT
    if _PERMISSION_PATTERNS.search(text):
        return ErrorCategory.PERMISSION_ERROR
    if _DEPENDENCY_PATTERNS.search(text):
        return ErrorCategory.DEPENDENCY_ERROR
    if _TOOL_PATTERNS.search(text):
        return ErrorCategory.TOOL_ERROR
    if _VALIDATION_PATTERNS.search(text):
        return ErrorCategory.VALIDATION_ERROR
    if _ENVIRONMENT_PATTERNS.search(text):
        return ErrorCategory.ENVIRONMENT_ERROR
    if _OUTPUT_PATTERNS.search(text):
        return ErrorCategory.OUTPUT_ERROR
    return ErrorCategory.UNKNOWN


def default_recovery_action(category: ErrorCategory) -> RecoveryAction:
    return DEFAULT_RECOVERY[category]
