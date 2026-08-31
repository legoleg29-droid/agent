"""AI Agent Orchestrator.

A framework-free, in-house orchestration core for coordinating multiple
LLM-backed agents against a high-level user goal. See README.md for the
full architecture overview.
"""

from orchestrator.core.orchestrator import Orchestrator, OrchestrationResult

__all__ = ["Orchestrator", "OrchestrationResult"]
