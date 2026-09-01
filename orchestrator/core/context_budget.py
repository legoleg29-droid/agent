"""Context budgeting: priority-ordered assembly with graceful degradation.

Sections are added in priority order (lower number = higher priority) up
to a token budget. When a section doesn't fit, it is summarized
(truncated with an explicit "N tokens omitted" note) rather than silently
dropped, unless even a minimal summary wouldn't fit - only then is it
omitted, and that omission is recorded so callers/tests can see exactly
what was cut. Required sections (the current task) are never dropped or
summarized away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Rough, provider-agnostic token estimate (~4 chars/token), consistent
# with the heuristic MockProvider already uses elsewhere in this codebase.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN) if text else 0


@dataclass
class ContextSection:
    name: str
    text: str
    priority: int
    required: bool = False


@dataclass
class BudgetedContext:
    sections: dict[str, str]  # name -> final (possibly summarized) text, in priority order
    total_tokens: int
    budget_tokens: int
    truncated: list[str] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)

    def render(self) -> str:
        return "\n\n".join(text for text in self.sections.values() if text)


class ContextBudget:
    def __init__(self, max_tokens: int = 3000, *, min_summary_tokens: int = 30) -> None:
        self.max_tokens = max_tokens
        self.min_summary_tokens = min_summary_tokens

    def assemble(self, sections: list[ContextSection]) -> BudgetedContext:
        ordered = sorted((s for s in sections if s.text), key=lambda s: s.priority)
        included: dict[str, str] = {}
        truncated: list[str] = []
        omitted: list[str] = []
        used = 0

        for section in ordered:
            tokens = estimate_tokens(section.text)
            remaining = self.max_tokens - used

            if section.required or tokens <= max(remaining, 0):
                included[section.name] = section.text
                used += tokens
                continue

            if remaining >= self.min_summary_tokens:
                summary_chars = remaining * _CHARS_PER_TOKEN
                omitted_tokens = tokens - remaining
                summary = (
                    section.text[:summary_chars].rstrip()
                    + f"\n...[summarized: ~{omitted_tokens} tokens omitted for context budget]"
                )
                included[section.name] = summary
                used += estimate_tokens(summary)
                truncated.append(section.name)
            else:
                omitted.append(section.name)

        return BudgetedContext(sections=included, total_tokens=used, budget_tokens=self.max_tokens, truncated=truncated, omitted=omitted)
