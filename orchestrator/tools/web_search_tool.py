"""Example web-search tool.

This is a deliberately minimal placeholder: it searches a small local
in-memory corpus so the orchestrator is runnable offline and in tests
without external API keys. In production, swap the ``_search`` method (or
register a different tool under the same id) to call a real search API
or an MCP-exposed search server - agents and the orchestrator do not need
to change, since they only depend on the ``BaseTool`` interface.
"""

from __future__ import annotations

from orchestrator.tools.base import BaseTool, ToolResult
from orchestrator.tools.permissions import EXTERNAL_NETWORK

_DEMO_CORPUS = [
    {
        "title": "Acme Corp raises Series C to expand cloud analytics platform",
        "snippet": (
            "Acme Corp announced a $60M Series C to grow its real-time analytics "
            "product, competing with Globex and Initech in the mid-market BI space."
        ),
    },
    {
        "title": "Globex launches AI-powered dashboard builder",
        "snippet": (
            "Globex's new no-code dashboard builder targets small business analytics "
            "teams with drag-and-drop AI report generation."
        ),
    },
    {
        "title": "Initech pivots to vertical SaaS for logistics analytics",
        "snippet": (
            "Initech announced a strategic pivot away from general BI tooling toward "
            "logistics-specific analytics, citing crowded horizontal competition."
        ),
    },
    {
        "title": "State of the BI market 2026",
        "snippet": (
            "Analysts note increasing price pressure in business intelligence tooling "
            "as AI-native entrants lower the cost of building custom dashboards."
        ),
    },
]


class WebSearchTool(BaseTool):
    id = "web_search"
    name = "Web Search"
    description = (
        "Searches for information relevant to a query and returns matching snippets. "
        "Example/demo implementation backed by a small local corpus - replace with a "
        "real search API or MCP tool in production."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "results": {"type": "array"},
        },
        "required": ["query", "results"],
    }
    permissions = [EXTERNAL_NETWORK]  # represents "would call an external search API" in production
    capabilities = ["search", "research"]
    timeout_seconds = 15.0

    async def execute(self, *, query: str, max_results: int = 3) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(success=False, error="query must be non-empty")

        query_terms = {t.lower() for t in query.split() if len(t) > 2}
        scored = []
        for doc in _DEMO_CORPUS:
            haystack = f"{doc['title']} {doc['snippet']}".lower()
            score = sum(1 for term in query_terms if term in haystack)
            scored.append((score, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        results = [doc for score, doc in scored[:max_results]]
        if not any(score > 0 for score, _ in scored):
            results = _DEMO_CORPUS[:max_results]

        return ToolResult(success=True, output={"query": query, "results": results})
