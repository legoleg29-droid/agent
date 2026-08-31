from orchestrator.agents._common import LLMAgent
from orchestrator.tools.permissions import EXTERNAL_NETWORK


class ResearchAgent(LLMAgent):
    id = "research_agent"
    name = "Research Agent"
    description = "Gathers information relevant to a topic using search tools."
    capabilities = ["research", "information_gathering"]
    available_tools = ["web_search"]
    permissions = [EXTERNAL_NETWORK]
    system_instructions = (
        "You are a research analyst. Given an objective, use the web_search tool to "
        "gather relevant facts, then summarize them clearly and concisely, citing "
        "sources by title. Do not speculate beyond what you find; note uncertainty "
        "explicitly."
    )
