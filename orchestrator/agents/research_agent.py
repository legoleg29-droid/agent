from orchestrator.agents._common import LLMAgent


class ResearchAgent(LLMAgent):
    id = "research_agent"
    name = "Research Agent"
    description = "Gathers information relevant to a topic using search tools."
    capabilities = ["research", "information_gathering"]
    available_tools = ["web_search"]
    system_instructions = (
        "You are a research analyst. Given an objective, gather relevant facts and "
        "summarize them clearly and concisely, citing sources by title when you use "
        "search results. Do not speculate beyond what you find; note uncertainty "
        "explicitly."
    )
