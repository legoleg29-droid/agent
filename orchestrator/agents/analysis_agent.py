from orchestrator.agents._common import LLMAgent


class AnalysisAgent(LLMAgent):
    id = "analysis_agent"
    name = "Analysis Agent"
    description = "Analyzes gathered information to extract insights, patterns and comparisons."
    capabilities = ["analysis", "evaluation"]
    available_tools = ["calculator"]
    system_instructions = (
        "You are a business/data analyst. Given an objective and any prior task outputs, "
        "identify key patterns, comparisons, risks and opportunities. Use the calculator "
        "tool for any arithmetic. Structure your analysis with clear headings or bullet "
        "points."
    )
