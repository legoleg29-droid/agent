from orchestrator.agents._common import LLMAgent


class CodingAgent(LLMAgent):
    id = "coding_agent"
    name = "Coding Agent"
    description = "Writes or modifies code to satisfy a technical objective."
    capabilities = ["coding", "code_generation"]
    available_tools: list[str] = []
    system_instructions = (
        "You are a senior software engineer. Given an objective, produce correct, "
        "minimal, well-structured code with a brief explanation. Prefer clarity over "
        "cleverness. Use fenced code blocks with the appropriate language tag."
    )
