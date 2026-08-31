from orchestrator.agents._common import LLMAgent
from orchestrator.tools.permissions import FILESYSTEM_READ, FILESYSTEM_WRITE


class CodingAgent(LLMAgent):
    id = "coding_agent"
    name = "Coding Agent"
    description = "Writes or modifies code to satisfy a technical objective."
    capabilities = ["coding", "code_generation"]
    available_tools = ["file_read", "file_write", "list_files"]
    permissions = [FILESYSTEM_READ, FILESYSTEM_WRITE]
    system_instructions = (
        "You are a senior software engineer. Given an objective, produce correct, "
        "minimal, well-structured code with a brief explanation. Prefer clarity over "
        "cleverness. Use fenced code blocks with the appropriate language tag. Use the "
        "file_read/file_write/list_files tools if the objective requires reading or "
        "producing files - all paths are relative to a sandboxed working directory."
    )
