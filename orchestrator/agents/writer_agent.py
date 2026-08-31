from orchestrator.agents._common import LLMAgent
from orchestrator.tools.permissions import FILESYSTEM_READ, FILESYSTEM_WRITE


class WriterAgent(LLMAgent):
    id = "writer_agent"
    name = "Writer Agent"
    description = "Synthesizes prior findings into clear, well-structured written output (reports, strategies, summaries)."
    capabilities = ["writing", "synthesis", "strategy"]
    available_tools = ["file_read", "file_write"]
    permissions = [FILESYSTEM_READ, FILESYSTEM_WRITE]
    system_instructions = (
        "You are a professional writer. Given an objective and prior task outputs, "
        "synthesize them into a coherent, well-organized piece of writing that "
        "directly satisfies the objective. Do not introduce facts that contradict "
        "the prior outputs. Use the file_write tool if the objective asks you to save "
        "the result to a file - paths are relative to a sandboxed working directory."
    )
