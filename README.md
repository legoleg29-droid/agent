# AI Agent Orchestrator

A production-oriented, **framework-free** AI agent orchestrator. It takes a
high-level user goal, decomposes it into a dependency graph of tasks,
routes each task to a capability-matched agent, executes the graph
(respecting dependencies, retrying and re-planning where needed), and
synthesizes a final result.

No LangChain, LangGraph, CrewAI, AutoGen, or similar. The planning loop,
scheduler, routing, retry/recovery, and context management are all
implemented directly in this repository. [Claude](https://www.anthropic.com)
(via the official `anthropic` SDK) is the default LLM provider, behind an
abstraction that lets other providers be added without touching
orchestration code.

## Architecture

```
Orchestrator (orchestrator/core/orchestrator.py)
├── Planner            core/planner.py       - goal -> Task DAG (via LLM)
├── Task Graph          core/task_graph.py    - DAG, dependency resolution
├── Agent Router         core/router.py        - capability -> agent (no hardcoding)
├── Agent Registry        agents/registry.py    - capability-indexed agent lookup
├── Agent Runtime           agents/base.py + agents/*.py - agent execution
├── Tool Registry            tools/registry.py     - available tools
├── Tool Runtime               tools/runtime.py      - tool invocation + logging
├── Context Manager              core/context.py       - global/task/agent context
├── State Manager                  core/state.py         - run status, replan budget
├── Evaluator                        core/evaluator.py     - independent result judging
├── Retry / Recovery                   core/retry.py          - verdict -> action
└── Final Result Synthesizer             core/synthesizer.py    - combines outputs
```

### Execution loop

```
INSPECT -> PLAN -> ROUTE -> EXECUTE -> OBSERVE -> EVALUATE
       -> REPLAN or CONTINUE -> COMPLETE
```

1. **INSPECT** - the orchestrator gathers the currently registered agent
   capabilities and tools (nothing hardcoded).
2. **PLAN** - the `Planner` asks the LLM to decompose the goal into a task
   DAG, constrained to those real capabilities/tools, and validates the
   result (unknown capability/tool/dependency, cycles -> rejected).
3. **ROUTE** - for each ready task, the `AgentRouter` asks the
   `AgentRegistry` which agents declare the required capability and picks
   one (preferring a candidate that also covers the task's required
   tools).
4. **EXECUTE** - the chosen agent runs, optionally requesting a tool via
   the `ToolRuntime`.
5. **OBSERVE** - duration, tokens, tool calls, model, and status are
   logged as structured events.
6. **EVALUATE** - the `Evaluator` independently judges the result
   (success / partial success / failure / retry required / replan
   required) - it never blindly trusts the agent's own success flag.
7. **REPLAN or CONTINUE** - the `RetryPolicy` turns that verdict into an
   action: continue, retry (bounded by `max_retries` per task), replan
   (bounded by `max_replans` per run, via the `Planner.replan` path which
   splices new tasks into the live graph), or abort safely.
8. **COMPLETE** - once the graph is fully resolved (succeeded / failed /
   skipped), the `FinalResultSynthesizer` combines all successful task
   outputs into one final answer.

Independent tasks (no dependency relationship) are scheduled together and
executed concurrently via `asyncio.gather`.

### Context isolation

`ContextManager` deliberately does **not** hand every agent the full run
history. Each task only receives:
- its own objective / expected output (task context),
- the outputs of its *direct* dependencies (`upstream_outputs_for`), not
  siblings' outputs or the whole graph,
- run-level `global_context` metadata separately from task outputs.

### Routing without hardcoding

The orchestrator never contains `if task == "research": use ResearchAgent()`.
Every agent declares `capabilities: list[str]`; the `AgentRegistry` indexes
agents by capability, and the `AgentRouter` looks up candidates for a
task's required capability, preferring one that also covers the task's
`required_tools`. Adding a new agent (see below) never requires editing
the orchestrator.

## Installation

Requires Python 3.11+.

```bash
pip install -r requirements.txt        # runtime deps
pip install -r requirements-dev.txt    # + pytest for running tests
cp .env.example .env                   # then fill in ANTHROPIC_API_KEY
```

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (for real runs) | - | Claude API key |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-5-20250929` | Claude model id |
| `ORCHESTRATOR_MAX_RETRIES_PER_TASK` | No | `2` | Per-task retry budget |
| `ORCHESTRATOR_MAX_REPLANS` | No | `2` | Per-run re-plan budget |

No secrets are hardcoded anywhere in the codebase; `ClaudeProvider` reads
`ANTHROPIC_API_KEY` from the environment and raises immediately if it's
missing.

## Running the orchestrator

Against real Claude:

```bash
python main.py "Research competitors, analyze them, and create a strategy."
```

Offline, with no API key, using the deterministic `MockProvider` (useful
for demos/CI-adjacent smoke checks - it scripts a plausible
research -> analysis -> synthesis run without any network calls):

```bash
python main.py --mock "Research competitors, analyze them, and create a strategy."
```

Example output (`--mock`):

```
[ORCHESTRATOR] Received goal: '...'
[PLANNER] Generated plan with 3 task(s) for goal: '...'
[ROUTER] Routed task 'gather_information' (capability=research) to agent 'research_agent'
[TASK] Starting task 'gather_information' ...
[TOOL] web_search invoked ...
[AGENT] Agent 'research_agent' finished task 'gather_information' ...
[EVALUATOR] Task 'gather_information' evaluated as success: ...
...
[COMPLETE] Final result synthesized (status=succeeded)

TASK GRAPH
  [SUCCEEDED ] gather_information (agent=research_agent, retries=0)
  [SUCCEEDED ] analyze_information (agent=analysis_agent, retries=0)
  [SUCCEEDED ] produce_final_deliverable (agent=writer_agent, retries=0)

FINAL RESULT
Summary: research identified the competitive landscape, analysis surfaced
an underserved mid-market AI-native gap, and the resulting strategy is to
enter there with a price- and speed-led offering.
```

### Programmatic usage

```python
import asyncio
from orchestrator.agents.registry import AgentRegistry
from orchestrator.agents.research_agent import ResearchAgent
from orchestrator.agents.analysis_agent import AnalysisAgent
from orchestrator.agents.writer_agent import WriterAgent
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.providers.claude_provider import ClaudeProvider
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.web_search_tool import WebSearchTool
from orchestrator.tools.calculator_tool import CalculatorTool

async def main():
    provider = ClaudeProvider()  # reads ANTHROPIC_API_KEY from env

    tools = ToolRegistry()
    tools.register(WebSearchTool())
    tools.register(CalculatorTool())

    agents = AgentRegistry()
    orchestrator = Orchestrator(provider, agents, tools)
    agents.register(ResearchAgent(provider, orchestrator.tool_runtime))
    agents.register(AnalysisAgent(provider, orchestrator.tool_runtime))
    agents.register(WriterAgent(provider, orchestrator.tool_runtime))

    result = await orchestrator.run("Research competitors, analyze them, and create a strategy.")
    print(result.final_output)

asyncio.run(main())
```

## Observability

Every lifecycle event is emitted through `EventLog` (`orchestrator/core/logging_utils.py`)
with one of these tags: `ORCHESTRATOR`, `PLANNER`, `ROUTER`, `TASK`, `AGENT`,
`TOOL`, `EVALUATOR`, `RETRY`, `REPLAN`, `COMPLETE`. Each event is both
printed as `[TAG] message (field=value, ...)` and stored as a structured
dict (`OrchestrationResult.events`) carrying task id, agent id, status,
duration, retry count, tokens used, model, and tool call count where
applicable.

## Creating a new agent

1. Subclass `orchestrator.agents.base.BaseAgent` (or the convenience
   `orchestrator.agents._common.LLMAgent`, which wires up prompting and a
   simple tool-request protocol for you).
2. Declare `id`, `name`, `description`, `capabilities`, `available_tools`,
   `system_instructions`.
3. Implement `execute(self, agent_input: AgentInput) -> AgentOutput` (or
   just rely on `LLMAgent`'s implementation).
4. Register an instance with the `AgentRegistry` before calling
   `orchestrator.run(...)`.

The planner and router pick it up automatically via its declared
capabilities - no orchestrator code changes needed.

```python
from orchestrator.agents._common import LLMAgent

class LegalReviewAgent(LLMAgent):
    id = "legal_review_agent"
    name = "Legal Review Agent"
    description = "Reviews text for legal/compliance risk."
    capabilities = ["legal_review"]
    available_tools: list[str] = []
    system_instructions = "You are a contracts and compliance reviewer..."
```

## Creating a new tool

1. Subclass `orchestrator.tools.base.BaseTool`, set `name`, `description`,
   `input_schema`, and implement `async execute(self, **kwargs) -> ToolResult`.
2. Register an instance with the `ToolRegistry`.
3. Add its `name` to any agent's `available_tools` that should be able to
   request it.

```python
from orchestrator.tools.base import BaseTool, ToolResult

class MyApiTool(BaseTool):
    name = "my_api"
    description = "Calls my internal API."
    input_schema = {"type": "object", "properties": {"query": {"type": "string"}}}

    async def execute(self, *, query: str) -> ToolResult:
        ...
        return ToolResult(success=True, output=...)
```

The `ToolRuntime` is the single seam agents go through to call tools, so
swapping a demo tool for a real API - or for an MCP-backed tool server -
never requires touching agent or orchestrator code.

## Adding another LLM provider

1. Subclass `orchestrator.providers.base.LLMProvider` and implement
   `async complete(self, *, system, messages, max_tokens=None, temperature=None) -> LLMResponse`.
2. Pass an instance of it to `Orchestrator(...)` and to your agents -
   nothing else in the codebase references `ClaudeProvider` directly.

```python
from orchestrator.providers.base import LLMProvider, LLMResponse

class OpenAIProvider(LLMProvider):
    name = "openai"
    async def complete(self, *, system, messages, max_tokens=None, temperature=None) -> LLMResponse:
        ...
```

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite (45 tests, all offline/deterministic via `MockProvider` and
in-process test doubles) covers:

- Planner (plan generation, validation, rejection of bad output, replanning)
- Agent registry (registration, capability lookup, specialist ranking)
- Agent routing (capability match, tool-covering preference, no-match error)
- Dependency resolution (readiness, cycles, skip cascades, topological order)
- Task execution (dependency-ordered + parallel execution, context propagation)
- Retry behavior (retry-then-replan-then-abort budget enforcement)
- Evaluation (independent judging - never trusts the agent's own success flag)
- Re-planning (triggered on persistent failure, budgeted, splices in a new plan)
- Context isolation (agents only see their own task + direct dependency outputs)
- Full end-to-end orchestration (`tests/test_e2e.py`, matching the spec's
  "research -> analyze -> strategize" example, using the real `ResearchAgent`,
  `AnalysisAgent`, `WriterAgent` and real tools against a scripted provider)

## Repository layout

```
orchestrator/
  core/        orchestrator loop, planner, router, task graph, context,
               state, evaluator, retry, synthesizer, logging
  agents/      BaseAgent, AgentRegistry, ResearchAgent, AnalysisAgent,
               CodingAgent, WriterAgent
  tools/       BaseTool, ToolRegistry, ToolRuntime, CalculatorTool,
               WebSearchTool
  providers/   LLMProvider, ClaudeProvider, MockProvider
tests/         pytest suite (see above)
main.py        CLI entry point
.env.example   environment variable template
```

## Known limitations

- The example `WebSearchTool` is backed by a small local demo corpus, not
  a real search API - swap its implementation (or register a different
  tool under the same name) for production use; the tool abstraction is
  designed so that requires no agent/orchestrator changes.
- The tool-use protocol between an `LLMAgent` and the model is a simple
  single-round textual convention (`TOOL_CALL: name({...})`) rather than
  a vendor-specific function-calling schema, to stay provider-agnostic;
  it supports at most one tool call per agent turn today.
- The `Evaluator` is rule-based (empty/short output, refusal language,
  reported errors) rather than an LLM-as-judge; it's easy to extend with
  an LLM-backed check by giving `Evaluator` a provider, but the default
  keeps evaluation fast, deterministic, and cheap.
- MCP tool integration is not implemented, but the `ToolRuntime` seam is
  where it would plug in.
