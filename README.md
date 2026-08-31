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
USER
  |
ORCHESTRATOR        core/orchestrator.py
  |
PLANNER             core/planner.py         - goal -> Task DAG (via LLM)
  |
TASK GRAPH          core/task_graph.py      - DAG, dependency resolution, scheduler
  |
AGENT ROUTER        core/router.py          - capability -> agent (no hardcoding)
  |
AGENT RUNTIME       agents/base.py + agents/*.py + agents/_common.py
  |                 (structured Claude tool-use loop)
TOOL RUNTIME        tools/runtime.py        - permission check, input/output
  |                                           validation, timeout, observability
MCP / NATIVE TOOLS / APIS
  |                 tools/registry.py, tools/mcp_adapter.py,
  |                 tools/file_tools.py, tools/calculator_tool.py, ...
RESULT
  |
EVALUATOR           core/evaluator.py       - independent result judging
  |
REPLAN / COMPLETE   core/retry.py, core/synthesizer.py
```

Supporting components: `Agent Registry` (`agents/registry.py`, capability-indexed
agent lookup), `Context Manager` (`core/context.py`, global/task/agent/tool-result
context), `State Manager` (`core/state.py`, run status and replan budget).

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
| `ORCHESTRATOR_SANDBOX_DIR` | No | `./sandbox` | Root directory for `file_read`/`file_write`/`list_files` - no path can escape it |
| `ORCHESTRATOR_TOOL_TIMEOUT_SECONDS` | No | `30` | Default per-tool execution timeout |

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
[TOOL_REQUEST] Requested tool 'web_search' ...
[TOOL_PERMISSION] Permission granted for tool 'web_search' ...
[TOOL_VALIDATION] Arguments valid for tool 'web_search' ...
[TOOL_EXECUTION] Executing tool 'web_search' ...
[TOOL_RESULT] Tool 'web_search' succeeded ...
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

### Example tool execution (Claude tool-use acceptance flow)

These two work offline (`--mock`) exactly as they would against real Claude,
since the tool-use loop is identical either way - only the model backend
differs:

```bash
python main.py --mock "Calculate 12345 * 6789."
# -> AnalysisAgent requests the calculator tool via Claude's native tool-use
#    mechanism, ToolRuntime validates + executes it, the result (83810205)
#    is fed back to the model, which produces the final answer.

python main.py --mock "Create a text file containing the result of 12345 * 6789."
# -> compute_result (AnalysisAgent + calculator) -> write_result_file
#    (WriterAgent + file_write). file_write is permission-checked and
#    sandboxed; the file lands at $ORCHESTRATOR_SANDBOX_DIR/result.txt.
```

An agent can never read or write outside the sandbox: a path like
`../../etc/passwd` or an absolute path like `/etc/passwd` is rejected or
safely reinterpreted as relative to the sandbox root - see
[Filesystem security](#filesystem-security) below.

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
`EVALUATOR`, `RETRY`, `REPLAN`, `COMPLETE`, and the tool lifecycle tags
`TOOL_REQUEST`, `TOOL_VALIDATION`, `TOOL_PERMISSION`, `TOOL_EXECUTION`,
`TOOL_RESULT`, `TOOL_ERROR`. Each event is both printed as
`[TAG] message (field=value, ...)` and stored as a structured dict
(`OrchestrationResult.events`) carrying task id, agent id, tool id, status,
duration, retry count, tokens used, model, and tool call count where
applicable. Tool-call arguments are logged as *metadata only* (key names,
types, lengths) - see [Filesystem security](#filesystem-security) and
[Security controls](#permissions) below for why raw values, especially
anything that looks like a key/token/secret, are never written to logs.

## Tool architecture

```
Tool Registry
├── Native Tools   (CalculatorTool, FileReadTool, FileWriteTool, ListFilesTool, WebSearchTool)
├── MCP Tools      (anything an MCP server reports via list_tools() - auto-discovered)
└── Future API Tools (same BaseTool interface, e.g. a REST wrapper)
```

Every tool - native, MCP-backed, or a REST wrapper - implements the same
`BaseTool` interface (`orchestrator/tools/base.py`):

```python
class BaseTool(ABC):
    id: str                    # unique registry key, e.g. "calculator"
    name: str                  # human-readable name
    description: str
    input_schema: dict         # JSON-Schema-subset for arguments
    output_schema: dict        # JSON-Schema-subset for the result
    permissions: list[str]     # e.g. ["filesystem.write"] - enforced by ToolRuntime
    timeout_seconds: float
    source: str                # "native" | "mcp" | "api"

    async def execute(self, **kwargs) -> ToolResult: ...
```

Agents never depend on a tool's implementation - they only ever see this
interface (via schemas exposed by the `ToolRegistry` and calls routed
through the `ToolRuntime`), so a tool can be swapped from a local demo
implementation to a real API or an MCP server without touching agent or
orchestrator code.

### Tool Registry (`orchestrator/tools/registry.py`)

Central catalog of everything callable:

```python
registry.register(tool)                 # add
registry.unregister("tool_id")          # remove
registry.get("web_search")              # look up by id
registry.is_available("web_search")     # validate availability
registry.discover()                     # full schema/metadata for every tool
registry.search_by_capability("math")   # capability-tagged search
registry.claude_schemas(["calculator"]) # Claude-native tool schemas, scoped
```

Tool selection is never hardcoded into an agent: an agent declares
`available_tools` (a list of tool ids) and the `AgentRouter`/orchestrator
validate that every tool a task requires is (a) registered, (b) declared
by the routed agent, and (c) covered by that agent's permissions -
*before* any model call is made (see `Orchestrator._validate_tool_requirements`).

### Tool Runtime (`orchestrator/tools/runtime.py`)

The single place tool calls actually happen. Every call goes through the
full flow, and no step is ever silently skipped:

```
Agent -> Tool Request -> Permission Check -> Input Validation
      -> Tool Execution (with timeout) -> Output Validation -> Tool Result -> Agent
```

- **Permission check** - enforced here, in code, not just implied by the
  system prompt. A tool declares `permissions`; an agent declares its own
  `permissions`; a call is rejected with `error_code="permission_denied"`
  if the agent doesn't hold everything the tool requires.
- **Input/output validation** - against each tool's JSON-Schema-subset
  (`orchestrator/tools/schema_validation.py`), no `jsonschema` dependency
  needed. Invalid arguments never reach `execute()`; a malformed result
  never reaches the agent.
- **Timeout** - every call runs under `asyncio.wait_for` with the tool's
  own `timeout_seconds` (or a per-call/registry-wide override), producing
  `error_code="timeout"` rather than hanging.
- **Errors are never swallowed** - unavailable tool, invalid arguments,
  a raised exception, a timeout, a permission failure, or a schema
  mismatch each become a structured `ToolResult(success=False, error=...,
  error_code=...)`, logged via `TOOL_ERROR`, and returned to the caller.

```python
result = await tool_runtime.call(
    "calculator",
    agent_id="analysis_agent",
    task_id="t1",
    agent_permissions=["compute"],
    expression="12345 * 6789",
)
```

### Tool call schema

A tool request is structurally `{"tool": "<id>", "arguments": {...}}`. In
practice this arrives as Claude's native `tool_use` block
(`ToolCallRequest(id, name, arguments)` in `orchestrator/providers/base.py`)
rather than free text - see "Claude tool use" below - and `ToolRuntime`
validates `arguments` against the tool's `input_schema` before doing
anything else.

## Claude tool use

Tool use goes through Claude's native, structured mechanism - never text
parsing. `LLMProvider.complete()` accepts `tools` (Claude tool schemas) and
returns `LLMResponse.tool_calls` (structured `ToolCallRequest`s) when the
model decides to use one. `LLMAgent` (`orchestrator/agents/_common.py`)
drives the loop:

```
User Request -> Claude -> Tool Request -> Tool Runtime -> Tool Result -> Claude -> Final Response
```

```python
for _round in range(MAX_TOOL_ROUNDS):
    response = await provider.complete(system=..., messages=messages, tools=tool_schemas)
    if not response.tool_calls:
        return final_text(response)
    messages.append(assistant_turn_with_tool_use_blocks)
    for call in response.tool_calls:
        result = await tool_runtime.call(call.name, agent_permissions=self.permissions, **call.arguments)
        tool_result_blocks.append(as_tool_result(call.id, result))
    messages.append(user_turn_with_tool_result_blocks)
```

`MockProvider` (used by tests and `--mock`) can script this exact shape
offline via `ScriptedToolUse(name=..., arguments=...)`, so the tool loop
itself is exercised identically whether the backend is real Claude or the
mock - see `tests/test_claude_tool_loop.py`.

## Permissions

Permissions are plain strings (`orchestrator/tools/permissions.py`, e.g.
`filesystem.read`, `filesystem.write`, `external_network`, `compute`,
`database.delete`). A tool declares what it needs; an agent declares what
it holds:

| Agent | available_tools | permissions |
|---|---|---|
| `ResearchAgent` | `web_search` | `external_network` |
| `AnalysisAgent` | `calculator` | `compute` |
| `CodingAgent` | `file_read`, `file_write`, `list_files` | `filesystem.read`, `filesystem.write` |
| `WriterAgent` | `file_read`, `file_write` | `filesystem.read`, `filesystem.write` |

Enforcement happens in two places, both in code:
1. **Before execution** - the orchestrator checks a task's `required_tools`
   against the routed agent's `available_tools` and `permissions` and fails
   the task cleanly (no model call wasted) if they don't line up.
2. **At call time** - `ToolRuntime.call()` re-checks `agent_permissions`
   against the tool's declared `permissions` regardless of what the prompt
   said, so a permission is never "soft" - it's checked in code, not
   merely implied to the LLM.

## MCP architecture

```
Tool Registry
├── Native Tools
├── MCP Tools     <- MCPToolAdapter wraps whatever an MCP server exposes
└── Future API Tools
```

`orchestrator/tools/mcp_adapter.py` defines the shape a real MCP client
must satisfy (`MCPClient.list_tools()` / `.call_tool()`) and
`register_mcp_server(registry, client, server_name)` discovers every tool
a server reports and registers it - nothing about individual MCP tools is
hardcoded. Once registered, an `MCPToolAdapter` instance is a `BaseTool`
like any other: same permission checks, same input/output validation, same
timeout handling, same observability tags. Agents and the orchestrator
cannot tell (and don't need to) whether a given tool id is native,
MCP-backed, or a REST API wrapper.

```python
from orchestrator.tools.mcp_adapter import register_mcp_server

# `mcp_client` is anything satisfying the MCPClient protocol - a real MCP
# client, or a test double (see tests/test_mcp_adapter.py).
registered_ids = await register_mcp_server(tool_registry, mcp_client, server_name="search_server")
# -> e.g. ["mcp__search_server__web_search", "mcp__search_server__fetch_page"]
```

This repository does not ship a concrete MCP wire-protocol client (stdio/SSE
JSON-RPC) - that's the one piece intentionally left for the actual
integration - but the adapter, discovery, and registry seams are all in
place and tested against a fake client.

## Filesystem security

All filesystem access goes through `FileSandbox` (`orchestrator/tools/sandbox.py`),
used by `FileReadTool`, `FileWriteTool`, and `ListFilesTool`
(`orchestrator/tools/file_tools.py`). There is no unrestricted filesystem
access anywhere in this codebase, and no shell/arbitrary command execution
tool exists.

- **Configurable root** - `ORCHESTRATOR_SANDBOX_DIR` (default `./sandbox`),
  created automatically.
- **Path traversal protection** - every path is resolved (`Path.resolve()`)
  and checked with `Path.relative_to(root)`; `..` segments that would
  escape the root raise `SandboxViolationError`.
- **Absolute-path override protection** - a naive `root / "/etc/passwd"`
  join in `pathlib` replaces the whole path; this is guarded against by
  stripping leading separators first, so an "absolute" input path is
  always treated as relative to the sandbox root instead of escaping it.
- **Symlink protection** - `resolve()` follows symlinks, so a symlink
  planted inside the sandbox that points outside it is caught by the same
  `relative_to(root)` check.
- **Size limits** - reads/writes over `max_file_size_bytes` (default 5MB)
  are rejected.
- **Permission-gated** - `file_read`/`list_files` require
  `filesystem.read`; `file_write` requires `filesystem.write`, enforced by
  `ToolRuntime` regardless of what an agent's prompt says.

See `tests/test_sandbox.py` and `tests/test_file_tools.py` for the
traversal/absolute-path/symlink test coverage.

## Creating a new agent

1. Subclass `orchestrator.agents.base.BaseAgent` (or the convenience
   `orchestrator.agents._common.LLMAgent`, which wires up prompting and a
   simple tool-request protocol for you).
2. Declare `id`, `name`, `description`, `capabilities`, `available_tools`,
   `permissions` (must cover every tool in `available_tools`'s own
   `permissions`), `system_instructions`.
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

1. Subclass `orchestrator.tools.base.BaseTool`, set `id`, `name`,
   `description`, `input_schema`, `output_schema`, `permissions`, and
   implement `async execute(self, **kwargs) -> ToolResult`.
2. Register an instance with the `ToolRegistry`.
3. Add its `id` to any agent's `available_tools` (and make sure that
   agent's `permissions` cover the tool's `permissions`) so it can
   actually be used - the orchestrator validates this before execution.

```python
from orchestrator.tools.base import BaseTool, ToolResult
from orchestrator.tools.permissions import EXTERNAL_NETWORK

class MyApiTool(BaseTool):
    id = "my_api"
    name = "My API"
    description = "Calls my internal API."
    input_schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    output_schema = {"type": "object", "properties": {"data": {"type": "object"}}}
    permissions = [EXTERNAL_NETWORK]
    timeout_seconds = 10.0

    async def execute(self, *, query: str) -> ToolResult:
        ...
        return ToolResult(success=True, output={"data": ...})
```

The `ToolRuntime` is the single seam agents go through to call tools, so
swapping a demo tool for a real API - or for an MCP-backed tool server via
`register_mcp_server` (see [MCP architecture](#mcp-architecture)) - never
requires touching agent or orchestrator code.

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

The suite (93 tests, all offline/deterministic via `MockProvider` and
in-process test doubles) covers:

- Planner (plan generation, validation, rejection of bad output, replanning)
- Agent registry (registration, capability lookup, specialist ranking)
- Agent routing (capability match, tool-covering preference, no-match error)
- Dependency resolution (readiness, cycles, skip cascades, topological order)
- Task execution (dependency-ordered + parallel execution, context propagation)
- Retry behavior (retry-then-replan-then-abort budget enforcement)
- Evaluation (independent judging - never trusts the agent's own success flag)
- Re-planning (triggered on persistent failure, budgeted, splices in a new plan)
- Context isolation (agents only see their own task + direct dependency outputs;
  `context.tool_results` is structured, never a raw dump)
- Tool registry (registration/unregistration, discovery, capability search,
  Claude schema export) - `tests/test_tool_registry.py`
- Tool runtime (permission checks, input/output validation, timeouts,
  execution errors, argument redaction) - `tests/test_tool_runtime.py`
- Filesystem sandbox (path traversal, absolute-path override, symlink
  escape, size limits) - `tests/test_sandbox.py`, `tests/test_file_tools.py`
- Claude's structured tool-use loop (multi-round tool calling, disallowed
  tool rejection, max-rounds handling) - `tests/test_claude_tool_loop.py`
- MCP adapter (auto-discovery with zero hardcoded tool names, permission
  parity with native tools, transport-error handling) - `tests/test_mcp_adapter.py`
- Orchestrator-level tool/permission pre-flight validation - `tests/test_tool_availability_validation.py`
- Full end-to-end orchestration (`tests/test_e2e.py`, matching the spec's
  "research -> analyze -> strategize" example, using the real `ResearchAgent`,
  `AnalysisAgent`, `WriterAgent` and real tools - including a native Claude
  tool-use round - against a scripted provider)

## Repository layout

```
orchestrator/
  core/        orchestrator loop, planner, router, task graph, context,
               state, evaluator, retry, synthesizer, logging
  agents/      BaseAgent, AgentRegistry, ResearchAgent, AnalysisAgent,
               CodingAgent, WriterAgent, LLMAgent (Claude tool-use loop)
  tools/       BaseTool, ToolRegistry, ToolRuntime, schema_validation,
               permissions, sandbox (FileSandbox), file_tools
               (FileReadTool/FileWriteTool/ListFilesTool), calculator_tool,
               web_search_tool, mcp_adapter (MCPToolAdapter, MCPClient,
               register_mcp_server)
  providers/   LLMProvider, ClaudeProvider (native tool-use), MockProvider
               (ScriptedToolUse for offline tool-loop testing)
tests/         pytest suite (see above)
main.py        CLI entry point
.env.example   environment variable template
```

## Known limitations

- The example `WebSearchTool` is backed by a small local demo corpus, not
  a real search API - swap its implementation (or register a different
  tool under the same id) for production use; the tool abstraction is
  designed so that requires no agent/orchestrator changes.
- The `Evaluator` is rule-based (empty/short output, refusal language,
  reported errors) rather than an LLM-as-judge; it's easy to extend with
  an LLM-backed check by giving `Evaluator` a provider, but the default
  keeps evaluation fast, deterministic, and cheap.
- `orchestrator/tools/mcp_adapter.py` defines the client protocol,
  auto-discovery, and adapter that make MCP tools indistinguishable from
  native ones, but this repository does not ship a concrete MCP wire
  transport (stdio/SSE JSON-RPC) - `register_mcp_server` needs to be
  pointed at a real `MCPClient` implementation.
- `LLMAgent`'s tool loop supports one round of (potentially multiple)
  tool calls per iteration, bounded by `MAX_TOOL_ROUNDS` (4); a task
  needing more sequential tool calls than that fails cleanly rather than
  looping forever.
- `FileSandbox` is not hardened against TOCTOU races (a symlink swapped in
  between the `resolve()` check and the actual read/write) - acceptable
  for a single-process orchestrator but worth noting for a
  multi-tenant/adversarial deployment.
- No shell/arbitrary command execution tool exists by design (per the
  Phase 2 spec); adding one would need its own sandboxing story beyond
  what `FileSandbox` provides.
