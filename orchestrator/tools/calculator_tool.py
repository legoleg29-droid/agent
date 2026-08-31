"""Safe arithmetic calculator tool.

A minimal, dependency-free example tool. It evaluates arithmetic
expressions using Python's ``ast`` module restricted to a safe subset of
operators, so it never uses ``eval``.
"""

from __future__ import annotations

import ast
import operator

from orchestrator.tools.base import BaseTool, ToolErrorCode, ToolResult

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


class CalculatorTool(BaseTool):
    id = "calculator"
    name = "Calculator"
    description = "Evaluates a basic arithmetic expression (+ - * / % **) and returns the numeric result."
    input_schema = {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    }
    output_schema = {
        "type": "object",
        "properties": {"result": {"type": "number"}},
        "required": ["result"],
    }
    permissions: list[str] = []  # pure computation, no filesystem/network access
    capabilities = ["math", "compute"]
    timeout_seconds = 5.0

    async def execute(self, *, expression: str) -> ToolResult:
        try:
            tree = ast.parse(expression, mode="eval")
            value = _eval_node(tree.body)
            return ToolResult(success=True, output={"result": value})
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                error=f"Invalid expression: {exc}",
                error_code=ToolErrorCode.INVALID_ARGUMENTS,
            )
