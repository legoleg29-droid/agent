import pytest

from orchestrator.tools.calculator_tool import CalculatorTool
from orchestrator.tools.registry import ToolNotFoundError, ToolRegistry
from orchestrator.tools.web_search_tool import WebSearchTool


def test_register_and_get():
    registry = ToolRegistry()
    calc = CalculatorTool()
    registry.register(calc)
    assert registry.get("calculator") is calc


def test_get_missing_raises():
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        registry.get("nope")


def test_unregister_removes_tool():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    assert registry.has("calculator")
    registry.unregister("calculator")
    assert not registry.has("calculator")


def test_unregister_missing_is_a_noop():
    registry = ToolRegistry()
    registry.unregister("does_not_exist")  # must not raise


def test_is_available_validates_registration():
    registry = ToolRegistry()
    assert not registry.is_available("calculator")
    registry.register(CalculatorTool())
    assert registry.is_available("calculator")


def test_discover_exposes_full_schema_metadata():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    [entry] = registry.discover()
    assert entry["id"] == "calculator"
    assert entry["input_schema"]["required"] == ["expression"]
    assert entry["permissions"] == []
    assert entry["source"] == "native"


def test_search_by_capability():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())

    assert [t.id for t in registry.search_by_capability("math")] == ["calculator"]
    assert [t.id for t in registry.search_by_capability("search")] == ["web_search"]
    assert registry.search_by_capability("nonexistent") == []


def test_claude_schemas_restricted_to_given_ids():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())

    schemas = registry.claude_schemas(["calculator"])
    assert [s["name"] for s in schemas] == ["calculator"]
    assert set(schemas[0]) == {"name", "description", "input_schema"}


def test_claude_schemas_all_when_unrestricted():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    assert {s["name"] for s in registry.claude_schemas()} == {"calculator", "web_search"}
