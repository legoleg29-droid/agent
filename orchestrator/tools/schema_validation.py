"""Minimal JSON-Schema-subset validator.

Deliberately small and dependency-free rather than pulling in the
``jsonschema`` package: tool schemas here only ever need object/property
validation with a handful of primitive types, which this covers. Returns a
list of human-readable error strings (empty = valid) instead of raising,
so callers can decide how to report validation failures.
"""

from __future__ import annotations

from typing import Any

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


def validate_against_schema(data: Any, schema: dict[str, Any], *, path: str = "$") -> list[str]:
    """Validate ``data`` against a (subset of) JSON Schema ``schema``."""
    if not schema:
        return []
    errors: list[str] = []
    _validate_node(data, schema, path, errors)
    return errors


def _validate_node(data: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    expected_type = schema.get("type")
    if expected_type:
        allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
        py_types: tuple[type, ...] = tuple(t for name in allowed_types for t in _TYPE_MAP.get(name, ()))
        # bool is a subclass of int in Python; only accept it where "boolean" was allowed.
        if isinstance(data, bool) and "boolean" not in allowed_types and int in py_types:
            errors.append(f"{path}: expected type {allowed_types}, got boolean")
            return
        if py_types and not isinstance(data, py_types):
            errors.append(f"{path}: expected type {allowed_types}, got {type(data).__name__}")
            return

    enum = schema.get("enum")
    if enum is not None and data not in enum:
        errors.append(f"{path}: value {data!r} not in allowed enum {enum}")

    if isinstance(data, dict) and (expected_type == "object" or "properties" in schema):
        properties = schema.get("properties", {})
        for required_key in schema.get("required", []):
            if required_key not in data:
                errors.append(f"{path}: missing required property '{required_key}'")
        for key, value in data.items():
            if key in properties:
                _validate_node(value, properties[key], f"{path}.{key}", errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property '{key}'")

    if isinstance(data, list) and (expected_type == "array" or "items" in schema):
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(data):
                _validate_node(item, item_schema, f"{path}[{i}]", errors)
