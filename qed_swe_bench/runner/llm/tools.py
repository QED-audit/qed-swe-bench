"""Convert MCP tool definitions to Anthropic / OpenAI tool schemas.

MCP tools (per `mcp.types.Tool`) carry:
  name: str
  description: str | None
  inputSchema: dict   # JSON Schema, may be missing on some servers

Anthropic format:
  {"name": ..., "description": ..., "input_schema": {...}}

OpenAI tools format:
  {"type": "function",
   "function": {"name": ..., "description": ..., "parameters": {...}}}

Both providers accept the JSON Schema verbatim; this module is a pure
shape translation.
"""

from __future__ import annotations

from typing import Any


def _normalize_schema(schema: Any) -> dict[str, Any]:
    """Coerce missing/empty schemas into a valid object schema for any provider.

    Two cases we have to fix to satisfy OpenAI's stricter function-schema
    validation (Anthropic accepts both shapes, but OpenAI 400s the second):

      1. Missing/empty schema entirely → `{type: object, properties: {}}`
      2. Object schema that omits `properties` (some MCP servers return
         `{type: object, additionalProperties: false}` for no-arg tools
         like `setup`) → add `properties: {}`. Without this, OpenAI returns
         400 invalid_function_parameters: "object schema missing properties".
    """
    if not schema or not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    if schema.get("type") == "object" and "properties" not in schema:
        return {**schema, "properties": {}}
    return schema


def mcp_to_anthropic(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP tool list to Anthropic tool definitions."""
    return [
        {
            "name": t["name"],
            "description": t.get("description") or "",
            "input_schema": _normalize_schema(t.get("inputSchema")),
        }
        for t in mcp_tools
    ]


def mcp_to_openai(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP tool list to OpenAI tools-format definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description") or "",
                "parameters": _normalize_schema(t.get("inputSchema")),
            },
        }
        for t in mcp_tools
    ]
