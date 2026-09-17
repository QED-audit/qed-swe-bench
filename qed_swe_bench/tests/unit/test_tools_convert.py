"""MCP tool list → Anthropic / OpenAI format."""

from __future__ import annotations

from qed_swe_bench.runner.llm.tools import mcp_to_anthropic, mcp_to_openai

SAMPLE_MCP_TOOLS = [
    {
        "name": "setup",
        "description": "Initialize the env",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "exec",
        "description": "Run a shell command",
        "inputSchema": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
    {
        "name": "missing_schema",
        "description": "no schema",
        # no inputSchema
    },
]


def test_mcp_to_anthropic() -> None:
    out = mcp_to_anthropic(SAMPLE_MCP_TOOLS)
    assert len(out) == 3
    assert out[0]["name"] == "setup"
    assert out[0]["description"] == "Initialize the env"
    assert out[0]["input_schema"] == {"type": "object", "properties": {}, "required": []}
    assert out[1]["input_schema"]["properties"]["cmd"]["type"] == "string"
    # Missing schema gets a sane default.
    assert out[2]["input_schema"] == {"type": "object", "properties": {}}


def test_mcp_to_openai() -> None:
    out = mcp_to_openai(SAMPLE_MCP_TOOLS)
    assert len(out) == 3
    for entry in out:
        assert entry["type"] == "function"
    assert out[0]["function"]["name"] == "setup"
    assert out[1]["function"]["parameters"]["properties"]["cmd"]["type"] == "string"
    # Missing schema → default.
    assert out[2]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_mcp_to_anthropic_empty_list() -> None:
    assert mcp_to_anthropic([]) == []
    assert mcp_to_openai([]) == []


def test_description_falls_back_to_empty_string() -> None:
    tools = [{"name": "x"}]  # no description
    assert mcp_to_anthropic(tools)[0]["description"] == ""
    assert mcp_to_openai(tools)[0]["function"]["description"] == ""
