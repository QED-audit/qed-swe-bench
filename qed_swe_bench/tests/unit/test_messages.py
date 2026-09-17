"""Per-provider message builders."""

from __future__ import annotations

import json

from qed_swe_bench.runner.llm.base import (
    NormalizedResponse,
    NormalizedToolCall,
    NormalizedUsage,
)
from qed_swe_bench.runner.llm.messages import (
    AnthropicMessages,
    OpenAIMessages,
    build_messages_for,
)


def _resp(text=None, tool_calls=(), reasoning_content=None, raw=None):
    return NormalizedResponse(
        text=text,
        tool_calls=tuple(tool_calls),
        usage=NormalizedUsage(),
        stop_reason="ok",
        model="m",
        reasoning_content=reasoning_content,
        raw=raw or {},
    )


# ---------------- AnthropicMessages ----------------


def test_anthropic_initial_user_message() -> None:
    m = AnthropicMessages()
    m.initial("init")
    msgs = m.get()
    assert msgs == [{"role": "user", "content": "init"}]


def test_anthropic_assistant_text_only() -> None:
    m = AnthropicMessages()
    m.initial("init")
    m.append_assistant(_resp(text="thinking"))
    msgs = m.get()
    assert msgs[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "thinking"}],
    }


def test_anthropic_assistant_with_tool_calls() -> None:
    m = AnthropicMessages()
    m.initial("init")
    m.append_assistant(
        _resp(
            text="reasoning",
            tool_calls=[NormalizedToolCall(id="t1", name="exec", arguments={"cmd": "ls"})],
        )
    )
    blocks = m.get()[1]["content"]
    assert blocks[0] == {"type": "text", "text": "reasoning"}
    assert blocks[1] == {
        "type": "tool_use",
        "id": "t1",
        "name": "exec",
        "input": {"cmd": "ls"},
    }


def test_anthropic_tool_results_grouped_into_one_user_message() -> None:
    m = AnthropicMessages()
    tc1 = NormalizedToolCall(id="t1", name="exec", arguments={"cmd": "ls"})
    tc2 = NormalizedToolCall(id="t2", name="read_file", arguments={"path": "/x"})
    m.append_tool_results([(tc1, "stdout", False), (tc2, "ENOENT", True)])
    msgs = m.get()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "stdout"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "ENOENT", "is_error": True},
    ]


def test_anthropic_append_user_for_nudges() -> None:
    m = AnthropicMessages()
    m.append_user("nudge text")
    assert m.get() == [{"role": "user", "content": "nudge text"}]


def test_anthropic_assistant_replays_content_blocks_verbatim() -> None:
    """When content_blocks is populated, replay it as-is.

    Extended thinking is in this regime: thinking blocks must round-trip
    with their signatures intact and in original position relative to
    tool_use blocks, or Anthropic 400s the next request.
    """
    blocks = (
        {"type": "thinking", "thinking": "step 1", "signature": "sig-1"},
        {"type": "tool_use", "id": "t1", "name": "exec", "input": {"cmd": "ls"}},
        {"type": "thinking", "thinking": "step 2", "signature": "sig-2"},
        {"type": "redacted_thinking", "data": "encrypted"},
        {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "/x"}},
    )
    resp = NormalizedResponse(
        text=None,
        tool_calls=(
            NormalizedToolCall(id="t1", name="exec", arguments={"cmd": "ls"}),
            NormalizedToolCall(id="t2", name="read_file", arguments={"path": "/x"}),
        ),
        usage=NormalizedUsage(),
        stop_reason="tool_use",
        model="claude-opus-4-6",
        content_blocks=blocks,
    )

    m = AnthropicMessages()
    m.initial("init")
    m.append_assistant(resp)

    # Original block order preserved; signatures intact.
    assert m.get()[1] == {"role": "assistant", "content": list(blocks)}


def test_anthropic_assistant_replay_is_a_copy_not_alias() -> None:
    """Mutating the message list shouldn't bleed back into the response."""
    blocks = (
        {"type": "thinking", "thinking": "x", "signature": "s"},
        {"type": "text", "text": "out"},
    )
    resp = NormalizedResponse(
        text="out",
        tool_calls=(),
        usage=NormalizedUsage(),
        stop_reason="end_turn",
        model="m",
        content_blocks=blocks,
    )
    m = AnthropicMessages()
    m.append_assistant(resp)
    m.get()[0]["content"][0]["thinking"] = "MUTATED"
    assert resp.content_blocks[0]["thinking"] == "x"


# ---------------- OpenAIMessages ----------------


def test_openai_assistant_serializes_tool_call_args_as_json() -> None:
    m = OpenAIMessages()
    m.initial("init")
    m.append_assistant(
        _resp(
            text="ok",
            tool_calls=[NormalizedToolCall(id="c1", name="exec", arguments={"cmd": "ls"})],
        )
    )
    msg = m.get()[1]
    assert msg["role"] == "assistant"
    assert msg["content"] == "ok"
    assert msg["tool_calls"][0] == {
        "id": "c1",
        "type": "function",
        "function": {"name": "exec", "arguments": json.dumps({"cmd": "ls"})},
    }


def test_openai_tool_results_one_message_each() -> None:
    m = OpenAIMessages()
    tc1 = NormalizedToolCall(id="c1", name="exec", arguments={})
    tc2 = NormalizedToolCall(id="c2", name="read_file", arguments={})
    m.append_tool_results([(tc1, "ok", False), (tc2, "fail", True)])
    msgs = m.get()
    assert [m_["role"] for m_ in msgs] == ["tool", "tool"]
    assert msgs[0]["tool_call_id"] == "c1"
    assert msgs[0]["content"] == "ok"
    # Errors are inlined in the content for OpenAI (no is_error field).
    assert "[tool error]" in msgs[1]["content"]
    assert "fail" in msgs[1]["content"]


def test_openai_assistant_text_only_content_is_string() -> None:
    m = OpenAIMessages()
    m.append_assistant(_resp(text="hello"))
    assert m.get()[0]["content"] == "hello"
    assert "tool_calls" not in m.get()[0]


def test_openai_assistant_replays_encrypted_reasoning_items_when_enabled() -> None:
    item = {
        "id": "rs_123",
        "type": "reasoning",
        "encrypted_content": "ciphertext",
        "summary": [],
    }
    response = _resp(
        tool_calls=[NormalizedToolCall(id="c1", name="exec", arguments={"cmd": "ls"})],
        raw={"choices": [{"message": {"reasoning_items": [item]}}]},
    )

    default = OpenAIMessages()
    default.append_assistant(response)
    assert "reasoning_items" not in default.get()[0]

    replay = OpenAIMessages(preserve_reasoning_items=True)
    replay.append_assistant(response)
    assert replay.get()[0]["reasoning_items"] == [item]
    assert replay.get()[0]["reasoning_items"][0] is not item


def test_factory_enables_reasoning_item_replay_only_for_openai() -> None:
    openai = build_messages_for("litellm", model_id="openai/gpt-5.5-2026-04-23")
    assert isinstance(openai, OpenAIMessages)
    assert openai._preserve_reasoning_items is True

    other = build_messages_for("litellm", model_id="gemini/gemini-3.1-pro-preview")
    assert isinstance(other, OpenAIMessages)
    assert other._preserve_reasoning_items is False


# ---------------- build_messages_for ----------------


def test_factory_picks_anthropic_for_native_route() -> None:
    assert isinstance(build_messages_for("anthropic_native"), AnthropicMessages)


def test_factory_picks_openai_for_litellm_routes() -> None:
    assert isinstance(build_messages_for("litellm"), OpenAIMessages)
    assert isinstance(build_messages_for("litellm_gateway"), OpenAIMessages)
    assert isinstance(build_messages_for("mock"), OpenAIMessages)


# ---------------- Moonshot reasoning_content preservation ----------------


def test_openai_assistant_omits_reasoning_content_by_default() -> None:
    """Default OpenAIMessages must NOT include reasoning_content in
    assistant messages — OpenAI/Gemini/etc. reject unknown fields and
    would 400 the request. Moonshot is the only provider that requires
    (and accepts) it; the gate is set by build_messages_for via
    `model_id`. Pin the default-off behavior so non-Moonshot providers
    don't accidentally receive the field."""
    m = OpenAIMessages()  # default: preserve_reasoning_content=False
    m.append_assistant(_resp(text="thinking visibly", reasoning_content="secret thoughts"))
    msg = m.get()[0]
    assert msg["content"] == "thinking visibly"
    assert "reasoning_content" not in msg


def test_openai_assistant_preserves_reasoning_content_when_enabled() -> None:
    """When preserve_reasoning_content=True (Moonshot route), the field
    is included on the outgoing assistant message so Moonshot's API
    sees the model's prior thinking trace and doesn't 400. Replaces
    LiteLLM's placeholder injection with the real trace."""
    m = OpenAIMessages(preserve_reasoning_content=True)
    m.append_assistant(_resp(text="visible", reasoning_content="full chain of thought"))
    msg = m.get()[0]
    assert msg["content"] == "visible"
    assert msg["reasoning_content"] == "full chain of thought"


def test_openai_preserves_reasoning_content_only_when_present() -> None:
    """If the response has no reasoning_content (None), don't add an
    empty/null field to the message. Some providers reject explicit
    nulls on unknown fields even when they accept the field omitted."""
    m = OpenAIMessages(preserve_reasoning_content=True)
    m.append_assistant(_resp(text="visible only", reasoning_content=None))
    assert "reasoning_content" not in m.get()[0]


def test_factory_enables_reasoning_preservation_for_required_families() -> None:
    """Reasoning replay follows the underlying provider through gateways."""
    for model in (
        "moonshot/kimi-k2.6",
        "moonshotai/kimi-k2.6",
        "qwen/qwen3.8-27b",
        "zai/glm-5.3",
        "z-ai/glm-5.3",
        "openrouter/qwen/qwen3.8-27b",
        "openrouter/z-ai/glm-5.3",
        "litellm_proxy/qwen/qwen3.8-2.4t-a95b",
        "litellm_proxy/zai/glm-5.3",
    ):
        builder = build_messages_for("litellm", model_id=model)
        assert isinstance(builder, OpenAIMessages)
        assert builder._preserve_reasoning is True, (
            f"{model} should preserve reasoning_content"
        )


def test_factory_keeps_reasoning_replay_off_for_other_families() -> None:
    """Families without a plaintext replay contract keep the safe default."""
    for model in (
        "openai/gpt-5.5",
        "gemini/gemini-3.1-pro-preview",
        "openrouter/minimax/minimax-m2.7",
    ):
        builder = build_messages_for("litellm", model_id=model)
        assert isinstance(builder, OpenAIMessages)
        assert builder._preserve_reasoning is False, (
            f"{model} should not preserve reasoning_content"
        )


def test_factory_default_model_id_is_safe() -> None:
    """build_messages_for(route) with no model_id — used by older test
    fixtures and any caller that doesn't know the model id — must
    default to NOT preserving reasoning_content (the safe default for
    most providers)."""
    builder = build_messages_for("litellm")
    assert isinstance(builder, OpenAIMessages)
    assert builder._preserve_reasoning is False
