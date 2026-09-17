"""Unit tests for AnthropicNative — caching shape + response translation.

These do NOT hit the real API. A real-API cache regression test lives
separately under the `slow` marker.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qed_swe_bench.runner.llm.anthropic_native import (
    DEFAULT_RATE_LIMIT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT_S,
    AnthropicNative,
    _rate_limit_max_retries,
    _request_timeout_s,
)


def test_mark_last_cache_only_marks_final() -> None:
    tools = [
        {"name": "a", "description": "A", "input_schema": {}},
        {"name": "b", "description": "B", "input_schema": {}},
        {"name": "c", "description": "C", "input_schema": {}},
    ]
    out = AnthropicNative._mark_last_cache(tools)

    # First two unchanged.
    assert "cache_control" not in out[0]
    assert "cache_control" not in out[1]
    # Last has cache_control.
    assert out[2]["cache_control"] == {"type": "ephemeral"}
    # Original list unmodified (deep copy).
    assert "cache_control" not in tools[2]


def test_mark_last_cache_empty_tools() -> None:
    assert AnthropicNative._mark_last_cache([]) == []


def test_mark_last_message_cache_string_content() -> None:
    """String-content message gets wrapped as a single text block w/ cache_control."""
    msgs = [{"role": "user", "content": "hello"}]
    out = AnthropicNative._mark_last_message_cache(msgs)
    assert out[-1]["content"] == [
        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
    ]
    # Original untouched.
    assert msgs[-1]["content"] == "hello"


def test_mark_last_message_cache_list_content_marks_last_block() -> None:
    """For a list-content message, only the LAST block gets cache_control."""
    msgs = [
        {"role": "user", "content": "one"},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "x"},
                {"type": "tool_result", "tool_use_id": "b", "content": "y"},
            ],
        },
    ]
    out = AnthropicNative._mark_last_message_cache(msgs)
    last_blocks = out[-1]["content"]
    assert "cache_control" not in last_blocks[0]
    assert last_blocks[1]["cache_control"] == {"type": "ephemeral"}
    # Earlier message untouched.
    assert out[0]["content"] == "one"
    # Original untouched (deep-ish copy).
    assert "cache_control" not in msgs[-1]["content"][1]


def test_mark_last_message_cache_empty() -> None:
    assert AnthropicNative._mark_last_message_cache([]) == []


def test_to_normalized_extracts_text_and_tool_calls() -> None:
    """Translate a stub Message-like object."""
    fake_resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="thinking..."),
            SimpleNamespace(
                type="tool_use",
                id="toolu_01",
                name="setup",
                input={},
            ),
            SimpleNamespace(
                type="tool_use",
                id="toolu_02",
                name="exec",
                input={"cmd": "ls"},
            ),
        ],
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=45,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=100,
        ),
        stop_reason="tool_use",
        model="claude-sonnet-4-5-20250929",
    )

    norm = AnthropicNative._to_normalized(fake_resp)
    assert norm.text == "thinking..."
    assert len(norm.tool_calls) == 2
    assert norm.tool_calls[0].id == "toolu_01"
    assert norm.tool_calls[0].name == "setup"
    assert norm.tool_calls[0].arguments == {}
    assert norm.tool_calls[1].arguments == {"cmd": "ls"}
    assert norm.usage.input_tokens == 120
    assert norm.usage.output_tokens == 45
    assert norm.usage.cache_creation_tokens == 100
    assert norm.usage.cache_read_tokens == 0
    assert norm.stop_reason == "tool_use"
    assert norm.model == "claude-sonnet-4-5-20250929"


def test_to_normalized_handles_missing_cache_fields() -> None:
    """Older API responses may not have cache_* fields populated."""
    fake_resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
        stop_reason="end_turn",
        model="claude-haiku-4-5",
    )
    norm = AnthropicNative._to_normalized(fake_resp)
    assert norm.usage.cache_read_tokens == 0
    assert norm.usage.cache_creation_tokens == 0
    assert norm.text == "hi"
    assert norm.tool_calls == ()


def test_to_normalized_handles_only_tool_use_no_text() -> None:
    fake_resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", id="toolu_x", name="grade", input={"path": "/foo"}),
        ],
        usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        stop_reason="tool_use",
        model="claude-haiku-4-5",
    )
    norm = AnthropicNative._to_normalized(fake_resp)
    assert norm.text is None
    assert len(norm.tool_calls) == 1
    assert norm.tool_calls[0].arguments == {"path": "/foo"}


# ---------------- request-timeout resolution (mirrors litellm_client) ----


def test_request_timeout_default_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_REQUEST_TIMEOUT_S", raising=False)
    assert _request_timeout_s() == DEFAULT_REQUEST_TIMEOUT_S
    assert DEFAULT_REQUEST_TIMEOUT_S >= 60, (
        "default must be generous enough for multi-turn requests; "
        "see cap-demo seed-4 TimeoutError diagnosis"
    )


def test_request_timeout_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_REQUEST_TIMEOUT_S", "120")
    assert _request_timeout_s() == 120


def test_request_timeout_garbage_env_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("ANTHROPIC_REQUEST_TIMEOUT_S", "junk")
    with caplog.at_level("WARNING"):
        assert _request_timeout_s() == DEFAULT_REQUEST_TIMEOUT_S
    assert any("not a positive int" in r.message for r in caplog.records)


def test_request_timeout_zero_or_negative_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for bad in ("0", "-30"):
        monkeypatch.setenv("ANTHROPIC_REQUEST_TIMEOUT_S", bad)
        assert _request_timeout_s() == DEFAULT_REQUEST_TIMEOUT_S


def test_anthropic_native_passes_timeout_to_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction should plumb the resolved timeout into anthropic.Anthropic."""
    captured: dict[str, object] = {}

    def fake_anthropic(**kwargs):  # noqa: ANN001 — typing the SDK adds noise
        captured.update(kwargs)
        return SimpleNamespace(messages=SimpleNamespace())

    import qed_swe_bench.runner.llm.anthropic_native as mod
    monkeypatch.setattr(mod.anthropic, "Anthropic", fake_anthropic)
    monkeypatch.setenv("ANTHROPIC_REQUEST_TIMEOUT_S", "240")

    AnthropicNative(model="anthropic/claude-haiku-4-5", api_key="sk-x")
    assert captured["timeout"] == 240
    assert captured["api_key"] == "sk-x"


# ---------------- rate-limit retry plumbing ----------------


def test_rate_limit_default_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", raising=False)
    assert _rate_limit_max_retries() == DEFAULT_RATE_LIMIT_MAX_RETRIES
    assert DEFAULT_RATE_LIMIT_MAX_RETRIES >= 3, (
        "must allow enough retries to ride out a multi-second Retry-After "
        "(OpenAI's TPM cap returns ~6s waits); see SLACK_RATE_LIMIT context"
    )


def test_rate_limit_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", "10")
    assert _rate_limit_max_retries() == 10


def test_rate_limit_zero_disables_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting to 0 explicitly disables retries — useful when debugging
    rate-limit failures without backoff masking them."""
    monkeypatch.setenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", "0")
    assert _rate_limit_max_retries() == 0


def test_rate_limit_garbage_env_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", "junk")
    with caplog.at_level("WARNING"):
        assert _rate_limit_max_retries() == DEFAULT_RATE_LIMIT_MAX_RETRIES
    assert any("non-negative int" in r.message for r in caplog.records)


def test_rate_limit_negative_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", "-1")
    assert _rate_limit_max_retries() == DEFAULT_RATE_LIMIT_MAX_RETRIES


def test_anthropic_native_passes_max_retries_to_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction should plumb the retry count into anthropic.Anthropic."""
    captured: dict[str, object] = {}

    def fake_anthropic(**kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return SimpleNamespace(messages=SimpleNamespace())

    import qed_swe_bench.runner.llm.anthropic_native as mod
    monkeypatch.setattr(mod.anthropic, "Anthropic", fake_anthropic)
    monkeypatch.setenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", "8")

    AnthropicNative(model="anthropic/claude-haiku-4-5", api_key="sk-x")
    assert captured["max_retries"] == 8


# ---------------- params pass-through (extended thinking, temperature) -----


def _stub_client_capturing(captured: dict[str, object]) -> object:
    """Build a fake `anthropic.Anthropic()` whose messages.create captures kwargs
    and returns a minimal text-only Message-shaped response."""

    def fake_create(**kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
            model="claude-opus-4-6",
        )

    return SimpleNamespace(messages=SimpleNamespace(create=fake_create))


def test_params_pass_through_to_messages_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-model `params` should be merged into messages.create kwargs."""
    captured: dict[str, object] = {}

    import qed_swe_bench.runner.llm.anthropic_native as mod
    monkeypatch.setattr(
        mod.anthropic, "Anthropic", lambda **_: _stub_client_capturing(captured),
    )

    client = AnthropicNative(
        model="anthropic/claude-opus-4-6",
        api_key="sk-x",
        params={
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "temperature": 1.0,
        },
    )
    client.complete(messages=[], tools=[], max_tokens=4096)

    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 10000}
    assert captured["temperature"] == 1.0
    assert captured["max_tokens"] == 4096
    assert captured["model"] == "claude-opus-4-6"


def test_params_none_pops_existing_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    """A None value should *remove* the kwarg, not pass None through."""
    captured: dict[str, object] = {}

    import qed_swe_bench.runner.llm.anthropic_native as mod
    monkeypatch.setattr(
        mod.anthropic, "Anthropic", lambda **_: _stub_client_capturing(captured),
    )

    # max_tokens is a default kwarg AnthropicNative always sets.
    client = AnthropicNative(
        model="claude-opus-4-6",
        api_key="sk-x",
        params={"max_tokens": None},
    )
    client.complete(messages=[], tools=[], max_tokens=4096)

    assert "max_tokens" not in captured


# ---------------- content_blocks capture (extended thinking + tool use) ---


def test_to_normalized_captures_content_blocks_in_order() -> None:
    """Interleaved thinking/tool_use blocks must be captured in original order
    so AnthropicMessages can replay them with signatures intact."""
    fake_resp = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="thinking", thinking="reasoning step 1", signature="sig-1",
            ),
            SimpleNamespace(
                type="tool_use", id="toolu_01", name="exec", input={"cmd": "ls"},
            ),
            SimpleNamespace(
                type="thinking", thinking="reasoning step 2", signature="sig-2",
            ),
            SimpleNamespace(type="redacted_thinking", data="encrypted-blob"),
            SimpleNamespace(
                type="tool_use", id="toolu_02", name="grade", input={"path": "/x"},
            ),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
        model="claude-opus-4-6",
    )

    norm = AnthropicNative._to_normalized(fake_resp)

    # All five blocks captured, in original order.
    assert len(norm.content_blocks) == 5
    types = [b["type"] for b in norm.content_blocks]
    assert types == [
        "thinking", "tool_use", "thinking", "redacted_thinking", "tool_use",
    ]
    # Signatures and encrypted data round-trip verbatim.
    assert norm.content_blocks[0]["signature"] == "sig-1"
    assert norm.content_blocks[2]["signature"] == "sig-2"
    assert norm.content_blocks[3]["data"] == "encrypted-blob"

    # Convenience views populated correctly: thinking blocks excluded from
    # `text`, both tool calls present.
    assert norm.text is None
    assert len(norm.tool_calls) == 2
    assert norm.tool_calls[0].id == "toolu_01"
    assert norm.tool_calls[1].id == "toolu_02"


def test_to_normalized_handles_dict_blocks() -> None:
    """Some test fixtures may supply already-dict blocks; capture path must
    cope (real SDK returns pydantic models, but we don't want to crash on
    dicts)."""
    fake_resp = SimpleNamespace(
        content=[
            {"type": "thinking", "thinking": "x", "signature": "sig"},
            {"type": "text", "text": "answer"},
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
        model="claude-opus-4-6",
    )
    norm = AnthropicNative._to_normalized(fake_resp)
    assert norm.content_blocks[0] == {
        "type": "thinking", "thinking": "x", "signature": "sig",
    }
    assert norm.text == "answer"


def test_to_normalized_empty_when_thinking_disabled() -> None:
    """Without thinking, content_blocks still captures text/tool_use.

    Whether the loop chooses the verbatim or the rebuild path, behavior is
    identical for non-thinking turns — but content_blocks should still be
    populated, since AnthropicNative always populates it.
    """
    fake_resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hi"),
            SimpleNamespace(type="tool_use", id="t1", name="exec", input={"cmd": "ls"}),
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
        model="claude-opus-4-6",
    )
    norm = AnthropicNative._to_normalized(fake_resp)
    assert [b["type"] for b in norm.content_blocks] == ["text", "tool_use"]
