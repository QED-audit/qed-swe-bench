"""TranscriptWriter + JSONL entry shapes match bench-v8's format."""

from __future__ import annotations

import json
from pathlib import Path

from qed_swe_bench.runner.llm.base import (
    NormalizedResponse,
    NormalizedToolCall,
    NormalizedUsage,
)
from qed_swe_bench.runner.transcript import (
    TranscriptWriter,
    ai_entry,
    grade_log_entry,
    human_entry,
    system_entry,
    tool_call_log_entry,
    tool_message_entry,
    try_parse_grade_result,
)

# ---------------- entry shapes ----------------


def test_system_entry_shape() -> None:
    e = system_entry("hello world", ts="2026-04-24T12:00:00+00:00")
    assert e == {
        "ts": "2026-04-24T12:00:00+00:00",
        "role": "system",
        "content": "hello world",
    }


def test_human_entry_shape() -> None:
    e = human_entry("init", ts="2026-04-24T12:00:00+00:00")
    assert e["role"] == "human"
    assert e["content"] == "init"


def test_ai_entry_text_only_no_usage() -> None:
    resp = NormalizedResponse(
        text="thinking",
        tool_calls=(),
        usage=NormalizedUsage(),
        stop_reason="end_turn",
        model="claude-haiku-4-5",
    )
    e = ai_entry(resp, ts="2026-04-24T12:00:00+00:00")
    assert e == {
        "ts": "2026-04-24T12:00:00+00:00",
        "role": "ai",
        "content": "thinking",
        "stop_reason": "end_turn",
        "served_model": "claude-haiku-4-5",
    }


def test_ai_entry_omits_served_model_when_blank() -> None:
    """Loop tracks first-non-empty model; if a stub/mock returns "" we
    shouldn't write a noisy empty field."""
    resp = NormalizedResponse(
        text="x",
        tool_calls=(),
        usage=NormalizedUsage(),
        stop_reason="end_turn",
        model="",
    )
    e = ai_entry(resp, ts="T")
    assert "served_model" not in e


def test_ai_entry_includes_reasoning_tokens_when_nonzero() -> None:
    """reasoning_tokens proves an OpenAI reasoning model actually reasoned."""
    resp = NormalizedResponse(
        text="ok",
        tool_calls=(),
        usage=NormalizedUsage(
            input_tokens=120, output_tokens=2048, reasoning_tokens=2000
        ),
        stop_reason="stop",
        model="gpt-5.5-2026-04-23",
    )
    e = ai_entry(resp, ts="T")
    assert e["served_model"] == "gpt-5.5-2026-04-23"
    assert e["usage"]["reasoning_tokens"] == 2000


def test_ai_entry_omits_reasoning_tokens_when_zero() -> None:
    resp = NormalizedResponse(
        text="ok",
        tool_calls=(),
        usage=NormalizedUsage(input_tokens=10, output_tokens=2),
        stop_reason="stop",
        model="claude-haiku-4-5",
    )
    e = ai_entry(resp, ts="T")
    assert "reasoning_tokens" not in e["usage"]


def test_ai_entry_includes_reasoning_content_when_present() -> None:
    """OpenAI-compat providers (Moonshot, MiniMax) return the reasoning trace
    in message.reasoning_content. Persisting it in transcript.jsonl makes the
    trace available for paper analysis even when not replayed in history."""
    resp = NormalizedResponse(
        text="",
        tool_calls=(NormalizedToolCall(id="t1", name="exec", arguments={}),),
        usage=NormalizedUsage(input_tokens=10, output_tokens=2),
        stop_reason="tool_use",
        model="MiniMax-M2.7",
        reasoning_content="step 1: call exec\nstep 2: inspect output",
    )
    e = ai_entry(resp, ts="T")
    assert e["reasoning_content"] == "step 1: call exec\nstep 2: inspect output"


def test_ai_entry_omits_reasoning_content_when_absent() -> None:
    """Providers without a reasoning trace (Anthropic, GLM, etc.) must not get
    a stray empty reasoning_content key in the entry."""
    resp = NormalizedResponse(
        text="ok",
        tool_calls=(),
        usage=NormalizedUsage(input_tokens=10, output_tokens=2),
        stop_reason="stop",
        model="claude-haiku-4-5",
    )
    e = ai_entry(resp, ts="T")
    assert "reasoning_content" not in e


def test_ai_entry_persists_encrypted_reasoning_items_from_raw_response() -> None:
    item = {
        "id": "rs_123",
        "type": "reasoning",
        "encrypted_content": "ciphertext",
        "summary": [],
    }
    resp = NormalizedResponse(
        text="",
        tool_calls=(NormalizedToolCall(id="t1", name="exec", arguments={}),),
        usage=NormalizedUsage(input_tokens=10, output_tokens=2),
        stop_reason="tool_use",
        model="gpt-5.5-2026-04-23",
        raw={"choices": [{"message": {"reasoning_items": [item]}}]},
    )

    e = ai_entry(resp, ts="T")
    assert e["reasoning_items"] == [item]
    assert e["reasoning_items"][0] is not item


def test_ai_entry_omits_reasoning_items_for_other_response_shapes() -> None:
    resp = NormalizedResponse(
        text="ok",
        tool_calls=(),
        usage=NormalizedUsage(),
        stop_reason="stop",
        model="claude-haiku-4-5",
    )

    assert "reasoning_items" not in ai_entry(resp, ts="T")


def test_ai_entry_with_tool_calls_and_usage() -> None:
    resp = NormalizedResponse(
        text="",
        tool_calls=(
            NormalizedToolCall(id="toolu_1", name="setup", arguments={}),
            NormalizedToolCall(id="toolu_2", name="exec", arguments={"cmd": "ls"}),
        ),
        usage=NormalizedUsage(
            input_tokens=120,
            output_tokens=45,
            cache_read_tokens=80,
            cache_creation_tokens=40,
        ),
        stop_reason="tool_use",
        model="claude-sonnet-4-5",
    )
    e = ai_entry(resp, ts="T")
    # Field key ordering matches bench-v8: id, name, args (NOT 'arguments').
    assert e["tool_calls"] == [
        {"id": "toolu_1", "name": "setup", "args": {}},
        {"id": "toolu_2", "name": "exec", "args": {"cmd": "ls"}},
    ]
    assert e["usage"]["input_tokens"] == 120
    assert e["usage"]["output_tokens"] == 45
    assert e["usage"]["cache_read"] == 80
    assert e["usage"]["cache_creation"] == 40


def test_ai_entry_omits_cache_block_when_zero() -> None:
    resp = NormalizedResponse(
        text="ok",
        tool_calls=(),
        usage=NormalizedUsage(input_tokens=10, output_tokens=2),
        stop_reason="end_turn",
        model="x",
    )
    e = ai_entry(resp, ts="T")
    assert "cache_read" not in e["usage"]
    assert "cache_creation" not in e["usage"]


def test_ai_entry_includes_content_blocks_when_thinking_present() -> None:
    blocks = (
        {"type": "thinking", "thinking": "...", "signature": "sig-1"},
        {"type": "tool_use", "id": "t1", "name": "exec", "input": {"cmd": "ls"}},
    )
    resp = NormalizedResponse(
        text="",
        tool_calls=(NormalizedToolCall(id="t1", name="exec", arguments={"cmd": "ls"}),),
        usage=NormalizedUsage(),
        stop_reason="tool_use",
        model="claude-opus-4-6",
        content_blocks=blocks,
    )
    e = ai_entry(resp, ts="T")
    assert e["content_blocks"] == list(blocks)


def test_ai_entry_omits_content_blocks_when_no_thinking() -> None:
    """Don't bloat transcripts on non-thinking turns; the existing
    content / tool_calls fields already cover them."""
    blocks = (
        {"type": "text", "text": "hi"},
        {"type": "tool_use", "id": "t1", "name": "exec", "input": {}},
    )
    resp = NormalizedResponse(
        text="hi",
        tool_calls=(NormalizedToolCall(id="t1", name="exec", arguments={}),),
        usage=NormalizedUsage(),
        stop_reason="tool_use",
        model="claude-haiku-4-5",
        content_blocks=blocks,
    )
    e = ai_entry(resp, ts="T")
    assert "content_blocks" not in e


def test_ai_entry_omits_usage_block_when_zero() -> None:
    resp = NormalizedResponse(
        text="ok",
        tool_calls=(),
        usage=NormalizedUsage(),
        stop_reason="end_turn",
        model="x",
    )
    e = ai_entry(resp, ts="T")
    assert "usage" not in e


def test_tool_message_entry_shape() -> None:
    e = tool_message_entry(
        tool_call_id="toolu_1",
        name="exec",
        content="hello stdout",
        ts="T",
    )
    assert e == {
        "ts": "T",
        "role": "tool",
        "tool_call_id": "toolu_1",
        "name": "exec",
        "content": "hello stdout",
    }


def test_tool_call_log_entry_rounds_duration() -> None:
    e = tool_call_log_entry(
        tool="exec",
        args={"cmd": "ls"},
        result="...",
        duration_s=0.123456,
        ts="T",
    )
    assert e["tool"] == "exec"
    assert e["duration_s"] == 0.123  # 3 decimal places


def test_tool_call_log_entry_handles_none_duration() -> None:
    e = tool_call_log_entry(
        tool="exec", args={}, result="...", duration_s=None, ts="T"
    )
    assert e["duration_s"] is None


def test_grade_log_entry_with_dict_result() -> None:
    e = grade_log_entry(
        path="/rlenv/workspace/poc.js",
        result={"capabilities": {"crash": True}, "reason": "1 capability"},
        duration_s=2.5,
        ts="T",
    )
    assert e["path"] == "/rlenv/workspace/poc.js"
    assert e["result"]["capabilities"]["crash"] is True
    assert e["duration_s"] == 2.5


def test_try_parse_grade_result_dict() -> None:
    raw = '{"capabilities": {"crash": true}, "reason": "x"}'
    parsed = try_parse_grade_result(raw)
    assert isinstance(parsed, dict)
    assert parsed["capabilities"]["crash"] is True


def test_try_parse_grade_result_falls_back_to_string() -> None:
    parsed = try_parse_grade_result("not json")
    assert parsed == "not json"


def test_try_parse_grade_result_array_falls_back_to_string() -> None:
    """A JSON array isn't a dict; we treat it as un-parseable."""
    parsed = try_parse_grade_result("[1,2,3]")
    assert parsed == "[1,2,3]"


# ---------------- TranscriptWriter ----------------


def test_writer_appends_three_files(tmp_path: Path) -> None:
    resp = NormalizedResponse(
        text="hi",
        tool_calls=(NormalizedToolCall(id="toolu_1", name="setup", arguments={}),),
        usage=NormalizedUsage(input_tokens=10, output_tokens=5),
        stop_reason="tool_use",
        model="m",
    )
    with TranscriptWriter(tmp_path) as w:
        w.write_system("sys")
        w.write_human("init")
        w.write_ai(resp)
        w.write_tool_message(tool_call_id="toolu_1", name="setup", content="setup-result")
        w.write_tool_log(
            tool="setup", args={}, result="setup-result", duration_s=0.05
        )
        w.write_grade_log(
            path="/x.js", result={"capabilities": {"crash": True}}, duration_s=1.2
        )

    transcript = (tmp_path / "transcript.jsonl").read_text().splitlines()
    tool_calls = (tmp_path / "tool_calls.jsonl").read_text().splitlines()
    grade_calls = (tmp_path / "grade_calls.jsonl").read_text().splitlines()

    assert len(transcript) == 4  # system + human + ai + tool
    assert len(tool_calls) == 1
    assert len(grade_calls) == 1

    # Roles in order
    assert [json.loads(line)["role"] for line in transcript] == [
        "system",
        "human",
        "ai",
        "tool",
    ]


def test_writer_round_trip_unicode(tmp_path: Path) -> None:
    """ensure_ascii=False preserves non-ASCII content."""
    with TranscriptWriter(tmp_path) as w:
        w.write_system("sys with ünicödé and 中文")
    line = (tmp_path / "transcript.jsonl").read_text().strip()
    obj = json.loads(line)
    assert obj["content"] == "sys with ünicödé and 中文"


def test_writer_creates_run_dir(tmp_path: Path) -> None:
    target = tmp_path / "deeply" / "nested" / "rundir"
    with TranscriptWriter(target) as w:
        w.write_system("ok")
    assert target.exists()
    assert (target / "transcript.jsonl").exists()


def test_writer_appends_across_invocations(tmp_path: Path) -> None:
    """Mode 'a' lets resume re-open and continue writing without truncating."""
    with TranscriptWriter(tmp_path) as w:
        w.write_system("first")
    with TranscriptWriter(tmp_path) as w:
        w.write_human("second")
    lines = (tmp_path / "transcript.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["role"] == "system"
    assert json.loads(lines[1])["role"] == "human"
