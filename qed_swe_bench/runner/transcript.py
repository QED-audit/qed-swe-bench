"""JSONL writers matching bench-v8/agent.py's exact format.

Three log files per episode, written in lockstep with the agent loop:

  transcript.jsonl   one entry per chat message (system, human, ai, tool)
  tool_calls.jsonl   one entry per MCP tool call paired with its result
  grade_calls.jsonl  one entry per grade() call with the parsed result

Format matches bench-v8 so:
  - the existing 70 historical Opus runs in bench-v8/eval/ ingest cleanly via
    `qed_swe_bench import-eval` without format conversion;
  - tier-2 golden parity tests can compare qed_swe_bench's writes against
    bench-v8's recorded transcripts byte-for-byte.

JSON encoding: ensure_ascii=False (preserves non-ASCII content as-is) plus a
trailing newline per record. Files are line-flushed after every record so
killing the process mid-run still leaves a complete prefix on disk.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from io import TextIOBase
from pathlib import Path
from typing import Any

from qed_swe_bench.runner.llm.base import NormalizedResponse


def now_iso() -> str:
    """ISO-8601 timestamp with timezone, matching bench-v8's `_now`."""
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Append helper (private to module; tests use TranscriptWriter directly)
# ---------------------------------------------------------------------------


def _write(fh: TextIOBase, obj: dict[str, Any]) -> None:
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fh.flush()


# ---------------------------------------------------------------------------
# Builders for transcript.jsonl entries
# ---------------------------------------------------------------------------


def system_entry(content: str, *, ts: str | None = None) -> dict[str, Any]:
    return {"ts": ts or now_iso(), "role": "system", "content": content}


def human_entry(content: str, *, ts: str | None = None) -> dict[str, Any]:
    return {"ts": ts or now_iso(), "role": "human", "content": content}


def marker_entry(content: str, *, ts: str | None = None) -> dict[str, Any]:
    """Operational marker: visible to ops/audit but NEVER replayed into the
    LLM context on resume. Use for `[resume]` checkpoints and similar
    metadata that would otherwise bias the model if fed back as a user
    turn (`role=human`)."""
    return {"ts": ts or now_iso(), "role": "marker", "content": content}


def ai_entry(resp: NormalizedResponse, *, ts: str | None = None) -> dict[str, Any]:
    """Convert a NormalizedResponse into a transcript.jsonl entry.

    Mirrors agent.py:_message_to_dict for AIMessage:
      - content: text portion (string; bench-v8 uses str when LangChain emits str)
      - tool_calls: [{"id", "name", "args"}] (note "args", not "arguments")
      - usage: {"input_tokens", "output_tokens", "cache_read", "cache_creation"}

    We additionally emit `served_model` (the model id the provider echoes back
    in the response — this is what was *actually* served, which can differ from
    the model id we sent if the provider routes / aliases / silently downgrades)
    and, when nonzero, `usage.reasoning_tokens` (visible reasoning-tier output
    for OpenAI reasoning models). These are qed_swe_bench extensions; bench-v8
    transcripts predate them and are unaffected.
    """
    entry: dict[str, Any] = {
        "ts": ts or now_iso(),
        "role": "ai",
        "content": resp.text or "",
        "stop_reason": resp.stop_reason,
    }
    if resp.model:
        entry["served_model"] = resp.model
    if resp.tool_calls:
        entry["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "args": tc.arguments}
            for tc in resp.tool_calls
        ]
    if resp.usage.input_tokens or resp.usage.output_tokens:
        entry["usage"] = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        if resp.usage.cache_read_tokens or resp.usage.cache_creation_tokens:
            entry["usage"]["cache_read"] = resp.usage.cache_read_tokens
            entry["usage"]["cache_creation"] = resp.usage.cache_creation_tokens
        if resp.usage.reasoning_tokens:
            entry["usage"]["reasoning_tokens"] = resp.usage.reasoning_tokens
    if any(
        b.get("type") in ("thinking", "redacted_thinking")
        for b in resp.content_blocks
    ):
        entry["content_blocks"] = [dict(b) for b in resp.content_blocks]
    # OpenAI-compat reasoning trace (Moonshot, MiniMax, etc.). LiteLLM
    # surfaces it via message.reasoning_content; persisting here makes
    # the trace visible in transcript.jsonl for paper analysis even when
    # we don't replay it back in the next-turn history.
    if resp.reasoning_content:
        entry["reasoning_content"] = resp.reasoning_content
    # LiteLLM's Responses bridge exposes encrypted OpenAI reasoning state on
    # the raw assistant message. Persist it so an auditor can verify that the
    # all-turns treatment returned replayable state on every applicable turn.
    choices = resp.raw.get("choices") or []
    raw_message = choices[0].get("message", {}) if choices else {}
    reasoning_items = raw_message.get("reasoning_items") or []
    if reasoning_items:
        entry["reasoning_items"] = [dict(item) for item in reasoning_items]
    # Provider-side response id, captured when present. For OpenRouter cells
    # this is the `gen-XXX...` id reused by runner/openrouter_recon.py to
    # fetch the authoritative billed cost via GET /api/v1/generation.
    if resp.provider_response_id:
        entry["provider_response_id"] = resp.provider_response_id
    return entry


def tool_message_entry(
    *, tool_call_id: str, name: str, content: str, ts: str | None = None
) -> dict[str, Any]:
    """Transcript entry for a tool result.

    Mirrors _message_to_dict for ToolMessage: tool_call_id, name, content.
    """
    return {
        "ts": ts or now_iso(),
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content,
    }


# ---------------------------------------------------------------------------
# tool_calls.jsonl + grade_calls.jsonl entry shapes
# ---------------------------------------------------------------------------


def tool_call_log_entry(
    *,
    tool: str,
    args: dict[str, Any] | None,
    result: str,
    duration_s: float | None,
    ts: str | None = None,
) -> dict[str, Any]:
    """One row of tool_calls.jsonl. Matches agent.py:404-412."""
    return {
        "ts": ts or now_iso(),
        "tool": tool,
        "args": args,
        "result": result,
        "duration_s": round(duration_s, 3) if duration_s is not None else None,
    }


def grade_log_entry(
    *,
    path: str | None,
    result: dict[str, Any] | str,
    duration_s: float | None,
    ts: str | None = None,
) -> dict[str, Any]:
    """One row of grade_calls.jsonl. Matches agent.py:424-429."""
    return {
        "ts": ts or now_iso(),
        "path": path,
        "result": result,
        "duration_s": round(duration_s, 3) if duration_s is not None else None,
    }


def try_parse_grade_result(content: str) -> dict[str, Any] | str:
    """Parse a grade() tool result.

    The MCP server returns JSON-encoded text. If parsing fails (older format,
    error string, etc.) we fall back to the raw string; aggregate.py treats
    string-typed results as un-parseable but still preserves them.
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
    if not isinstance(parsed, dict):
        return content
    return parsed


# ---------------------------------------------------------------------------
# TranscriptWriter — owns the three open file handles for an episode
# ---------------------------------------------------------------------------


class TranscriptWriter:
    """Writes the three JSONL streams for one episode.

    Use as a context manager to ensure files are flushed and closed:

        with TranscriptWriter(run_dir) as t:
            t.write_system("...")
            t.write_ai(resp)
            t.write_tool(tool_call_id="...", name="exec", content="...")
            t.write_tool_log(tool="exec", args={...}, result="...", duration_s=0.1)
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._transcript: TextIOBase | None = None
        self._tool_calls: TextIOBase | None = None
        self._grade_calls: TextIOBase | None = None

    def __enter__(self) -> TranscriptWriter:
        self._transcript = (self.run_dir / "transcript.jsonl").open("a", encoding="utf-8")
        self._tool_calls = (self.run_dir / "tool_calls.jsonl").open("a", encoding="utf-8")
        self._grade_calls = (self.run_dir / "grade_calls.jsonl").open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for fh in (self._transcript, self._tool_calls, self._grade_calls):
            if fh is not None:
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass

    # ----- transcript.jsonl --------------------------------------------------

    def write_system(self, content: str, *, ts: str | None = None) -> None:
        assert self._transcript is not None
        _write(self._transcript, system_entry(content, ts=ts))

    def write_human(self, content: str, *, ts: str | None = None) -> None:
        assert self._transcript is not None
        _write(self._transcript, human_entry(content, ts=ts))

    def write_marker(self, content: str, *, ts: str | None = None) -> None:
        """Append an ops-only marker (role=marker) — not replayed by resume."""
        assert self._transcript is not None
        _write(self._transcript, marker_entry(content, ts=ts))

    def write_ai(self, resp: NormalizedResponse, *, ts: str | None = None) -> None:
        assert self._transcript is not None
        _write(self._transcript, ai_entry(resp, ts=ts))

    def write_tool_message(
        self, *, tool_call_id: str, name: str, content: str, ts: str | None = None
    ) -> None:
        assert self._transcript is not None
        _write(
            self._transcript,
            tool_message_entry(tool_call_id=tool_call_id, name=name, content=content, ts=ts),
        )

    # ----- tool_calls.jsonl --------------------------------------------------

    def write_tool_log(
        self,
        *,
        tool: str,
        args: dict[str, Any] | None,
        result: str,
        duration_s: float | None,
        ts: str | None = None,
    ) -> None:
        assert self._tool_calls is not None
        _write(
            self._tool_calls,
            tool_call_log_entry(
                tool=tool, args=args, result=result, duration_s=duration_s, ts=ts
            ),
        )

    # ----- grade_calls.jsonl -------------------------------------------------

    def write_grade_log(
        self,
        *,
        path: str | None,
        result: dict[str, Any] | str,
        duration_s: float | None,
        ts: str | None = None,
    ) -> None:
        assert self._grade_calls is not None
        _write(
            self._grade_calls,
            grade_log_entry(path=path, result=result, duration_s=duration_s, ts=ts),
        )
