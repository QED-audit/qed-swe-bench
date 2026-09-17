"""Tier-2 transcript-format compatibility against bench-v8's eval/.

We can't byte-identically reproduce bench-v8's transcript.jsonl from our
NormalizedResponse types (the LangChain → NormalizedResponse conversion is
lossy on some content shapes). What we CAN guarantee — and must — is that
qed_swe_bench READS bench-v8's transcript format without errors and finds
the same load-bearing fields the rest of our code relies on.

This catches regressions from format drift on either side of the boundary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

BENCH_V8_EVAL = Path(
    os.environ.get(
        "BENCH_V8_EVAL_PATH",
        "/nonexistent/bench-v8/eval",  # opt-in via env var; tests skip otherwise
    )
)


@pytest.mark.golden
@pytest.mark.skipif(not BENCH_V8_EVAL.is_dir(), reason="bench-v8/eval not present")
def test_bench_v8_transcripts_parse_and_have_expected_fields() -> None:
    """Each line of every transcript.jsonl is JSON with one of the expected
    roles plus the right field set."""
    transcripts = sorted(BENCH_V8_EVAL.glob("*/transcript.jsonl"))
    assert transcripts, "no transcript.jsonl fixtures found"

    seen_roles: set[str] = set()
    seen_with_usage = 0
    seen_with_tool_calls = 0
    n_lines = 0

    for tpath in transcripts:
        if tpath.stat().st_size == 0:
            continue
        with tpath.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                entry = json.loads(line)  # must parse
                # ts and role are universal
                assert "ts" in entry
                role = entry["role"]
                seen_roles.add(role)
                assert role in {"system", "human", "ai", "tool"}, f"unknown role {role!r}"

                if role == "tool":
                    assert "tool_call_id" in entry
                    assert "name" in entry
                    assert "content" in entry

                if role == "ai":
                    # content may be empty string when only tool_calls.
                    assert "content" in entry
                    if entry.get("tool_calls"):
                        seen_with_tool_calls += 1
                        for tc in entry["tool_calls"]:
                            assert "id" in tc
                            assert "name" in tc
                            assert "args" in tc  # bench-v8 uses 'args', not 'arguments'
                    if entry.get("usage"):
                        seen_with_usage += 1
                        assert "input_tokens" in entry["usage"]
                        assert "output_tokens" in entry["usage"]

    assert {"system", "human", "ai", "tool"}.issubset(seen_roles), (
        f"expected all 4 roles, saw {seen_roles}"
    )
    assert seen_with_tool_calls > 50, (
        f"expected many AI turns with tool calls, saw {seen_with_tool_calls}"
    )
    assert seen_with_usage > 50, (
        f"expected many AI turns with usage_metadata, saw {seen_with_usage}"
    )
    # Sanity: thousands of lines across the historical corpus.
    assert n_lines > 1000, f"only {n_lines} lines parsed — corpus suspiciously small?"


@pytest.mark.golden
@pytest.mark.skipif(not BENCH_V8_EVAL.is_dir(), reason="bench-v8/eval not present")
def test_bench_v8_tool_calls_have_id_name_args_only() -> None:
    """Lock the tool_call schema to (id, name, args). If bench-v8 ever adds
    a field, we want to learn about it before silently dropping it during
    import-eval."""
    transcripts = sorted(BENCH_V8_EVAL.glob("*/transcript.jsonl"))
    extra_fields: set[str] = set()
    for tpath in transcripts:
        if tpath.stat().st_size == 0:
            continue
        with tpath.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("role") != "ai":
                    continue
                for tc in entry.get("tool_calls") or []:
                    extra_fields |= set(tc.keys()) - {"id", "name", "args"}
    assert not extra_fields, (
        f"unexpected tool_call fields in bench-v8 transcripts: {extra_fields}"
    )
