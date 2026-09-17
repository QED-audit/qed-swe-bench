"""Aggregate output formats."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from qed_swe_bench.aggregate import aggregate
from qed_swe_bench.db.schema import connect, init_db


@pytest.fixture
def tmp_db_with_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "test.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    init_db(db)
    with connect(db) as con:
        for i, (model, env, seed, caps, score, cost) in enumerate(
            [
                ("anthropic/claude-haiku-4-5", "e1", 1, {"crash": True, "asan": False}, 1.0, 0.05),
                ("anthropic/claude-haiku-4-5", "e1", 2, {"crash": True, "diff": True}, 2.0, 0.06),
                ("anthropic/claude-haiku-4-5", "e2", 1, {}, 0.0, 0.04),
                ("openai/gpt-5", "e1", 1, {"ace": True}, 16.0, 0.50),
            ]
        ):
            con.execute(
                """
                INSERT INTO runs (run_id, benchmark_id, model, env_id, image_ref,
                    image_digest, task_type, seed, status,
                    capabilities, score, cost_usd, turns_used)
                VALUES (?, 'b1', ?, ?, 'r', 'sha256:x', 'binary_task', ?,
                    'succeeded', ?, ?, ?, 5)
                """,
                (f"r{i}", model, env, seed, json.dumps(caps), score, cost),
            )
    return db


def test_aggregate_markdown_includes_each_pair(tmp_db_with_runs: Path) -> None:
    text = aggregate("b1", output_format="markdown")
    assert "claude-haiku-4-5" in text
    assert "gpt-5" in text
    assert "★" in text  # gpt-5's ace glyph
    # Expected (model, env) pairs
    assert "`e1`" in text
    assert "`e2`" in text


def test_aggregate_csv_one_row_per_run(tmp_db_with_runs: Path) -> None:
    text = aggregate("b1", output_format="csv")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 4
    assert {r["env_id"] for r in rows} == {"e1", "e2"}


def test_aggregate_json_structure(tmp_db_with_runs: Path) -> None:
    text = aggregate("b1", output_format="json")
    obj = json.loads(text)
    assert obj["benchmark_id"] == "b1"
    assert obj["n_runs"] == 4
    assert len(obj["runs"]) == 4
    # capabilities are parsed back to dicts
    assert isinstance(obj["runs"][0]["capabilities"], dict)


def test_aggregate_unknown_benchmark_returns_message(tmp_db_with_runs: Path) -> None:
    text = aggregate("nonexistent", output_format="markdown")
    assert "no runs" in text


def test_aggregate_writes_to_file(tmp_db_with_runs: Path, tmp_path: Path) -> None:
    out = tmp_path / "table.md"
    aggregate("b1", output_format="markdown", output=out)
    assert out.exists()
    assert "claude-haiku" in out.read_text()


@pytest.mark.parametrize(
    ("fmt", "extension", "parse"),
    [
        ("csv", "csv", lambda s: list(csv.DictReader(io.StringIO(s)))),
        ("json", "json", json.loads),
    ],
)
def test_aggregate_csv_and_json_file_round_trip(
    tmp_db_with_runs: Path, tmp_path: Path,
    fmt: str, extension: str, parse,
) -> None:
    """`--output <path> -f csv|json` writes a parseable file."""
    out = tmp_path / f"results.{extension}"
    aggregate("b1", output_format=fmt, output=out)
    assert out.exists()
    parsed = parse(out.read_text(encoding="utf-8"))
    if fmt == "csv":
        assert len(parsed) == 4
        assert {r["model"] for r in parsed} == {
            "anthropic/claude-haiku-4-5",
            "openai/gpt-5",
        }
    else:
        assert parsed["n_runs"] == 4
        assert parsed["benchmark_id"] == "b1"


# ---------------- --compare-with regime comparison ----------------


@pytest.fixture
def tmp_db_with_two_regimes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Two benchmark_ids on the same (model, env) cell so we can test the
    comparison view: same primary regime gets crash, secondary adds diff."""
    db = tmp_path / "test.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    init_db(db)
    fixtures = [
        # primary regime: nudges=false equivalent — score 1, only crash
        ("v8", "anthropic/claude-haiku-4-5", "e1", 1, {"crash": True}, 1.0),
        ("v8", "anthropic/claude-haiku-4-5", "e1", 2, {"crash": True}, 1.0),
        # secondary regime: nudges=true equivalent — score 2, adds diff
        ("v8-nudged", "anthropic/claude-haiku-4-5", "e1", 1,
         {"crash": True, "diff": True}, 2.0),
        ("v8-nudged", "anthropic/claude-haiku-4-5", "e1", 2,
         {"crash": True, "diff": True}, 2.0),
        # cell present only in primary (model exists in primary, not secondary)
        ("v8", "openai/gpt-5", "e1", 1, {"asan": True}, 1.0),
    ]
    with connect(db) as con:
        for i, (bid, model, env, seed, caps, score) in enumerate(fixtures):
            con.execute(
                """
                INSERT INTO runs (run_id, benchmark_id, model, env_id, image_ref,
                    image_digest, task_type, seed, status,
                    capabilities, score, cost_usd, turns_used)
                VALUES (?, ?, ?, ?, 'r', 'sha256:x', 'binary_task', ?,
                    'succeeded', ?, ?, 0.10, 5)
                """,
                (f"r{i}", bid, model, env, seed, json.dumps(caps), score),
            )
    return db


def test_compare_markdown_per_cell_delta(tmp_db_with_two_regimes: Path) -> None:
    """Comparison table shows score delta + caps gained by secondary."""
    text = aggregate("v8", compare_with="v8-nudged", output_format="markdown")
    # Header reflects both regimes
    assert "`v8`" in text and "`v8-nudged`" in text
    # The shared (haiku, e1) cell: primary 1.0, secondary 2.0, Δ +1.0, gained=diff
    assert "1.0" in text and "2.0" in text
    assert "+1.0" in text
    assert "diff" in text  # gained-by-secondary cell


def test_compare_markdown_handles_cell_only_in_primary(
    tmp_db_with_two_regimes: Path,
) -> None:
    """Cells present in only one regime should still appear, with `-` for
    the missing side rather than crashing."""
    text = aggregate("v8", compare_with="v8-nudged", output_format="markdown")
    # gpt-5 / e1 exists only in v8, not v8-nudged
    assert "gpt-5" in text
    # Score of 1.0 in primary, "-" in secondary
    lines = [l for l in text.splitlines() if "gpt-5" in l]
    assert lines, "gpt-5 row missing"
    # The secondary score field should be "-" (no rows for that cell in v8-nudged)
    assert "| - |" in lines[0] or " - " in lines[0]


def test_compare_json_structured_output(tmp_db_with_two_regimes: Path) -> None:
    """JSON compare emits per-cell structure with mean scores and gained caps."""
    text = aggregate("v8", compare_with="v8-nudged", output_format="json")
    parsed = json.loads(text)
    assert parsed["primary_id"] == "v8"
    assert parsed["secondary_id"] == "v8-nudged"
    # Find the shared cell
    haiku_cells = [
        c for c in parsed["cells"]
        if c["model"] == "anthropic/claude-haiku-4-5" and c["env_id"] == "e1"
    ]
    assert len(haiku_cells) == 1
    cell = haiku_cells[0]
    assert cell["primary"]["mean_score"] == 1.0
    assert cell["secondary"]["mean_score"] == 2.0
    assert cell["delta_score"] == pytest.approx(1.0)
    assert cell["caps_gained_by_secondary"] == ["diff"]
    assert cell["caps_lost_by_secondary"] == []


def test_compare_no_runs_in_either_regime(tmp_path: Path, monkeypatch) -> None:
    """Empty DB → friendly message rather than crash."""
    db = tmp_path / "empty.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    init_db(db)
    text = aggregate(
        "nonexistent", compare_with="also-nonexistent", output_format="markdown"
    )
    assert "no runs for either" in text
