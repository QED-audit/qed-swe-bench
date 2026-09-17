"""Round-trip tests for the runs table."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from qed_swe_bench.db.schema import connect, init_db


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.sqlite"
    init_db(db)
    return db


def test_init_db_creates_runs_table(tmp_db: Path) -> None:
    with connect(tmp_db) as con:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    names = {r["name"] for r in rows}
    assert "runs" in names
    assert "meta" in names


def test_schema_version_recorded(tmp_db: Path) -> None:
    with connect(tmp_db) as con:
        row = con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert row is not None
    # Bumped to 6 with the addition of the `nudges_used` column
    # (NOT NULL DEFAULT 0). Bumping this requires a migration plan for
    # any existing DB; today we rely on idempotent ADD COLUMN for late
    # additions, plus a one-shot UPDATE to flip imported_from_eval rows
    # to nudges_used=1 at first connect after the migration.
    assert int(row["value"]) == 7


def test_runs_table_has_config_snapshot_column(tmp_db: Path) -> None:
    """D-13: full source YAML lives on the row so reproduce is
    self-sufficient. Stored as TEXT (raw YAML, comments preserved)."""
    with connect(tmp_db) as con:
        cols = {r["name"]: r["type"] for r in con.execute("PRAGMA table_info(runs)").fetchall()}
    assert "config_snapshot" in cols
    assert cols["config_snapshot"] == "TEXT"


def test_runs_table_has_nudges_used_column_not_null_default_zero(
    tmp_db: Path,
) -> None:
    """`nudges_used` records whether mid-episode scaffolding nudges fired.
    NOT NULL DEFAULT 0 so every row has a definite value — the rule going
    forward is `vr-agent imports = 1, native = bool(cfg.nudges)`.
    """
    with connect(tmp_db) as con:
        cols = {
            r["name"]: r
            for r in con.execute("PRAGMA table_info(runs)").fetchall()
        }
    assert "nudges_used" in cols
    col = cols["nudges_used"]
    assert col["type"] == "INTEGER"
    assert col["notnull"] == 1
    # SQLite reports DEFAULT as a string token; integer 0 round-trips as '0'.
    assert col["dflt_value"] == "0"


def test_nudges_used_backfill_flips_imported_from_eval_rows(
    tmp_path: Path,
) -> None:
    """Migration path: a pre-existing DB without `nudges_used` gets the
    column added with DEFAULT 0; vr-agent imports (provenance =
    'imported_from_eval') are then backfilled to 1 by `init_db` at first
    connect. Native rows stay at 0.

    Simulates the migration end-to-end by constructing a DB with the
    pre-migration schema, inserting one native + one imported row,
    then running `init_db` to apply the late column + backfill.
    """
    db = tmp_path / "premigration.sqlite"
    # Build a v5-shaped runs table (no nudges_used column yet) and seed
    # two rows: one native, one imported.
    with sqlite3.connect(str(db)) as con:
        con.execute(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                benchmark_id TEXT NOT NULL,
                model TEXT NOT NULL,
                env_id TEXT NOT NULL,
                image_ref TEXT NOT NULL,
                image_digest TEXT NOT NULL,
                task_type TEXT NOT NULL,
                seed INTEGER NOT NULL,
                status TEXT NOT NULL,
                provenance TEXT NOT NULL DEFAULT 'native',
                UNIQUE(benchmark_id, model, env_id, seed)
            )
            """,
        )
        con.executemany(
            "INSERT INTO runs (run_id, benchmark_id, model, env_id, "
            "image_ref, image_digest, task_type, seed, status, provenance) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("native1", "v8", "anthropic/claude-haiku-4-5", "e1",
                 "img1", "sha256:" + "0" * 64, "binary_task", 1,
                 "succeeded", "native"),
                ("imp1", "imported-opus", "anthropic/claude-opus-4-6",
                 "e01", "img2", "imported:e01",
                 "binary_task", 1, "succeeded", "imported_from_eval"),
            ],
        )
        con.commit()

    # Apply the migration.
    init_db(db)

    with connect(db) as con:
        rows = {
            r["run_id"]: r["nudges_used"]
            for r in con.execute(
                "SELECT run_id, nudges_used FROM runs ORDER BY run_id"
            ).fetchall()
        }
    assert rows == {"native1": 0, "imp1": 1}

    # Idempotent: re-running init_db doesn't re-run the backfill (it
    # only fires when the column is freshly added). A manual flip on
    # the native row should survive a second init_db.
    with connect(db) as con:
        con.execute("UPDATE runs SET nudges_used = 1 WHERE run_id = 'native1'")
    init_db(db)
    with connect(db) as con:
        row = con.execute(
            "SELECT nudges_used FROM runs WHERE run_id = 'native1'"
        ).fetchone()
    assert row["nudges_used"] == 1


def test_init_db_is_idempotent(tmp_db: Path) -> None:
    init_db(tmp_db)
    init_db(tmp_db)  # should not raise


def test_runs_round_trip(tmp_db: Path) -> None:
    caps = {"crash": True, "diff": True, "asan": False, "ace": False}
    with connect(tmp_db) as con:
        con.execute(
            """
            INSERT INTO runs (
                run_id, benchmark_id, model, env_id, image_ref, image_digest,
                task_type, seed, status, capabilities, score, cost_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "r1",
                "bench-1",
                "anthropic/claude-haiku-4-5",
                "sample-stack-bof",
                "local/sample-stack-bof:latest",
                "sha256:abc",
                "binary_task",
                1,
                "succeeded",
                json.dumps(caps),
                2.0,
                0.05,
            ),
        )
        row = con.execute("SELECT * FROM runs WHERE run_id='r1'").fetchone()

    assert row["model"] == "anthropic/claude-haiku-4-5"
    assert row["env_id"] == "sample-stack-bof"
    assert row["image_digest"] == "sha256:abc"
    assert json.loads(row["capabilities"])["crash"] is True
    assert row["score"] == 2.0


def test_unique_constraint_blocks_duplicates(tmp_db: Path) -> None:
    with connect(tmp_db) as con:
        con.execute(
            """
            INSERT INTO runs (
                run_id, benchmark_id, model, env_id, image_ref, image_digest,
                task_type, seed, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("r1", "b", "m", "e", "ref", "sha256:x", "binary_task", 1, "queued"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                """
                INSERT INTO runs (
                    run_id, benchmark_id, model, env_id, image_ref, image_digest,
                    task_type, seed, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                # different run_id but same (benchmark_id, model, env_id, seed)
                ("r2", "b", "m", "e", "ref2", "sha256:y", "binary_task", 1, "queued"),
            )
