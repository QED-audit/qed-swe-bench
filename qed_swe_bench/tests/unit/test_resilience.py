"""Unit tests for runner/resilience.py: stale-queued janitor +
delete-failed-rows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from qed_swe_bench.db.schema import connect, init_db
from qed_swe_bench.runner.resilience import (
    STALE_QUEUED_AFTER,
    HeartbeatTracker,
    delete_failed_rows,
    heartbeat_run_ids,
    mark_stale_queued_as_failed,
)


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "qed_swe_bench.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    init_db(db)
    return db


def _insert(con, *, run_id: str, benchmark_id: str, status: str,
            started_at: str, finished_at: str | None = None,
            model: str = "anthropic/claude-haiku-4-5",
            env_id: str = "e1", seed: int = 1) -> None:
    con.execute(
        """
        INSERT INTO runs (
            run_id, benchmark_id, model, env_id, image_ref, image_digest,
            task_type, seed, status, run_dir, started_at, finished_at, provenance
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'native')
        """,
        (run_id, benchmark_id, model, env_id, "local/x:latest",
         "sha256:" + "a" * 64, "binary_task", seed, status,
         "/tmp", started_at, finished_at),
    )


# ---------------- stale-queued janitor ----------------


def test_mark_stale_queued_converts_old_queued_rows(tmp_db: Path) -> None:
    """A queued row started long enough ago becomes infra_failed."""
    now = datetime.now(UTC)
    long_ago = (now - STALE_QUEUED_AFTER - timedelta(minutes=5)).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="stale", benchmark_id="b", status="queued",
                started_at=long_ago)

    n = mark_stale_queued_as_failed(now=now)
    assert n == 1

    with connect(tmp_db) as con:
        row = con.execute("SELECT status, failure_reason, finished_at "
                          "FROM runs WHERE run_id='stale'").fetchone()
    assert row["status"] == "infra_failed"
    assert row["failure_reason"] == "stale_queued_recovery"
    assert row["finished_at"] is not None


def test_mark_stale_queued_leaves_recent_rows_alone(tmp_db: Path) -> None:
    """An in-progress run from a parallel orchestrator must not be stolen."""
    now = datetime.now(UTC)
    recent = (now - timedelta(minutes=2)).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="recent", benchmark_id="b", status="queued",
                started_at=recent)

    n = mark_stale_queued_as_failed(now=now)
    assert n == 0
    with connect(tmp_db) as con:
        status = con.execute("SELECT status FROM runs WHERE run_id='recent'").fetchone()["status"]
    assert status == "queued"


def test_mark_stale_queued_leaves_succeeded_rows_alone(tmp_db: Path) -> None:
    now = datetime.now(UTC)
    long_ago = (now - STALE_QUEUED_AFTER - timedelta(hours=1)).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="ok", benchmark_id="b", status="succeeded",
                started_at=long_ago, finished_at=long_ago)

    n = mark_stale_queued_as_failed(now=now)
    assert n == 0


# ---------------- delete-failed-rows ----------------


def test_delete_failed_rows_removes_only_failed(tmp_db: Path) -> None:
    """Both infra_failed and model_failed are deleted; succeeded stays."""
    now = datetime.now(UTC).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="ok", benchmark_id="b", status="succeeded",
                started_at=now, finished_at=now, env_id="e1", seed=1)
        _insert(con, run_id="bad1", benchmark_id="b", status="infra_failed",
                started_at=now, finished_at=now, env_id="e1", seed=2)
        _insert(con, run_id="bad2", benchmark_id="b", status="model_failed",
                started_at=now, finished_at=now, env_id="e1", seed=3)

    n = delete_failed_rows("b")
    assert n == 2

    with connect(tmp_db) as con:
        survivors = {r["run_id"] for r in con.execute(
            "SELECT run_id FROM runs WHERE benchmark_id='b'"
        ).fetchall()}
    assert survivors == {"ok"}


def test_delete_failed_rows_scoped_by_benchmark(tmp_db: Path) -> None:
    """Failed rows from a different benchmark must NOT be touched."""
    now = datetime.now(UTC).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="b1-bad", benchmark_id="b1", status="infra_failed",
                started_at=now, finished_at=now, env_id="e1", seed=1)
        _insert(con, run_id="b2-bad", benchmark_id="b2", status="infra_failed",
                started_at=now, finished_at=now, env_id="e1", seed=2)

    n = delete_failed_rows("b1")
    assert n == 1

    with connect(tmp_db) as con:
        survivors = {r["run_id"] for r in con.execute("SELECT run_id FROM runs").fetchall()}
    assert survivors == {"b2-bad"}


def test_delete_failed_rows_zero_when_no_failures(tmp_db: Path) -> None:
    n = delete_failed_rows("nonexistent-benchmark")
    assert n == 0


# ---------------- heartbeat: pure SQL helper ----------------


def test_heartbeat_updates_queued_row_timestamp(tmp_db: Path) -> None:
    """heartbeat_run_ids sets last_heartbeat on a queued row."""
    now = datetime.now(UTC)
    long_ago = (now - timedelta(hours=2)).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="r1", benchmark_id="b", status="queued",
                started_at=long_ago)

    n = heartbeat_run_ids(["r1"], now=now)
    assert n == 1

    with connect(tmp_db) as con:
        hb = con.execute(
            "SELECT last_heartbeat FROM runs WHERE run_id='r1'"
        ).fetchone()["last_heartbeat"]
    assert hb == now.isoformat()


def test_heartbeat_skips_non_queued_rows(tmp_db: Path) -> None:
    """A succeeded row must NOT get its heartbeat bumped — that would
    let a finished row mask itself as live forever."""
    now = datetime.now(UTC).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="r1", benchmark_id="b", status="succeeded",
                started_at=now, finished_at=now)

    n = heartbeat_run_ids(["r1"])
    assert n == 0


def test_heartbeat_updates_running_rows_too(tmp_db: Path) -> None:
    """`running` is also an in-flight state owned by the orchestrator.
    The heartbeat tracker doesn't know whether the row was promoted to
    running yet, so it must refresh either status."""
    now = datetime.now(UTC)
    with connect(tmp_db) as con:
        _insert(con, run_id="r-run", benchmark_id="b", status="running",
                started_at=now.isoformat())

    n = heartbeat_run_ids(["r-run"], now=now)
    assert n == 1
    with connect(tmp_db) as con:
        hb = con.execute(
            "SELECT last_heartbeat FROM runs WHERE run_id='r-run'"
        ).fetchone()["last_heartbeat"]
    assert hb == now.isoformat()


def test_heartbeat_empty_input_is_noop(tmp_db: Path) -> None:
    """Empty run_ids → 0 rows updated, no error."""
    assert heartbeat_run_ids([]) == 0


# ---------------- stale-recovery: heartbeat-aware ----------------


def test_stale_recovery_skips_recent_heartbeat(tmp_db: Path) -> None:
    """A queued row with a recent heartbeat is alive — don't reap.

    This is the regression test for the original bug: long V8 episodes
    crossed the 30-min started_at cutoff while still actively running,
    and a sibling-process startup wrongly killed them. With heartbeats,
    a recent heartbeat takes precedence over started_at age.
    """
    now = datetime.now(UTC)
    long_ago = (now - STALE_QUEUED_AFTER - timedelta(minutes=30)).isoformat()
    recent_hb = (now - timedelta(minutes=1)).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="alive", benchmark_id="b", status="queued",
                started_at=long_ago)
        con.execute(
            "UPDATE runs SET last_heartbeat = ? WHERE run_id = 'alive'",
            (recent_hb,),
        )

    n = mark_stale_queued_as_failed(now=now)
    assert n == 0

    with connect(tmp_db) as con:
        status = con.execute(
            "SELECT status FROM runs WHERE run_id='alive'"
        ).fetchone()["status"]
    assert status == "queued"


def test_stale_recovery_reaps_stale_heartbeat(tmp_db: Path) -> None:
    """A queued row with an OLD heartbeat (owner went silent) is reaped."""
    now = datetime.now(UTC)
    started = (now - timedelta(minutes=10)).isoformat()
    stale_hb = (now - timedelta(minutes=10)).isoformat()  # > 5 min default
    with connect(tmp_db) as con:
        _insert(con, run_id="dead", benchmark_id="b", status="queued",
                started_at=started)
        con.execute(
            "UPDATE runs SET last_heartbeat = ? WHERE run_id = 'dead'",
            (stale_hb,),
        )

    n = mark_stale_queued_as_failed(now=now)
    assert n == 1

    with connect(tmp_db) as con:
        row = con.execute(
            "SELECT status, failure_reason FROM runs WHERE run_id='dead'"
        ).fetchone()
    assert row["status"] == "infra_failed"
    assert row["failure_reason"] == "stale_queued_recovery"


def test_stale_recovery_reaps_running_with_stale_heartbeat(tmp_db: Path) -> None:
    """A `running` row whose heartbeat went stale (mid-episode crash,
    OOM, kill -9) is reaped via the heartbeat-staleness branch. Same
    contract as for queued rows; the WHERE clause now covers both."""
    now = datetime.now(UTC)
    started = (now - timedelta(minutes=10)).isoformat()
    stale_hb = (now - timedelta(minutes=10)).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="dead-run", benchmark_id="b", status="running",
                started_at=started)
        con.execute(
            "UPDATE runs SET last_heartbeat=? WHERE run_id='dead-run'",
            (stale_hb,),
        )

    n = mark_stale_queued_as_failed(now=now)
    assert n == 1
    with connect(tmp_db) as con:
        row = con.execute(
            "SELECT status, failure_reason FROM runs WHERE run_id='dead-run'"
        ).fetchone()
    assert row["status"] == "infra_failed"
    assert row["failure_reason"] == "stale_queued_recovery"


def test_stale_recovery_skips_running_with_recent_heartbeat(tmp_db: Path) -> None:
    """A `running` row with a fresh heartbeat is alive — same protection
    as for queued rows. Pin this so the WHERE clause expansion to
    cover `running` doesn't accidentally reap healthy in-flight rows."""
    now = datetime.now(UTC)
    started = (now - timedelta(minutes=20)).isoformat()
    fresh_hb = (now - timedelta(seconds=30)).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="alive-run", benchmark_id="b", status="running",
                started_at=started)
        con.execute(
            "UPDATE runs SET last_heartbeat=? WHERE run_id='alive-run'",
            (fresh_hb,),
        )

    n = mark_stale_queued_as_failed(now=now)
    assert n == 0
    with connect(tmp_db) as con:
        status = con.execute(
            "SELECT status FROM runs WHERE run_id='alive-run'"
        ).fetchone()["status"]
    assert status == "running"


def test_stale_recovery_reaps_no_heartbeat_old_started(tmp_db: Path) -> None:
    """A queued row with NO heartbeat AND old started_at (legacy case)
    still gets reaped via the fallback branch."""
    now = datetime.now(UTC)
    very_old = (now - STALE_QUEUED_AFTER - timedelta(minutes=10)).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="legacy", benchmark_id="b", status="queued",
                started_at=very_old)
        # No UPDATE on last_heartbeat; default NULL.

    n = mark_stale_queued_as_failed(now=now)
    assert n == 1


def test_stale_recovery_skips_no_heartbeat_recent_started(tmp_db: Path) -> None:
    """A brand-new queued row (no heartbeat yet, started recently) is
    NOT reaped — the heartbeat just hasn't fired its first tick yet."""
    now = datetime.now(UTC)
    recent = (now - timedelta(seconds=30)).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="new", benchmark_id="b", status="queued",
                started_at=recent)

    n = mark_stale_queued_as_failed(now=now)
    assert n == 0


# ---------------- HeartbeatTracker: async lifecycle ----------------


@pytest.mark.asyncio
async def test_tracker_add_writes_heartbeat_immediately(tmp_db: Path) -> None:
    """Adding a run_id triggers an immediate single-row heartbeat write —
    no waiting for the next interval tick."""
    now = datetime.now(UTC).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="r1", benchmark_id="b", status="queued",
                started_at=now)

    tracker = HeartbeatTracker()
    await tracker.add("r1")

    with connect(tmp_db) as con:
        hb = con.execute(
            "SELECT last_heartbeat FROM runs WHERE run_id='r1'"
        ).fetchone()["last_heartbeat"]
    assert hb is not None


@pytest.mark.asyncio
async def test_tracker_remove_stops_refreshing(tmp_db: Path) -> None:
    """After remove(), the tracker no longer touches that row."""
    now = datetime.now(UTC).isoformat()
    with connect(tmp_db) as con:
        _insert(con, run_id="r1", benchmark_id="b", status="queued",
                started_at=now)

    tracker = HeartbeatTracker()
    await tracker.add("r1")
    await tracker.remove("r1")

    # Read current heartbeat
    with connect(tmp_db) as con:
        hb_before = con.execute(
            "SELECT last_heartbeat FROM runs WHERE run_id='r1'"
        ).fetchone()["last_heartbeat"]

    # Manually call the internal _loop iteration would update only
    # tracked ids; since we removed, no update should happen.
    # Easier check: the removed set is verified by inspecting the
    # internal state.
    assert "r1" not in tracker._run_ids
    assert hb_before is not None  # was set by add()


@pytest.mark.asyncio
async def test_tracker_start_stop_lifecycle() -> None:
    """start() spawns the loop task; stop() cancels it cleanly."""
    tracker = HeartbeatTracker()
    tracker.start()
    assert tracker._task is not None
    await tracker.stop()
    assert tracker._task is None


@pytest.mark.asyncio
async def test_tracker_remove_unknown_id_is_noop(tmp_db: Path) -> None:
    """Removing a run_id that was never added doesn't error."""
    tracker = HeartbeatTracker()
    await tracker.remove("never-added")  # should not raise
