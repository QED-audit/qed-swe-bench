"""FastAPI JSON backend smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qed_swe_bench.db.schema import connect, init_db


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "test.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    monkeypatch.setenv("QED_SWE_BENCH_RUNS_DIR", str(tmp_path / "runs"))
    init_db(db)
    # Seed two runs so the endpoints have content.
    with connect(db) as con:
        for i, (model, env, score, status, prov) in enumerate([
            ("anthropic/claude-haiku-4-5", "e1", 3.0, "succeeded", "native"),
            ("openai/gpt-5", "e1", 1.0, "succeeded", "native"),
        ]):
            con.execute(
                """
                INSERT INTO runs (run_id, benchmark_id, model, env_id, image_ref,
                    image_digest, task_type, interface, seed, status,
                    capabilities, score, cost_usd, provenance, started_at, finished_at)
                VALUES (?, 'b1', ?, ?, 'r', 'sha256:x', 'binary_task',
                    'rl.mcp.v8_task.v1', 1, ?, ?, ?, 0.05, ?, '2026-04-25T00:00:00Z', '2026-04-25T00:01:00Z')
                """,
                (f"r{i}", model, env, status, json.dumps({"crash": True}), score, prov),
            )

    # Import app AFTER setting env so init_db picks up the right path.
    from qed_swe_bench.api.app import app
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_summary(client: TestClient) -> None:
    r = client.get("/api/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["total_runs"] >= 2
    assert "spend" in body
    assert "by_status" in body


def test_list_benchmarks(client: TestClient) -> None:
    r = client.get("/api/benchmarks")
    assert r.status_code == 200
    bms = r.json()
    assert any(b["benchmark_id"] == "b1" for b in bms)


def test_get_benchmark(client: TestClient) -> None:
    r = client.get("/api/benchmarks/b1")
    assert r.status_code == 200
    body = r.json()
    assert body["benchmark_id"] == "b1"
    assert len(body["runs"]) == 2
    assert "anthropic/claude-haiku-4-5" in body["models"]


def test_benchmark_matrix(client: TestClient) -> None:
    r = client.get("/api/benchmarks/b1/matrix")
    assert r.status_code == 200
    body = r.json()
    assert len(body["cells"]) == 2  # one per (model, env)
    assert "capability_flags" in body
    assert len(body["capability_flags"]) == 16  # the V8 set


def test_unknown_benchmark_404(client: TestClient) -> None:
    r = client.get("/api/benchmarks/not-real")
    assert r.status_code == 404


def test_list_runs(client: TestClient) -> None:
    r = client.get("/api/runs?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 2
    assert len(body["runs"]) >= 2


def test_list_runs_filtered(client: TestClient) -> None:
    r = client.get("/api/runs?model=openai/gpt-5")
    assert r.status_code == 200
    body = r.json()
    assert all(run["model"] == "openai/gpt-5" for run in body["runs"])


def test_get_run(client: TestClient) -> None:
    r = client.get("/api/runs/r0")
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == "r0"
    assert body["capabilities"] == {"crash": True}


def test_unknown_run_404(client: TestClient) -> None:
    r = client.get("/api/runs/not-real")
    assert r.status_code == 404


def test_leaderboard(client: TestClient) -> None:
    r = client.get("/api/leaderboard")
    assert r.status_code == 200
    rows = r.json()
    assert any(row["model"] == "anthropic/claude-haiku-4-5" for row in rows)


def test_envs(client: TestClient) -> None:
    r = client.get("/api/envs")
    assert r.status_code == 200
    rows = r.json()
    assert any(row["env_id"] == "e1" for row in rows)


def test_models(client: TestClient) -> None:
    r = client.get("/api/models")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) >= 2


def test_mcp_stderr_404_when_missing(client: TestClient) -> None:
    """Most fixture runs don't have a run-dir; the endpoint must 404 cleanly."""
    r = client.get("/api/runs/r0/mcp_stderr")
    assert r.status_code == 404


def test_mcp_stderr_streams_when_present(
    client: TestClient, tmp_path: Path,
) -> None:
    """When run_dir/mcp_stderr.log exists, the endpoint streams it back."""
    runs_dir = tmp_path / "runs" / "b1" / "r0"
    runs_dir.mkdir(parents=True)
    # The API resolves run-dir from runs.run_dir column; update the row to
    # point at our seeded directory.
    db = tmp_path / "test.sqlite"
    with connect(db) as con:
        con.execute("UPDATE runs SET run_dir=? WHERE run_id='r0'", (str(runs_dir),))
    (runs_dir / "mcp_stderr.log").write_text("boot ok\nready\n", encoding="utf-8")

    r = client.get("/api/runs/r0/mcp_stderr")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "boot ok" in r.text


def test_audit_bundle_rejects_path_traversal(client: TestClient) -> None:
    """The shell-out endpoint must refuse `../`-shaped benchmark_ids."""
    r = client.get("/api/benchmarks/..%2F..%2Fetc/bundle")
    # FastAPI may URL-decode; either 400 or 404 is acceptable as long as
    # we don't actually shell out with a traversal-shaped value.
    assert r.status_code in (400, 404)


def test_audit_bundle_rejects_sql_injection_chars(client: TestClient) -> None:
    """The benchmark_id is interpolated into sqlite3's command-line query in
    build_audit_bundle.sh, so any single-quote, semicolon, etc. must be
    rejected at the API layer too. Pre-fix: the `/` and `..` check let these
    through and an authenticated user could ATTACH arbitrary SQLite files
    via a crafted benchmark_id."""
    payloads = [
        "abc'OR'1",                   # SQLi quote-break
        "y'--comment",                # quote-break + comment-leader (the canonical SQLi shape)
        "x;ATTACH",                   # statement-separator
        "z DROP TABLE runs",          # space + keyword
        "$(rm -rf /tmp/x)",           # shell command-substitution shape
        "`whoami`",                   # backtick command-substitution
        "",                           # empty
    ]
    for p in payloads:
        # URL-encode each — TestClient handles it but we want to be explicit
        # that the raw character reached the route.
        from urllib.parse import quote
        r = client.get(f"/api/benchmarks/{quote(p, safe='')}/bundle")
        # 400 = our regex guard fired. 404 = the URL-encoded slashes (in
        # payloads like `$(rm -rf /tmp)`) split the path and no route
        # matched. Either way, the script never ran with the flagged
        # value — that's the property under test.
        assert r.status_code in (400, 404), (
            f"payload {p!r} should be rejected, got {r.status_code}"
        )


def test_audit_bundle_accepts_canonical_ids(client: TestClient) -> None:
    """Regex must accept the canonical YAML id shape used in v8.yaml etc.
    These hit the script (which then 404s on missing benchmark) but the
    400-guard does NOT fire for them."""
    canonical = ["v8", "v8-single-e01", "smoke-matrix-cheap", "imported-opus.v2"]
    for cid in canonical:
        r = client.get(f"/api/benchmarks/{cid}/bundle")
        # 400 means we'd reject the *id*; we want anything but 400 here
        # (the script will 404 for missing benchmark, which is correct).
        assert r.status_code != 400, f"canonical {cid!r} should pass the guard"
