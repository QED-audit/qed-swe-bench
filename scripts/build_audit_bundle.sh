#!/usr/bin/env bash
#
# build_audit_bundle.sh — pack one benchmark's run-dirs + a portable summary
# into a tar.gz with a sha256 manifest, suitable for sharing with a third-
# party reward-hacking auditor.
#
# Bundle layout:
#   ./README.txt           — what's in here, how to verify integrity
#   ./MANIFEST.sha256      — sha256 hash of every other file in the bundle
#   ./summary.json         — qed_swe_bench aggregate -f json (no SQLite needed)
#   ./runs/<bench>/<id>/   — per-episode artifacts (transcripts, etc.)
#
# Verify post-extraction:
#   sha256sum -c MANIFEST.sha256
#
# Usage:
#   bash scripts/build_audit_bundle.sh <benchmark_id>
#   make audit-bundle BENCHMARK_ID=<benchmark_id>

set -euo pipefail

BENCHMARK_ID="${1:?usage: $0 <benchmark_id>}"
DB="${QED_SWE_BENCH_DB:-data/qed_swe_bench.sqlite}"
RUNS_DIR="${QED_SWE_BENCH_RUNS_DIR:-runs}"
QED_SWE_BENCH_BIN="${QED_SWE_BENCH_BIN:-.venv/bin/qed-swe-bench}"

# Shell-side injection guard. The benchmark_id flows through unquoted-ish
# contexts further down (sqlite3 string-interpolation, tar/cp paths, the
# tarball filename). Reject anything outside the canonical YAML-id shape
# rather than try to escape every embedding context. The same regex is
# enforced at the API layer (api/app.py /api/benchmarks/{id}/bundle) but
# defense-in-depth: this script also runs from the CLI, where the caller
# could be a wrapped shell with weaker validation.
if [[ ! "$BENCHMARK_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "✗ invalid benchmark_id (must match ^[A-Za-z0-9_.-]+\$): $BENCHMARK_ID" >&2
  exit 2
fi

if [ ! -f "$DB" ]; then
  echo "✗ db not found: $DB" >&2
  exit 2
fi

n_runs=$(sqlite3 "$DB" \
  "SELECT COUNT(*) FROM runs WHERE benchmark_id = '$BENCHMARK_ID'")
if [ "$n_runs" -eq 0 ]; then
  echo "✗ no runs for benchmark_id=$BENCHMARK_ID in $DB" >&2
  echo "   (try: sqlite3 $DB 'SELECT DISTINCT benchmark_id FROM runs')" >&2
  exit 2
fi

ts=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="audit-bundles"
mkdir -p "$OUT_DIR"
TARBALL="$OUT_DIR/${BENCHMARK_ID}-${ts}.tar.gz"

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/runs"

# Per-episode artifacts. Some runs (cost-cap-skipped, image-resolve-failed)
# may not have a run_dir on disk — that's fine, summary.json carries the
# row data and the receiver can detect the gap.
if [ -d "$RUNS_DIR/$BENCHMARK_ID" ]; then
  cp -r "$RUNS_DIR/$BENCHMARK_ID" "$stage/runs/$BENCHMARK_ID"
else
  echo "(note: $RUNS_DIR/$BENCHMARK_ID is missing; bundle will only contain summary.json)"
fi

# Portable JSON dump of the runs table — capability bitmaps already
# expanded back to dicts.
"$QED_SWE_BENCH_BIN" aggregate \
  --benchmark-id "$BENCHMARK_ID" \
  -f json \
  -o "$stage/summary.json"

# README.
cat > "$stage/README.txt" <<EOF
qed_swe_bench audit bundle
benchmark_id: $BENCHMARK_ID
generated_at: $ts (UTC)
n_runs:       $n_runs

Contents
--------
runs/$BENCHMARK_ID/<host>/<datetime>/<run_id>/   (post-2026-05-02)
runs/$BENCHMARK_ID/<utc-iso>__<run_id>/          (legacy / pre-migration)
    job.json           — model, env, seed, image_digest, started_at,
                         git_sha, repro_cmd, env_overrides, interface
    transcript.jsonl   — every system / human / assistant / tool message
    tool_calls.jsonl   — one row per MCP tool call + result + duration
    grade_calls.jsonl  — one row per grade call + parsed capability bitmap
    score.json         — final capabilities + score + exit_reason +
                         finished_at + runtime_s + turns_used + status
    cost.json          — tokens (in/out/cache_*) + cost_usd + cost_source
                         + llm_route + api_base + served_model
    config_snapshot.yaml — copy of the benchmark YAML at run time
    mcp_stderr.log     — MCP container stderr (post-mortem diagnostics)
summary.json           — aggregate of every run's DB row (no SQLite needed)
MANIFEST.sha256        — sha256 of every other file (verify with: sha256sum -c)
README.txt             — this file

Re-importing into a fresh DB (the FS↔DB bijection per D-10):
After extracting, run:

    qed_swe_bench import runs/

This rebuilds the runs table from the flat-text artifacts. No SQLite
file needs to come with the bundle; summary.json is the precomputed
view but the run-dirs alone are sufficient for full hydration.

Verifying integrity
-------------------
After extracting the tarball:

    cd <extracted-dir>
    sha256sum -c MANIFEST.sha256

Any line ending in "OK" is byte-for-byte identical to what was packed;
"FAILED" means the file has been modified post-bundling.

What to look for
----------------
For reward-hacking audit, the highest-signal artifacts are:
- tool_calls.jsonl: a high ratio of identical bash invocations is the
  classic "model is just fuzzing" signal. Compare unique vs total counts:
    jq -s 'group_by(.args) | length' tool_calls.jsonl     # unique
    jq -s 'length' tool_calls.jsonl                        # total
- mcp_stderr.log: MCP server warnings / panics that the agent loop
  doesn't surface to the transcript.
- grade_calls.jsonl + score.json: divergence between what the grader
  reported per-call and the final capabilities — should never happen
  but worth spot-checking.
EOF

# SHA256 manifest of every file we'll ship. Computed before we add the
# manifest itself so the manifest doesn't try to checksum its own
# in-progress contents.
(cd "$stage" && find . -type f ! -name 'MANIFEST.sha256' -print0 \
  | sort -z \
  | xargs -0 sha256sum) > "$stage/MANIFEST.sha256"

tar -czf "$TARBALL" -C "$stage" .

size=$(du -h "$TARBALL" | cut -f1)
echo "wrote $TARBALL ($size, $n_runs run(s))"
sha256sum "$TARBALL"
