# qed_swe_bench architecture

This is the developer-facing description of how the code fits together.
Pair it with `decisions.md` (which captures locked methodology choices
and design rationale, replacing the earlier `PLAN.md`) and `README.md`
(which is end-user-facing).

## Data model — FS ↔ DB bijection

**The filesystem and the SQLite DB are equivalent views of the same
logical data.** Either store can be reconstructed from the other via
the `import` / `export` CLIs.

```
            ┌────────────────────┐         ┌────────────────────┐
            │ runs/<benchmark_id>│         │ data/qed_swe_bench  │
            │  /<host>/          │ ◄────►  │  .sqlite           │
            │  <datetime>/       │         │   `runs` table     │
            │  <run_id>/         │         │   `rlenv_images`   │
            │   job.json         │         │   `meta`           │
            │   score.json       │         │                    │
            │   cost.json        │         │                    │
            │   transcript.jsonl │         │                    │
            │   tool_calls.jsonl │         │                    │
            │   grade_calls.jsonl│         │                    │
            │   config_snapshot  │         │                    │
            │   .yaml            │         │                    │
            │   mcp_stderr.log   │         │                    │
            └────────────────────┘         └────────────────────┘
              ▲                                          ▲
              │                                          │
              │   `qed_swe_bench import <path>`           │
              └──────────────────────────────────────────┤
                                                         │
              ┌──────────────────────────────────────────┘
              │   `qed_swe_bench export <path>`
              ▼
            ┌────────────────────┐
            │  reconstructed     │
            │  flat-text dir     │
            └────────────────────┘
```

Every DB column has a flat-text counterpart in some run-dir artifact;
every artifact field is recorded somewhere queryable in the DB. The
Python CLI writes FS-then-DB today, but the contract is the
**bijection**, not the order. Either store can be the "source of
truth" without architectural change.

The `runs.config_snapshot` column carries the verbatim source YAML
the row was produced under (raw bytes, comments preserved — D-13).
The same bytes also land at `<run_dir>/config_snapshot.yaml`, so the
two halves of the bijection are byte-identical on this artifact and
**`qed_swe_bench rerun <run_id>` is fully self-sufficient from the DB
row alone** — no run-dir access required.

**Run-dir layout** (the path itself encodes scoping):

- `runs/<benchmark_id>/` — top: aggregation by benchmark
- `runs/<benchmark_id>/<host>/` — middle: rsync-from-multiple-hosts
  isolation. `<host>` from `QED_SWE_BENCH_HOST` env or
  `socket.gethostname()` fallback.
- `runs/<benchmark_id>/<host>/<datetime>/` — chronological / archival
- `runs/<benchmark_id>/<host>/<datetime>/<run_id>/` — leaf: the run

`benchmark_id` is a human-readable name (e.g., `v8`, `v8-r2`,
`v8-nudged`) — not a UUID. Naming convention: lowercase, kebab-case,
release-tag suffix for revisions, variant-tag for forks.

**Operational implications**:

- **Multi-host runs**: each EC2 writes to its own local SQLite; rsync
  the run-dirs to a central machine; `qed_swe_bench import` rebuilds
  the central DB. No remote DB writes; no shared SQLite over the
  network.
- **Archival**: `make audit-bundle BENCHMARK_ID=v8` produces a
  self-contained tarball. HuggingFace publication (D-9) is the same
  tarball pattern for external distribution.
- **Backup story**: `tar runs/` is sufficient. DB is regenerable from
  the filesystem at any time.
- **Future Postgres / Drizzle / API removal** (P-6, P-7): each is a
  one-side change. The bijection keeps the other side as recovery.

See `docs/decisions.md` D-10 for the locked rationale.

## Layered view

```
┌─────────────────────────────────────────────────────────────────────────┐
│ qed_swe_bench/cli.py                Typer CLI (benchmark, summary, etc.) │
├─────────────────────────────────────────────────────────────────────────┤
│ runner/orchestrator.py            asyncio.Semaphore over (m,e,s) tuples │
│                                   + per-tuple state machine + DB writes │
│ runner/orchestrator_config.py     EnvSpec/ModelSpec/Budgets/             │
│                                   BenchmarkConfig + parse_config        │
│ runner/run_dir.py                 job/score/cost.json + grade jsonl     │
│ runner/resilience.py              janitor + delete-failed (--retry)     │
│ runner/spend_tracker.py           cap-aware running cost                │
├──────────────────────────────────┬──────────────────────────────────────┤
│ runner/loop.py                   │ historical.py    aggregate.py        │
│   the agent loop                 │   eval/ → runs   runs → md/csv/json  │
├──────────────────┬───────────────┴──────────────────────────────────────┤
│ runner/llm/      │ runner/                                              │
│  factory.py      │  mcp_client.py    docker run → MCP stdio             │
│  base.py         │  image_ref.py     tag/digest/local resolver          │
│  anthropic_native│  transcript.py    JSONL writers (bench-v8 format)    │
│  litellm_client  │  budget.py        weighted tokens, nudges            │
│  mock.py         │  cost.py          provider usage → cost_usd (table)  │
│  tools.py        │  openrouter_recon.py  post-episode OR cost lookup   │
│                  │  capabilities.py  bitmap extraction + merge + scoring│
│                  │  cli_oneshot_grader.py  rl_mcp_v1 post-episode grade │
│                  │  runs_db.py       runs-table CRUD (insert/mark_running│
│                  │                   /update_finished/record_failure)   │
├──────────────────┴──────────────────────────────────────────────────────┤
│ qed_swe_bench/db/schema.py            runs + rlenv_images (SQLite v4)    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key invariants

1. **One agent loop, three providers.** `runner/loop.py:run_episode` is the
   only loop. It talks to whichever `LLMClient` the factory hands it
   (`AnthropicNative`, `LiteLLMClient`, `MockClient`) through the
   `NormalizedResponse` interface in `runner/llm/base.py`. Adding a fourth
   provider is one new file under `runner/llm/`, one prefix in `factory.py`,
   no loop changes.

2. **Provider message shapes are local to LLM clients.** Each client's
   message-list shape (Anthropic blocks vs OpenAI tools format) is built by
   `runner/llm/messages.py:build_messages_for(client_route)`. The loop is
   provider-agnostic past the factory call.

3. **Capability extraction matches bench-v8 byte-for-byte.** Tier-2 golden
   tests in `tests/golden/test_grade_parity.py` pin
   `runner/capabilities.py:best_caps_from_grade_log` against bench-v8's
   `eval/results.py:parse_run` across all 70 historical runs. If you change
   the parser, this test must continue to pass.

4. **Budget arithmetic matches bench-v8's weighted formula.**
   `tokens_used += output + cache_creation + int(cache_read * 0.1)`. The
   cache_read 0.1× weight is Anthropic's billing model and is held by
   `runner/budget.py:CACHE_READ_WEIGHT`. If providers change pricing, update
   the formula and the pricing table in `runner/cost.py` together.

   Per D-11 (turn-as-effort), `token_budget` and `context_budget` are
   *optional with always-report*: setting either to `None` (or `null`
   in YAML) disables per-episode enforcement on that axis while
   `Budget` continues to track `weighted_tokens_used` and
   `peak_per_turn_context`. Both diagnostics land in `score.json` and
   in the `runs` table (nullable INT columns, schema v4) so the same
   benchmark cell looks identical across "budget on" and "budget off"
   runs except for the optional early termination. A turn that
   exceeds the provider's context window propagates as
   `exit_reason=context_window_exceeded` with the provider's verbatim
   message in `failure_reason` — the runner does not pre-declare
   provider windows.

5. **Image refs resolve to immutable digests at run start.**
   `runner/image_ref.py:resolve` produces a `sha256:` digest for every
   accepted ref form. The digest is recorded in `runs.image_digest` and used
   for the actual `docker run`. Re-tagging the registry between resolution
   and execution causes the run to fail loudly rather than silently switch
   images.

6. **`UNIQUE(benchmark_id, model, env_id, seed)` makes `--resume` cheap.**
   No bookkeeping table, no completion markers; just a constraint that lets
   us insert speculatively and skip on collision.

## End-to-end data flow

```
qed_swe_bench benchmark --config v8.json
      │
      ▼
parse_config(JSON) → BenchmarkConfig
      │
      ▼
asyncio.Semaphore(N) over (model, env, seed) tuples
      │
   for each tuple:
      ├─ image_ref.resolve(env.image)            → sha256:...
      ├─ INSERT runs(status='queued')             ← UNIQUE blocks dupes
      ├─ build_client(model_id)                   → NormalizedResponse-shaped client
      ├─ write_job_json(...)                     ← FS half of the bijection
      ├─ UPDATE runs SET status='running' (mark_running) ← committed to run
      ├─ McpDockerSession.start(digest)           → docker run + stdio MCP
      ├─ TranscriptWriter(run_dir)                → 3 JSONL streams
      ├─ Budget(turn=300, token=null, ctx=null)  ← D-11 turn-as-effort
      └─ run_episode(client, sess, transcript, budget, init_prompt, seed)
              │
              while True:
                ├─ if budget.exceeded(): break
                ├─ if budget.should_nudge_*: append nudge text
                ├─ resp = client.complete(messages, tools, seed)
                ├─ transcript.write_ai(resp); messages.append_assistant(resp)
                ├─ budget.tick_ai_turn(resp.usage)
                ├─ if no tool_calls: break
                ├─ for tc in resp.tool_calls:
                │     result = mcp_session.call_tool(tc.name, tc.arguments)
                │     transcript.write_tool_message(...)
                │     transcript.write_tool_log(...)
                │     if tc.name == 'grade':
                │         transcript.write_grade_log(parsed)
                │         merge_capabilities(best_caps, parsed.capabilities)
                ├─ messages.append_tool_results([...])
                └─ if best_caps['ace']: break
              │
              return EpisodeResult(capabilities, exit_reason, runtime, tokens)
      │
   manifest = catalog.get_env(env.id) → metadata.manifest_path → load()
   if manifest.grader_kind == 'cli_oneshot':
        run_cli_oneshot_grader(image, evaluate.command,
                               run_dir, timeout)              → GradeResult
        result.capabilities ← grade.capabilities              # replaces in-MCP
        append grade_calls.jsonl {source: cli_oneshot, ...}
   compute_cost(model_id, totals)              → (cost_usd, source)
   reconcile_run_dir(run_dir)                   → optional or_reconciliation
                                                  (OR cells only; HTTP best-effort)
   compute_score(capabilities)                  → score
   write run_dir/{score.json, cost.json, job.json}
   UPDATE runs(status='succeeded'|'model_failed'|'infra_failed', ...)
```

**Two grading paths, one runner.** bench-v8 V8 envs grade *during* the
episode by calling the in-MCP `grade` tool — the agent loop sees its
results and merges capabilities into `best_caps` (lines `if tc.name ==
'grade':` above). rlenv-mcp envs (detect/solve/patch) grade *after*
the episode by shelling `<evaluate.command>` in a fresh
`--network none --rm` container with the run-dir mounted read-only at
`/run`. The decision is data-driven: `manifest.grader_kind` from the
manifest schema. Envs without a manifest default to the in-MCP path
(V8 backward compat).

## Lifecycle states (matches `runs.status`)

In-flight states (orchestrator-owned; not terminal):

- **`queued`** — row inserted but the per-tuple body has not yet
  committed to running. Either truly waiting for a `max_parallel`
  semaphore slot, or pre-`mark_running` (image resolve, cost-cap
  check, write_job_json).
- **`running`** — the orchestrator has called `mark_running` and the
  episode is actively burning compute (docker spawned, LLM calls in
  flight). `last_heartbeat` refreshes ~every 60s.

Terminal states:

- **`infra_failed`** — image ref didn't resolve, container failed to
  start, MCP session crashed, the per-tuple wallclock timeout fired,
  an uncaught exception escaped the safety wrap, the cost cap was
  reached, or the cli_oneshot grader failed. Failure is on us, not
  the model. Should be retried (use `qed_swe_bench benchmark
  --retry-failed` to delete-and-re-attempt).
- **`model_failed`** — the LLM raised inside `complete()` with
  `exit_reason` starting with `error:` (e.g. persistent rate limit,
  schema rejection, silent reroute caught by `served_matches_requested`).
  Failure is on the provider; recorded as a run with `failure_reason`
  set.
- **`succeeded`** — the loop ran to completion. `exit_reason` is one
  of `no_tool_calls`, `ace_achieved`, `context_window_exceeded` (the
  agent ran the experiment, just terminated for a non-error reason),
  or `budget: turn_budget`. The model may have achieved zero
  capabilities; that's still a successful run with `score=0`.

The PLAN's hard rule: a model producing the wrong answer is a successful run
with low score, *not* an infra failure. Mixing the two pollutes leaderboards.

## Resilience layer (`runner/resilience.py`, `runner/spend_tracker.py`)

Two small modules that keep an unattended sweep recoverable:

- **`mark_stale_queued_as_failed()`** runs at the start of every
  `run_benchmark`. Any row stuck in `status IN ('queued','running')`
  whose `last_heartbeat` is older than 5 min (or, for queued rows
  with no heartbeat, `started_at` older than 30 min) gets flipped to
  `infra_failed` with `failure_reason='stale_queued_recovery'`. The
  UNIQUE slot would block retry forever otherwise. Recovers from
  kill -9 / OOM / crashed orchestrator without manual SQL.
- **`delete_failed_rows(benchmark_id)`** is what `--retry-failed`
  calls. Removes infra_failed + model_failed rows for one benchmark so
  the next sweep re-inserts and re-runs them; succeeded rows are kept.
- **`SpendTracker`** is a tiny `asyncio.Lock`-guarded counter shared
  across every bound tuple coroutine. When `BenchmarkConfig.cost_cap_usd`
  is set, each tuple checks `cap_exceeded()` after `_insert_queued`
  and short-circuits to `infra_failed` if the cap is hit — the expensive
  docker / MCP / episode work never runs.

The orchestrator's per-tuple wrapper layers three guards on top of
`_run_one_body`:

1. `asyncio.wait_for(..., bench.episode_timeout_s)` around the
   docker / MCP / episode block — bounds wedged containers.
2. Outer `try/except` catches anything escaping the body (timeout,
   uncaught exception) and calls `_record_failure_if_row_exists` to
   convert the queued row to `infra_failed` if it exists.
3. The driver loop's `asyncio.gather(..., return_exceptions=True)` —
   even a bug that escapes guard 2 only loses *that* tuple, not the
   whole sweep.

## Audit logging

Each run-dir contains six artifacts: `job.json`, `transcript.jsonl`,
`tool_calls.jsonl`, `grade_calls.jsonl`, `score.json`, `cost.json`, and
`mcp_stderr.log` (the MCP container's stderr, captured via the mcp
SDK's `errlog` parameter on `stdio_client`). Together those let a
post-hoc reward-hacking audit see both sides of the agent ↔ environment
boundary: the assistant's full conversation + tool-call payloads
(agent-visible), and the MCP server's diagnostics + any panics
(server-side).

`scripts/build_audit_bundle.sh` (also `make audit-bundle BENCHMARK_ID=…`)
packs all run-dirs for one benchmark into a tar.gz with a `MANIFEST.sha256`
and a portable `summary.json` (the runs-table dump, no SQLite required).
Receivers verify integrity with `sha256sum -c MANIFEST.sha256`.

## Adding a new task type

The schema's `task_type` column is set per-env. The first shipped task type
was `binary_task` (V8). The supporting infrastructure for additional task
types is layered:

- **`rlenv_images` catalog + `register-dir` CLI** (shipped). One row per
  registered env; metadata blob carries `manifest_path` so the validator
  and runner can resolve the contract.
- **Manifest schema + validator** (shipped). Manifests are no longer flat:
  `contract/schemas/manifest.schema.json` defines required fields including
  `interface_flavor` (`bench_v8` | `rl_mcp_v1`), `evaluate.kind` (`mcp_tool`
  | `cli_oneshot`), `mcp.episode_tools` (subset of the interface registry's
  `mcp_tools`) vs `mcp.evaluation_tools` (must be disjoint — reward-hacking
  guard), `expected_capabilities`, and `integrity_baseline`. The 5-check
  validator (`qed_swe_bench validate-image`) confirms the manifest is
  internally consistent and that the live image still matches it.
- **`rl_mcp_v1` runner branch** (pending). Implement the adapter in
  `runner/`:
  - Spawn the rlenv-mcp container.
  - Use `evaluate.kind=cli_oneshot` to shell `--grade` at episode end (instead
    of calling an in-MCP `grade` tool, which is the reward-hacking surface
    we explicitly forbid for non-V8 task types).
  - Read task-type-specific submission shapes (e.g. patch = repo state).
- **`capability_class` taxonomy + leaderboards** (pending).

The agent loop and the `LLMClient` layer don't change for new task types;
only the `mcp_client` orchestration + `capabilities` extraction differ.
