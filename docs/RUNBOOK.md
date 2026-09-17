# RUNBOOK — Methodology for operating the V8 benchmark

## Goal

The deliverable is the capability matrix declared in
[`benchmarks/v8.yaml`](../benchmarks/v8.yaml) — N models × 41 V8
bugs × M seeds, each at 300 turns, scored on a 16-flag capability
ladder from coverage to engine primitives to ACE. (The previous
14-bug baseline lives on as [`benchmarks/v8-small.yaml`](../benchmarks/v8-small.yaml)
for cheap iteration.)

This document is the **canonical operator's methodology** for running
that benchmark. It walks through the lifecycle from "convince yourself
one cell works" through "publish a matrix release" — with concrete
commands, decision gates, parallelization recipe, and failure-mode
catalog.

## Local working-copy convention

This file is the methodology — read-only reference. To track your own
progress through a release, **make a local copy** that you check items
off in:

```bash
make runbook-local        # bootstraps RUNBOOK.local.md from this file
# OR
cp docs/RUNBOOK.md RUNBOOK.local.md
```

`RUNBOOK.local.md` (and any `*.local.md` file) is gitignored so your
checkboxes, in-flight notes, and operator-specific context don't leak
into commits. Update the canonical `docs/RUNBOOK.md` when the
methodology itself changes; update your `RUNBOOK.local.md` while
running through it.

## Cross-references

- `docs/architecture.md` — system layout + FS↔DB bijection (D-10)
- `docs/decisions.md` — locked methodology choices (D-N entries)
- `docs/FINDINGS.md` — relevant observations log (F-N entries)

The runbook is **stage-aware**: early iteration is messy and you'll
restart often; steady-state matrix runs are reproducible; publication
graduates a working set into a release. Sections are ordered roughly
in the order you'd hit them.

---

## 1. Pre-flight (one-time before any benchmark run)

```
[ ] `make doctor` clean (env / docker / deps OK)
[ ] `make test` clean (~500 unit tests, no API spend)
[ ] docker daemon up: `docker info > /dev/null && echo OK`
[ ] .env has the right keys (presence-check, don't print values):
     [ ] ANTHROPIC_API_KEY
     [ ] OPENAI_API_KEY
     [ ] GEMINI_API_KEY
     [ ] OPENROUTER_API_KEY
     [ ] ZAI_API_KEY
     [ ] MOONSHOT_API_KEY
[ ] ECR auth fresh (token expires daily):
     `aws ecr get-login-password --region us-east-1 | \
        docker login --username AWS --password-stdin \
        990678687027.dkr.ecr.us-east-1.amazonaws.com`
[ ] Bug images either pulled or buildable:
     `docker images | grep qed_swe_bench` lists v8-e* tags
     If empty: pull the one you'll smoke first, e.g.
     `docker pull 302524490333.dkr.ecr.us-east-1.amazonaws.com/swe-bench/v8-r2:e01`
[ ] `qed_swe_bench summary` runs without error and shows current state
[ ] `git status` clean (or you know which branch / WIP you're on)
```

If any of these fail, **fix before running anything that costs money**.

---

## 2. Iteration phase: prove the pipeline on one cell

**Goal**: convince yourself one (model, env, seed) tuple works
end-to-end before scaling. Recipe is the verification ladder
(caching preflight → 20-turn smoke → full 300-turn → audit), all
described below.

Pick a cheap, working bug as your "proven pipeline" anchor.
`v8-e01` is the standard choice — Wasm bug, image pulls
cleanly, every model tested has reached at least cov_func.

```
[ ] caching preflight (per-provider, ~$0.05 each, total ~$0.11):
     [ ] all three at once:
         `pytest qed_swe_bench/tests/integration/test_caching.py -m slow -v`
     [ ] one provider at a time (skips when its key is unset):
         `pytest qed_swe_bench/tests/integration/test_caching.py::test_anthropic_cache -m slow`
         `pytest qed_swe_bench/tests/integration/test_caching.py::test_openai_cache_and_reasoning -m slow`
         `pytest qed_swe_bench/tests/integration/test_caching.py::test_gemini_implicit_cache -m slow`
     [ ] Z.ai/Moonshot: smoke handles it (no dedicated preflight yet)
     [ ] OR-routed: skip; cache is provider-dependent (see F-5 in FINDINGS.md)
[ ] 20-turn smoke on the chosen bug (~$0.10–$1 per model):
     `qed_swe_bench benchmark --config benchmarks/v8.yaml \
        --models <m> --envs v8-e01 --seeds 1 \
        --turn-budget 20 --cost-cap-usd 2`
[ ] inspect cost.json: served_model matches request, tokens_cache_read > 0
     `cat runs/v8/<latest>/cost.json | jq .`
[ ] full 300-turn on the same bug. **Match the cap to the model tier**
     (cap stops *new tuples* from being scheduled — it doesn't kill an
     in-flight episode, so a too-low cap on a single episode mostly
     just produces a misleading exit_reason on subsequent tuples; but
     for confidence, size the cap above the empirical full-episode cost):

     | Model | Empirical full-ep cost | Recommended cap |
     |---|---|---|
     | openai/gpt-5.5 (xhigh) | ~$52 | **--cost-cap-usd 80** |
     | anthropic/claude-opus-4-7 | ~$25 | **--cost-cap-usd 40** |
     | gemini/gemini-3.1-pro-preview | ~$5 | **--cost-cap-usd 15** |
     | zai/glm-5.1 | ~$5.50 | **--cost-cap-usd 15** |
     | openrouter/minimax/minimax-m2.7 | ~$5 | **--cost-cap-usd 15** |
     | moonshot/kimi-k2.6 | TBD (slow ep, modest tokens) | **--cost-cap-usd 15** |
     | openrouter/qwen/qwen3-coder (480B) | TBD | **--cost-cap-usd 15** |

     Rule of thumb: **2× the model's empirical full-ep cost** for
     comfortable runway, or 3× when first-running a model whose cost
     shape is unknown. Examples:
     `qed_swe_bench benchmark --config benchmarks/v8.yaml \
        --models openai/gpt-5.5 --envs v8-e01 --seeds 2 \
        --cost-cap-usd 80`
[ ] make audit BENCHMARK_ID=v8 → confirm C1–C11 clean
     (INFO findings on C7/C8 are usually fine; HIGH/MEDIUM = stop)
[ ] log finding to docs/FINDINGS.md if anything surprising
     (the findings-log skill in .claude/skills/ should auto-trigger)
```

**Decision gate**: if the full-300-turn cost matches your extrapolation
within ±50% AND the audit is clean, the cell graduates to **section 3**.

If cost is way off or audit fires HIGH:
- C10 (served_model_mismatch) → provider rerouted; check provider status
- C11 (reasoning_silently_dropped) → reasoning_effort param ignored
- C1/C2/C3 (paths/grader-fd) → reward-hacking attempt; investigate manually
- Cost variance → check `cost.py` registration vs invoice

---

## 3. Iteration phase: scale one model across the bug set

**Goal**: same model, all 14 bugs, single seed first. Catches
bug-specific quirks (some bugs the model OOMs, some have prompts that
trip refusals, some cost wildly more than the median).

```
[ ] benchmark --models <m> --seeds 1 --cost-cap-usd <budget>
    (no --envs filter; runs all 14 bugs. <budget> = 14 × per-ep cap from §2)
[ ] make audit BENCHMARK_ID=v8
[ ] inspect: any bugs where cost > 2× the median?
     `sqlite3 data/qed_swe_bench.sqlite "
        SELECT env_id, ROUND(cost_usd,2) FROM runs
        WHERE benchmark_id='v8' AND model=<m>
        ORDER BY cost_usd DESC"`
     → flag for tuning
[ ] inspect: any bugs scoring 0 across the board?
     → note for later analysis (might be model-bug interaction or image issue)
[ ] inspect: any infra_failed?
     → check failure_reason; rerun selectively with --retry-failed
```

Then **n=5 for that model** (the four remaining seeds):

```
[ ] benchmark --models <m> --seeds 2,3,4,5 --cost-cap-usd <4×budget>
    (or --retry-failed to clear infra_failed rows from the prior pass)
[ ] make audit
[ ] aggregate --benchmark-id v8 -f json -o runs/aggregate.<m>.json
     (capture a snapshot of the model's cell)
[ ] log decision to decisions.md if the model's behavior locks anything in
```

---

## 4. Parallelization recipe (steady state)

Three nested layers of parallelism: within-process (`max_parallel`),
across-process on one machine (multiple terminals), and across-machine
(multi-EC2). Pick the layer to match the matrix size.

### 4.1 Within one process

`max_parallel: 2` is the v8.yaml default. Each
parallel slot owns a docker container + an LLM API client.

- Crank up to 4–6 if you have API quota and docker daemon headroom
- Check `docker stats` for memory pressure
- API rate limits are usually the binding constraint, not local CPU

### 4.2 Across processes on one machine

Safe pattern: **partition (model, env, seed) slices across processes**,
never overlap.

```
[ ] partition strategy chosen (typical: one process per model):
     terminal 1: --models gemini/gemini-3.1-pro-preview
     terminal 2: --models zai/glm-5.1
     terminal 3: --models moonshot/kimi-k2.6
[ ] each process gets its own --cost-cap-usd matched to that cell's budget
[ ] all processes share one DB (data/qed_swe_bench.sqlite) — WAL handles concurrency
[ ] aggregate progress from a sixth terminal:
     `watch -n 30 'qed_swe_bench summary'`
     or
     `watch -n 30 "sqlite3 data/qed_swe_bench.sqlite \
       \"SELECT model, status, COUNT(*) FROM runs \
         WHERE benchmark_id='v8' GROUP BY model, status\""`
```

**Don't** run two processes hitting overlapping tuples. The DB
`UNIQUE(benchmark_id, model, env_id, seed)` constraint will dedup, but
you'll race on the API and burn duplicate spend. Also don't share
`cost_cap_usd` budgets across processes — each cap is per-process.

### 4.3 Across machines (multi-EC2)

For matrix-tier runs (490 episodes × maybe 30 min average = ~250
machine-hours), spreading across multiple EC2 instances is the right
scaling. **One instance per model** is the natural partition.

#### What scales, what doesn't

| Resource | Per-machine | Total (n machines) |
|---|---|---|
| CPU / docker daemon headroom | dedicated | n× |
| Github pull bandwidth | per-instance | n× |
| Wall-clock for the matrix | n× faster | — |
| API quota (provider-side) | shared via key | **same** (quota is per-key, not per-IP) |
| Cost cap | per-instance | summed manually |
| SQLite DB | per-instance | merged after |

**API quota does NOT scale with instance count.** Spinning up 7 EC2s
sharing the same `OPENAI_API_KEY` doesn't 7× your quota. If a model
hits TPM caps on one machine, more machines won't help — the bottleneck
is the provider, not the local runner.

#### Partition pattern

One EC2 per model, isolated DBs, merge after:

```
instance-1 (Gemini):    --models gemini/gemini-3.1-pro-preview --cost-cap-usd 350
instance-2 (Z.ai):      --models zai/glm-5.1                  --cost-cap-usd 200
instance-3 (Moonshot):  --models moonshot/kimi-k2.6            --cost-cap-usd 100
```

Each instance: own EBS volume, own SQLite, own .env (or pull keys from
AWS Secrets Manager), own docker daemon, own cached images.

#### Per-instance setup checklist

```
[ ] AMI choice: prebake with bug images cached (recommended) or pull
    on first launch (~90 min × 14 bugs)
[ ] EBS sized for 14 × 80GB images + ~500MB run artifacts ≈ 1.5TB minimum
[ ] .env injected (Secrets Manager → systemd EnvironmentFile recommended)
[ ] git clone the repo at the release SHA you intend to run
[ ] make doctor + make test on the instance to confirm parity
[ ] tmux / nohup the benchmark process so SSH disconnects don't kill it
[ ] log to S3 or persistent volume so a terminated spot instance
    doesn't lose run-dirs
```

#### Result-merge step (after all instances complete) — D-10 bijection

The bijection (D-10) makes multi-host merge a one-liner: rsync the
run-dirs, then `qed_swe_bench import`. The DB rebuilds itself from the
flat-text artifacts.

```
[ ] rsync each instance's runs/ to a central location:
     `rsync -av instance-N:qed_swe_bench/runs/ ~/merged-runs/`
     The layered layout `runs/<benchmark_id>/<host>/<datetime>/<run_id>/`
     means cross-instance rsync naturally lands non-colliding paths.
[ ] (optional) tar + sha256 the merged tree as a release artifact:
     `tar czf v8-r1-runs.tar.gz ~/merged-runs/v8/`
     `sha256sum v8-r1-runs.tar.gz`
[ ] import on the central machine (rebuilds the DB from FS):
     `qed_swe_bench import ~/merged-runs/`
[ ] aggregate:
     `qed_swe_bench aggregate --benchmark-id v8 -f json -o release.json`
[ ] make audit BENCHMARK_ID=v8 (post-merge)
```


---


### Filesystem ↔ DB bijection commands

```
qed_swe_bench import <path>      # rebuild DB from filesystem
                                # (idempotent; e.g., after rsync from EC2)
qed_swe_bench export <target>    # write flat-text for DB rows whose
                                # run-dir is missing on disk (recovery)
```

### Re-running one cell from a row (D-13)

```
qed_swe_bench rerun <run_id>            # re-run the cell using the
                                       # YAML stored in runs.config_snapshot
qed_swe_bench rerun <run_id> --dry-run  # print the resolved single-tuple
                                       # BenchmarkConfig, exit without launch
qed_swe_bench rerun <run_id> \
  --cost-cap-usd 5                     # override the snapshot's cost cap
```

`rerun` is a fresh episode through the native qed_swe_bench loop — same
config, new LLM trajectory. It is NOT a deterministic replay. For
deterministic tool-call replay against the live grader (no LLM calls,
just verifying the recorded tool sequence still produces the same
caps), use `qed_swe_bench audit --reproduce <run_id>`.

The DB row is self-sufficient: the original
`<run_dir>/config_snapshot.yaml` on disk is not required for post-D-13
rows. `data/qed_swe_bench.sqlite` can be copied to a fresh machine with
no `runs/` directory and `qed_swe_bench rerun <run_id>` still works.

For legacy rows (imported from run-dirs that predate the
`config_snapshot` column, or ones produced by `--mock-llm` / `--test`
which have no source YAML), the column is NULL and rerun falls back
to reading `<run_dir>/config_snapshot.yaml` (with a yellow warning).
Rows with neither (e.g. `--mock-llm` / `--test`) are not rerunnable —
the subcommand exits 1 with a clear error. Rows with provenance
`imported_from_*` (codex, vr-agent, etc.) are also refused — those
came from a different agent harness and the row's `repro_cmd` column
points at the right re-run path for the source harness.

### What's still missing (smaller list now)

```
[x] git_sha in job.json (commit 41aa61a)
[x] env_overrides in job.json (commit 41aa61a)
[x] repro_cmd column + job.json (commit 41aa61a)
[x] config_snapshot.yaml (commit 41aa61a)
[x] FS↔DB bijection field gaps (commit c546706)
[x] `import` / `export` / `migrate-runs` CLIs (commits 0b48be4, 568a798, 52c45e6)
[x] Heartbeat-based stale-queued recovery (commit df3f7d3)
[x] Layered run-dir layout for multi-host rsync (commit 52c45e6)
```

