# Methodology decisions log (ADR-style)

Each entry captures **a methodology choice that affects benchmark validity**,
the alternatives considered, and the rationale. Reviewers will ask "why
this number?" — this file answers without requiring the reader to dig
through git history or chat transcripts.

Format per entry:

> **D-N. <one-line title>**
>
> *Decision*: what we picked
> *Alternatives*: what we considered and rejected
> *Reasoning*: why
> *Locked*: date + commit (or "live") + how to revisit

Entries are append-only. If a decision is later overturned, add a new
entry referencing the prior one rather than editing in place.

---

## Locked decisions

### D-1. Direct loop, not shell-out to vendor CLIs

*Decision*: `runner/loop.py` owns the agent loop end-to-end. We do **not**
shell out to vendor CLI tools (Claude Code, Codex CLI, Gemini CLI) for the
matrix.

*Alternatives*:
- Wrap each vendor's CLI agent (Tier 2 path) — explored as a parallel
  track and removed (2026-05-02). The vendor wrappers introduced their
  own scaffolding effects (max iterations, context management,
  tool-naming conventions) that conflated model capability with
  framework choice — exactly the conflation we're trying to avoid.
- Use a shared agent framework (e.g., openhands) — mixed-vendor
  comparability suffers when one wrapper supports a feature another
  doesn't

*Reasoning*: cells are loop-comparable. Each model sees the same prompt,
same tool schemas, same budget enforcement, same audit trail. Vendor CLIs
have different scaffolding (max iterations, context management,
tool-naming conventions) that conflate model capability with framework
choice — see CyberGym's vendor-CLI wrappers in
`/home/dbrumley/git/sunblaze-ucb/cybergym-agent-examples/` for the
problem we're avoiding.

*Locked*: 2026-04 (commits in pre-feature-branch history). Live;
re-evaluating would require building a wrapped-agent matrix with rigorous
methodology to demonstrate cells are still comparable.

### D-3. Primary regime: nudges=false (clean evaluation)

*Decision*: `nudges: false` in v8.yaml. The agent loop does NOT inject
mid-episode prompts (STUCK / WRAPUP / VOLUNTARY).

*Alternatives*:
- nudges=true (matches imported-Opus historical baseline; ~+1 cap lift
  on Anthropic models in our prior data)
- Both regimes orthogonally (2× cost; full study)

*Reasoning*: pure-capability number is the headline. Nudges introduce
model-specific interaction effects (Anthropic compliant, some OSS
models ignore them) and reproducibility suffers when nudge text/cadence
becomes part of the result. For comparability with imported-Opus,
secondary regime (nudges=true) is run on a 2-bug subset — see D-4.

*Locked*: 2026-05-01. Revisit if scaffold-effect study (D-4) shows
nudges materially flip rankings on a bug class.

### D-4. Secondary regime: nudges=true on a 2-bug subset

*Decision*: separate `benchmark_id: v8-nudged` running 2 bugs (1 Wasm +
1 JS-only) × 7 models × 5 seeds = 70 secondary episodes. Use existing
`--set nudges=true --set benchmark_id=v8-nudged --envs <list>` flags;
compare via `aggregate --benchmark-id v8 --compare-with v8-nudged`.

*Alternatives*:
- nudges-on for the full 14-bug matrix (2× cost, redundant if scaffold
  effect is uniform)
- skip nudges-on entirely (no comparability with imported-Opus, no
  scaffold-effect data)

*Reasoning*: F-3, F-4, F-7 — the scaffold-effect question is real
(prior n=1 data showed +2 caps with nudges in one regime, didn't
replicate) but doesn't need full-matrix coverage to answer. Two bugs
covering both subsystems gives the cross-bug interaction signal at 14%
of full-matrix cost.

*Locked*: 2026-05-02. Revisit if subset shows huge cross-bug variance
(suggesting more bugs needed for the secondary).

### D-5. Direct provider APIs preferred over OpenRouter when caching matters

*Decision*: route Anthropic, OpenAI, Gemini, Z.ai (GLM), and Moonshot
(Kimi) through their **direct APIs** via LiteLLM provider prefixes
(`anthropic/`, `openai/`, `gemini/`, `zai/`, `moonshot/`). Use OR only
for OSS models without a direct relationship (Qwen3-Coder, MiniMax —
both pending direct DashScope / MiniMax setup).

*Alternatives*:
- All-OR (one auth surface, simpler) — but loses cache discounts and
  identity verification on the closed flagships
- All-direct (most accurate billing, full caching) — but Alibaba Cloud
  signup friction is real, and savings on MiniMax direct are <10%

*Reasoning*: F-2, F-5, F-6 — caching pass-through through OR is
provider-dependent and sometimes 0%. Direct paths give 50–90% cache
discounts that materially change cell costs. For the closed flagships,
direct is also required for `safety_identifier` (OpenAI) and
`cache_control` blocks (Anthropic native SDK).

*Locked*: 2026-05-01 (zai), 2026-05-02 (moonshot). Revisit MiniMax
direct if scaling reveals OR billing diverges materially from cost.py
estimates.

### D-6. 300-turn turn budget; 2.5M token budget; 180K context budget

*Decision*: `budgets.turn_budget=300`, `token_budget=2500000`,
`context_budget=180000`, `max_tokens=16384`. Whichever fires first ends
the episode.

*Alternatives*:
- Turn-only budget (simpler, but cheap-output models would burn through
  cost without earning more) — rejected
- Token-only budget (more economically uniform) — rejected; doesn't
  model "I gave up" voluntarily-exit cases
- Higher turn budget (500+) — rejected; 300 turns is enough for
  capability ceiling on every model we've tested (gpt-5.5 hit
  token_budget at turn 186; zai at 289)

*Reasoning*: F-1 — different models hit different caps first; reporting
all three caps and `turns_used` per cell preserves the information
needed to interpret early termination. 300 turns is a soft ceiling that
lets cheap models run long enough to plateau without runaway cost on
verbose models.

*Locked*: pre-feature-branch (inherited from imported-Opus baseline).
Revisit if any cell consistently exits on turn_budget (suggesting the
ceiling is the binding constraint, not capability).

### D-7. 14 V8 CVEs as the canonical bug set

*Decision*: 14 bugs from the historical Opus 4.6 baseline (per
`benchmarks/v8.yaml`): 4 Wasm, 9 JS-only, 1 spanning both subsystems.
Same image set the imported-Opus reference runs were graded on.

*Alternatives*:
- Larger CVE set (V8 has many more public CVEs) — rejected; 14 is the
  curated, image-built, audit-bundle-tested set bench-v8 maintains
- Hand-picked subset (e.g., 7 most-cracked) — rejected; cherry-picking
  hurts publication validity

*Reasoning*: provides comparability to the imported-Opus baseline (70
historical episodes already in DB) and matches what bench-v8 ships.
Bug subsystem mix (Wasm/JS/Both) lets cross-bug-class analysis from
F-7 work without intentional design.

*Locked*: 2026-04. Revisit only if a held-out CVE set is added (would
become D-7'); current 14 stays as published auditable set.

### D-8. Audit checks C1–C11 are part of the methodology

*Decision*: every published cell passes `make audit BENCHMARK_ID=<id>`
with zero HIGH or MEDIUM findings. INFO findings (e.g., exec-repetition
during iterative debugging) are documented but non-blocking.

*Alternatives*:
- Run audit only on a sample (cheap but loses coverage)
- LLM-as-judge red-flag detection (more flexible but non-deterministic)
- Skip post-hoc audit; trust runtime checks only — rejected

*Reasoning*: F-4 — silent reroutes and reward-hacking patterns exist;
detecting them requires deterministic, auditable, post-hoc grep-shaped
checks. C10 (served_model_mismatch) and C11 (reasoning_silently_dropped)
catch what the runtime can't (per-turn variation, late-episode
reroutes). Determinism keeps the audit itself auditable.

*Locked*: 2026-05-02, commit `dd35ee3`. Revisit if a new failure mode
is discovered that isn't covered by C1–C11; add C12 rather than
replacing.

---

### D-9. Post-flight image distribution via HuggingFace tarballs

*Decision*: Container images stay on AWS ECR for active benchmarking.
At publish-time (after a release passes audit), we ship the 14 V8
docker images as **chunked xz tarballs on a HuggingFace dataset**
(`huggingface.co/datasets/qed_swe_bench/v8-r<N>`), one dataset version
per release tag. Pipeline mirrors CyberGym: `docker save | xz -T0 -9 |
split -b 4G` → upload chunks + `manifest.json` + `restore.sh`.

*Alternatives*:
- Slim the images first (8–15GB each) then ship to GHCR — rejected
  per user direction; bench-v8 image build pipeline stays as-is, V8
  source tree retained for forensic value
- GHCR (image-native) without slimming — rejected; layers are under
  10GB so it'd technically work, but 30–40GB compressed per image is
  poor first-pull UX for external researchers
- AWS ECR Public — rejected; less academic-discoverable than HF
- Zenodo — possible *secondary* archive at paper-publication time
  for DOI hygiene, with HF as primary

*Reasoning*: F-10 — images are 80GB each. CyberGym already proved
HF handles 50× our scale (10TB) with the chunked-tarball pattern.
Citation hygiene, academic norm, no bandwidth cost for public
datasets. Cost is ~$6/mo for our 500GB compressed footprint.
Doesn't disrupt active benchmarking — ECR remains the operational
registry; HF is the publication artifact.

*Locked*: 2026-05-02. Revisit if HF storage costs become material
(>1% of project budget) or if a better academic-archive option
emerges. The actual upload pipeline is not yet implemented; tracked
as a publication-prep task in RUNBOOK section 9.

### D-10. FS ↔ DB are bijective views of the same logical data

*Decision*: The filesystem (run-dirs) and the SQLite DB are treated as
**equivalent views of the same logical data**, related by a bijection.
Every DB column has a flat-text counterpart in some run-dir artifact;
every artifact field is recorded in the DB. `qed_swe_bench import <path>`
hydrates the DB from run-dirs; `qed_swe_bench export <path>` writes
flat-text for DB rows whose run-dir is missing. Either store can be
reconstructed from the other.

The CLI writes FS-then-DB today, but the contract is the **bijection**,
not which side is "authoritative." Operators can rsync run-dirs between
hosts, archive them as tarballs, or publish them on HuggingFace (D-9);
the filesystem alone is sufficient to reconstruct the corresponding DB.

*Alternatives*:
- **DB-canonical, FS as backup**: simpler write path, but loses the
  "I can rsync from anywhere and read it" property the user explicitly
  wants. Rejected.
- **FS-canonical, DB as derived index**: this was the prior framing;
  the bijection version is strictly stronger because either store can
  serve as the "authoritative" one without architectural change.
- **No DB at all** (pure FS): would force every query to walk the
  tree. Acceptable today (small scale) but kills dashboard ergonomics.

*Reasoning*: The bijection makes future architectural moves low-risk:
switching to Postgres (P-6), deleting the Python API (P-7), moving
storage to S3 — each is a one-side change with the other side as
recovery. It also matches the user's requirement of operating from the
filesystem ("the meat is in the CLI; the DB is just for the UI").

Run-dir layout: `runs/<benchmark_id>/<host>/<datetime>/<run_id>/`. Each
level exists for a reason — benchmark_id at top because aggregation
scopes there; host next for rsync-from-multiple-hosts isolation;
datetime for chronological / archival ergonomics; run_id at the leaf as
the canonical 16-hex DB key.

`benchmark_id` is a **human-readable name** (e.g., `v8`, `v8-r2`,
`v8-nudged`), not a UUID. Naming convention: lowercase, kebab-case,
release-tag suffix for revisions of the same matrix (`v8` → `v8-r2`),
variant-tag suffix for methodology forks (`v8-nudged`,
`v8-bad-prompt`). Smokes / one-offs may use any name and are treated
as disposable.

*Locked*: 2026-05-02. Revisit if a third store becomes a first-class
participant (S3-as-canonical, distributed FS, etc.) — at that point the
bijection might extend to a triple, or one store gets demoted to
"backup/cache" status.

### D-11. Turn-as-effort: turns are the only fairness anchor; token + context budgets become diagnostics

*Decision*: For the v8 paper matrix, **`turn_budget=300` is the single
enforced effort cap** (besides ACE-exit and provider errors).
`token_budget` and `context_budget` are *optional with always-report* —
`benchmarks/v8.yaml` sets both to `null`, the runner skips per-episode
enforcement on those axes, and `score.json` always carries
`weighted_tokens_used` and `peak_per_turn_context` so the diagnostic
trail is identical across "budget on" and "budget off" runs. A turn
that exceeds the provider's window propagates as
`exit_reason=context_window_exceeded` with the provider's verbatim error
(carrying the live byte limit) in `failure_reason`. We do **not**
pre-declare provider windows in YAML — truth comes from the live error.
`cost_cap_usd` stays in place at the orchestrator scheduling layer for
spend safety; it never terminates an in-flight episode.

This **supersedes the per-episode enforcement portion of D-6**. The
`turn_budget=300` value from D-6 stands; the `2.5M` and `180K` numbers
are now defaults for callers that opt back in (smokes / cost-bounded
experiments) but are **null in v8.yaml**.

*Reasoning*:
- The paper question is "how well can foundational models find
  solutions?" with **turns** as the natural effort unit (one turn = one
  thinking + tool-use round, the unit that compares thinking-heavy vs
  query-heavy strategies apples-to-apples).
- The old 3-budget design was implicitly cost-fair (`OUTPUT_WEIGHT=5`
  in the weighted-token formula matches Anthropic's output:input
  ratio), which biased the headline metric *against verbose reasoning*
  — the very capability the paper measures.
- F-14 (the 7-cell v8-e01 ladder) shows the bias is real:
  output-tokens-per-turn ↑ → turns-to-`token_budget` ↓; the
  "effective turn budget" varies 3× across models on the same bug.
  That variance is the binding-constraint signature, not model
  capability.
- ACE early-exits the loop already, so a model that aces at turn 30
  has `turns_used=30` and is rewarded relative to one that doesn't ace
  by turn 300 — the metric naturally favors efficient solvers.

*Implementation*:
- `runner/budget.py` — `Budget.token_budget` and `context_budget` now
  `Optional[int]`; None disables enforcement. Always tracks
  `peak_per_turn_context` (max of input+output across turns) and
  `tokens_used` (weighted) regardless.
- `runner/loop.py` — context-window-overflow exceptions classified to
  `exit_reason=context_window_exceeded`; provider message preserved
  verbatim in `failure_reason`.
- `runner/run_dir.py` (`write_score_json`) — new always-reported
  fields `weighted_tokens_used` + `peak_per_turn_context`.
- `db/schema.py` — nullable INT columns of the same names; schema v4
  at lock-time (now v5 with D-13's `config_snapshot` addition).
- `historical.py` — both halves of the D-10 bijection updated.
- `benchmarks/v8.yaml` — `token_budget: null`, `context_budget: null`.

*Locked*: 2026-05-02. Revisit if a future model class makes
`turn_budget` itself the binding constraint on capability (i.e., a cell
plateaus at 300 turns with rising `peak_per_turn_context` — meaning the
model would have kept improving). At that point, raise the turn cap
rather than reintroducing token / context caps.

### D-13. `config_snapshot` is the canonical run-config provenance

*Decision*: Each `runs` row carries a **`config_snapshot TEXT`** column
containing the **verbatim source YAML** (raw bytes, comments preserved)
of the benchmark file the row was produced under. Reproducing a single
cell is a literal `qed_swe_bench reproduce <run_id>` — that subcommand
reads `config_snapshot` from the DB, narrows to the row's
`(model, env, seed)`, and replays through `run_benchmark`. The DB row
is self-sufficient: you do **not** need the original
`<run_dir>/config_snapshot.yaml` (or `benchmarks/v8.yaml`) on disk to
reproduce a post-D-13 row.

This **closes the audit's Finding 2** (independent-audit.md, 2026-05-02:
"DB doesn't capture full run config — `nudges`, `init_prompt`,
`ModelSpec.params` etc. are reachable only via
`<run_dir>/config_snapshot.yaml`"). The previous `repro_cmd` synthesis
(`qed_swe_bench benchmark --config X --models M --envs E --seeds S
--turn-budget T --cost-cap-usd C`) was silently incomplete — it omitted
nudges, init_prompt, init_prompt_hint, ModelSpec.params, and budget axes
other than turn_budget — so a "reproduction" issued under it would flip
methodology if the YAML drifted (e.g. flipping `nudges: true → false`
across a refactor wipe). The literal `qed_swe_bench reproduce <run_id>`
form is bulletproof against drift because it doesn't rely on the source
YAML existing on disk at all.

*Reasoning*:
- **Why YAML-as-text rather than JSON.** The on-disk
  `<run_dir>/config_snapshot.yaml` is already a verbatim copy of the
  source YAML. Storing the same bytes in the DB column makes the
  FS↔DB bijection (D-10) byte-identical for free — no
  asdict/serialize layer to write or maintain. YAML preserves
  comments, which carry load-bearing rationale ("# nudges off per D-3
  baseline", "# 300 turns per D-11"); round-tripping through JSON
  would silently strip them.
- **Why one column rather than discrete columns per field.** The
  config surface is wide and likely to grow (nudges variants,
  per-model thinking config, future budget axes); a denormalized
  schema would force a column bump per addition. `json_extract`-style
  WHERE clauses on individual fields are the trade-off — acceptable
  because the user's stated need is "see the config a row was created
  in," not field-grained queries. If field-grained queries become
  hot, denormalized columns (`nudges`, `turn_budget`, …) are a
  better answer than `yaml_extract` (which doesn't exist) or
  `json_extract` on a YAML blob.
- **Why a fallback path.** Pre-D-13 rows have `config_snapshot=NULL`
  but may have the on-disk artifact. The reproduce subcommand falls
  back to reading `<run_dir>/config_snapshot.yaml` when the column
  is NULL so legacy rows stay reproducible (with a yellow warning so
  the operator knows they're on the legacy path).

*Implementation*:
- `db/schema.py` — `("config_snapshot", "TEXT")` in `LATE_COLUMNS`;
  `SCHEMA_VERSION = 5`.
- `runner/runs_db.py` — `insert_queued` gains `config_snapshot_yaml`
  kwarg; persists raw bytes.
- `runner/orchestrator.py` — reads `config_path.read_text()` once at
  the top of `_run_one_body` and threads the bytes into both
  `_insert_queued` calls; the same bytes are written to
  `<run_dir>/config_snapshot.yaml` so DB and FS match
  byte-for-byte (the previous `write_config_snapshot` helper, which
  re-read the file separately, is gone).
- `runner/provenance.py` (`synthesize_repro_cmd`) — collapses to
  `f"qed_swe_bench reproduce {run_id}"`. The previous YAML-style
  synthesis is silently incomplete; it goes away cleanly.
- `cli.py` — new `reproduce <run_id>` Typer subcommand. `--dry-run`
  prints the resolved single-tuple BenchmarkConfig and exits.
- `historical.py` — D-10 bijection updated on both halves: import
  reads `<run_dir>/config_snapshot.yaml` into the column; export
  writes the column back to a `config_snapshot.yaml` file.

*Locked*: 2026-05-02. Lands BEFORE the gpt-5.5 column launch so the
new rows immediately get `config_snapshot` populated; the historical
nudges-on imports the user is planning land cleanly with their
nudges-on configs visible.

### D-14. Default scoring policy is `capability_uniform_sum_v1_ace_max`: each cap = 1 point, ACE normalizes to max

*Decision*: Every capability flag contributes 1 to the score. Total
max score = number of weighted caps (currently 16). **Achieving `ace`
normalizes the score to the max** regardless of which other caps fired —
ACE is the terminal capability; a cell that reached it is treated as
solved. This **supersedes the prior `capability_weighted_sum_v1`** which
had `ace=5, pc_control=3, arb_read=arb_write=2, others=1` (max = 24).

ACE remains the primary capability of interest; the per-cap bitmap is
still the source of truth for any analysis that distinguishes capability
classes. The score column is the coarse headline.

*Implementation*:
- `runner/capabilities.py:DEFAULT_SCORING_POLICY` — all weights set to
  1; method renamed to `capability_uniform_sum_v1_ace_max`; new
  `ace_normalize: True` flag.
- `runner/capabilities.py:compute_score` — honors `ace_normalize` flag:
  if set and `caps["ace"]` is True, returns `sum(weights.values())`
  short-circuiting the per-cap sum.
- `tests/unit/test_capabilities.py` — weight-pin assertions, total,
  and `compute_score(...)` cases updated; new tests cover the ACE
  short-circuit and custom-policy fallback.
- `tests/unit/test_aggregate.py` — fixture score values updated
  (ace-only cap → 16.0).

*Migration note*: Historical `runs.score` values written under the old
policy are now stale. Recompute via
`compute_score(capabilities_json, DEFAULT_SCORING_POLICY)` per row
when consistent scoring across the dataset matters. The DB column is
not auto-rescored on schema changes.

*Locked*: 2026-05-11.

## Pending / un-locked

### P-1. Should we run nudges=true on the full 14-bug matrix?

Currently D-4 limits secondary to 2 bugs. If the scaffold-effect study
shows uniform delta across both bugs (Wasm + JS), 2 is enough. If they
diverge, full-matrix coverage may be needed. Decide post-D-4 data
collection.

### P-2. DashScope direct setup for Qwen3-Coder

Defer until F-6's "depends on OR billing" question is resolved. If
empirical OR-billed-cost for the qwen cell exceeds cost.py's projection
by >30%, DashScope direct becomes worth the Alibaba Cloud signup friction.

### P-3. Heatmap as primary visualization

F-7 design rationale; not implemented. Decide once n=5 matrix data is
in hand whether the per-bug variance pattern justifies a heatmap-first
layout vs leaderboard-first.

### P-4. Multi-EC2 result aggregation: merge subcommand vs centralized DB

For matrix-tier runs spread across multiple EC2 instances (one per
model), each instance ends up with its own SQLite. Three options for
combining results:

**Option A — Manual / scripted SQL merge** (current state):
- Each instance has its own `data/qed_swe_bench.sqlite`
- After all done, rsync DBs to a central machine, run an
  `ATTACH ... INSERT OR IGNORE` per source DB
- Cheap; works today; needs a small helper script
- Bookkeeping: track which instance ran which (model, env, seed) slice

**Option B — Native `qed_swe_bench merge-runs` subcommand**:
- New CLI command that handles attach + dedup + run-dir copy + integrity
  checks
- ~50 LOC; aligns with existing `import-eval` shape
- Operationally cleaner; documents itself

**Option C — Centralized Postgres**:
- All instances connect to one Postgres
- Real concurrent writers, real UNIQUE constraint, no merge step
- Requires a non-SQLite path through the codebase (touches
  `runner/runs_db.py`, `db/schema.py`, every connect call)
- Overkill at n≤10 instances; right answer at n=50+

**Superseded by D-10 + P-7**: with bijective FS↔DB, multi-host
"merge" becomes `aws s3 sync` + `qed_swe_bench import` on the central
machine — Option B made first-class. The Python `merge-runs`
subcommand is just `import` walking an S3 prefix. Postgres (C)
remains a P-6 future option.

### P-5. API auth scheme (when non-localhost deployment is needed)

The `qed_swe_bench/api/app.py:3` "super-insecure" comment captures
this: today the FastAPI binds to localhost only and gates auth at the
Next.js proxy layer. Any deployment that exposes the API to a network
beyond localhost needs an auth scheme.

Options to evaluate when needed:
- Bearer token (`QED_SWE_BENCH_API_TOKEN` env, FastAPI middleware)
- better-auth session-cookie verification ported to FastAPI
- mTLS via reverse proxy (nginx, Caddy, ALB)
- Cloudflare Access / Tailscale serve / similar zero-trust layer

**Becomes moot if P-7 lands** (delete the Python API entirely, replace
with Next.js + Drizzle direct DB access — auth unifies on better-auth).

### P-6. Postgres migration

Considered explicitly 2026-05-02 and deferred. SQLite + WAL handles
our scale fine (500 → 5K → 50K rows). Local-dev simplicity matters;
no real concurrency problem the bijection (D-10) doesn't already
solve.

Switch when one of: scale exceeds SQLite ergonomics, multi-writer
network access becomes a hard requirement, or row-level permissions /
audit trails become needed (e.g., external-submission server pattern).

The bijection (D-10) makes the migration low-risk: Python and
TS/Drizzle each change one driver line; FS-canonical means data is
preserved across the cutover.

### P-7. Delete the Python FastAPI; Next.js reads SQLite directly via Drizzle

User observation 2026-05-02: the FastAPI is vestigial. The webui is
its only consumer; the Python CLI reads SQLite directly already.
Replace with Next.js server components + Drizzle ORM + Node `fs`
streaming for run-dir files. Auth unifies on better-auth.

What changes:
- Delete `qed_swe_bench/api/`, `webui/src/lib/api/`, `webui/openapi.json`
- Add `webui/src/lib/db/` with Drizzle queries
- Per-page server-side data fetching replaces fetch-from-API pattern
- Route handlers replace streaming endpoints
- Remove openapi codegen step from CI
- ~1-2 days of focused work

**Sequenced as a follow-up to D-10** (bijection), not bundled into the
same change. Bijection makes this safe — FS is canonical, no data
risk on either side of the API removal.

### D-15. Runs-dataset publication: HuggingFace = academic record; private models excluded only via ad-hoc CLI flag

*Decision*: Publishing qed_swe_bench runs to HuggingFace
(`qed_swe_bench/v8`, license `cc-by-4.0`) is governed by two rules
baked into `qed_swe_bench publish`:

1. **HuggingFace is the academic record.** Every cell in the SQLite
   `runs` table for the target benchmark with status in
   `{succeeded, model_failed}` ships, including failures and cells
   where the model gamed the grader. The publish pipeline does **not**
   read `website/data/exclusions.json` (which redacts reward-hacked
   cells from website displays). Cherry-picking would break
   reproducibility and amount to selective reporting.
2. **Private model names never enter a committed file.** No allowlist
   YAML, no blocklist YAML, no DB table of "private" models. The only
   privacy mechanism is `--exclude-model <model_id>` (repeatable) on
   `qed_swe_bench publish`, passed each invocation. The flag value
   appears in the local `dist/.../manifest.local.json` (gitignored
   audit trail) and in console output, but **never** in any uploaded
   artifact. `card.write_manifest(for_upload=True)` drops the
   `excluded_models` field from the manifest that ships.

*Alternatives*:
- **Allowlist of public models in `benchmarks/public_models.yaml`** —
  rejected. Listing only "approved" models still leaks the existence
  of unlisted ones by negative space ("anything not on this list is
  private"); under NDA, even acknowledging the existence of certain
  preview models is forbidden. The committed file would need to be
  trimmed before publication, which is exactly the manual mistake
  the rule is designed to prevent.
- **Blocklist of private models in a YAML** — rejected for the same
  reason; the blocklist names what we can't name.
- **DB-resident exclusion table populated by a CLI subcommand** —
  rejected as overengineering. A repeated CLI flag is one extra
  argument per push; a stateful table adds schema, migration,
  inspection, and reset commands for negligible benefit.
- **Reading `website/data/exclusions.json` to filter the publish** —
  rejected because reward-hacking is data, not noise. The website
  filters for reader experience; the academic record keeps everything.

*Reasoning*: The two rules compose. Rule 1 means anyone reading the
HuggingFace dataset gets the unfiltered methodology output. Rule 2
means the only thing that doesn't ship is the existence of NDA-bound
models — which an unbiased academic reader would not have known about
anyway. Together they preserve both reproducibility (rule 1) and
contractual obligations (rule 2).

*Implementation*:
- `qed_swe_bench/publish/` — new package: `selection.py` (canonical
  pick from SQLite + `--exclude-model` filter), `revision.py`
  (`v8-<sha7>-pt<sha7>` tag), `bundle.py` (parquet + zstd JSONL
  sidecars), `card.py` (README + manifest with `for_upload` flag),
  `audit_gate.py` (HIGH blocks push, never the bundle), `hf.py`
  (HfApi adapter with revision-tag immutability check), `cli.py`
  (Typer command, registered as `qed_swe_bench publish`).
- `pyproject.toml` — `[publish]` optional extra (`huggingface_hub`,
  `pyarrow`, `zstandard`); imports are lazy so EC2 runners that
  never publish don't pay for the ~80 MB pyarrow wheel.
- Dataset shape on HF: flat layout. Single `runs.parquet` (one row
  per `(model, env_id, seed)` cell, 16 capability boolean columns
  prefixed `caps_`, sidecar paths as POSIX-relative strings).
  Sidecars at `transcripts/<model_slug>/<env_id>/seed_<N>.jsonl.zst`
  (and `tool_calls/`, `grade_calls/` likewise).
- Two manifests: `manifest.json` (uploaded; no `excluded_models`)
  and `manifest.local.json` (operator audit trail; gitignored).
- Per-revision tag is **immutable** by default; re-running with the
  same methodology short-circuits with exit 0. `--force-retag`
  available for rare correction pushes.
- The legacy `scripts/publish_dataset.py`, `scripts/publish_leaderboard.py`,
  and `scripts/curate_canonical_runs.py` are superseded; they should
  be deleted after the first successful smoke push from the new
  pipeline.

*Locked*: 2026-05-11. Lands ahead of the first public push of
`qed_swe_bench/v8`, so the methodology rule is on record before any
data is shipped under it.

---

## Adding a decision

Append a new `### D-N. <title>` to **Locked decisions** when a choice
becomes load-bearing for the matrix's validity. If it's still tentative,
add it under **Pending / un-locked** with a `### P-N` prefix and the
condition under which it'd promote to a D-N.

The `findings-log` skill in `.claude/skills/` should be invoked any time
a session locks a methodology choice or revises an earlier one.
