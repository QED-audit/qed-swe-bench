# benchmarks/

Canonical benchmark configs we ship and run, plus the dataset / build
tooling for each task family as a git submodule.

| Config                 | Interface used         | Dataset / build tooling                           |
| ---------------------- | ---------------------- | ------------------------------------------------- |
| [`v8.yaml`](./v8.yaml) | `rl.mcp.v8_task.v1` | [`bench-v8/`](./bench-v8/README.md) (submodule)   |

To run a benchmark end-to-end you need both: (a) the YAML config in this
directory, and (b) the matching dataset / images, which the submodule's
own README walks through (build mcp-server → bootstrap dataset → build
docker images).

For dev / smoke / one-off scenarios there's no separate scratchpad
directory anymore — single-cell smokes are CLI overrides on `v8.yaml`
(see "Common single-bug invocations" below). The cheap-tier matrix
smoke lives alongside `v8.yaml` as `smoke-matrix-cheap.yaml`.

The split is intentional: configs in this directory are version-controlled,
production-shaped, and represent benchmarks we expect to publish numbers
for.

## Adding a new benchmark

1. Pick the right interface from `qed_swe_bench/contract/interfaces.py`. If
   none fits, add a new interface to the registry first.
2. List the envs (one per env_id) with their docker image ref + interface name.
3. List the models you'll evaluate.
4. Set the budgets for one episode (turns / tokens / context / max_tokens).
5. Set `max_parallel` to whatever your hardware supports for the env class.
6. Validate with `qed_swe_bench benchmark --config benchmarks/<your>.yaml --dry-run`.

## Common single-bug invocations against `v8.yaml`

`v8.yaml` is the publishable matrix (7 keepers × 14 V8 bugs × 5 seeds = 490 episodes).
Use the CLI to scope a run down to a single (model, env, seed) cell
without forking the YAML. There used to be a family of
`v8-single-*.yaml` configs for this; they were retired in favor of
filter + override flags:

| Filter / override flag | Behavior                                                                               |
| ---------------------- | -------------------------------------------------------------------------------------- |
| `--models`             | Filter `models:` by id (typo guard)                                                    |
| `--envs`               | Filter `envs:` by id (typo guard)                                                      |
| `--seeds`              | Filter `seeds:` by value                                                               |
| `--set <dotted>=<val>` | Override any YAML field. Value is YAML-parsed (int / bool / list). Dotted paths nest.  |
| `--turn-budget`        | Sugar for `--set budgets.turn_budget=<n>`                                              |
| `--cost-cap-usd`       | Sugar for `--set cost_cap_usd=<n>`                                                     |
| `--episode-timeout`    | Sugar for `--set episode_timeout_s=<n>`                                                |
| `--nudges`             | Override `nudges:` (true / false / list)                                               |

Recipes (Opus 4.7, 1 seed, 2h timeout, $25 cap):

```bash
# Clean eval against one bug
qed_swe_bench benchmark --config benchmarks/v8.yaml \
  --models anthropic/claude-opus-4-7 \
  --envs v8-e01 --seeds 1 \
  --set cost_cap_usd=25 --set episode_timeout_s=7200

# With nudges (matches the imported-opus historical baseline configuration)
qed_swe_bench benchmark --config benchmarks/v8.yaml \
  --models anthropic/claude-opus-4-7 \
  --envs v8-e01 --seeds 1 \
  --set cost_cap_usd=25 --set episode_timeout_s=7200 \
  --nudges true

# v2 init-prompt hint (see benchmarks/prompts/init-v2-hint.template)
qed_swe_bench benchmark --config benchmarks/v8.yaml \
  --models anthropic/claude-opus-4-7 \
  --envs v8-e01 --seeds 1 \
  --set cost_cap_usd=25 --set episode_timeout_s=7200 \
  --set init_prompt_hint_path=benchmarks/prompts/init-v2-hint.template

# Cheap pipeline check on a different bug, Haiku, 100 turns
qed_swe_bench benchmark --config benchmarks/v8.yaml \
  --models anthropic/claude-haiku-4-5 \
  --envs v8-e25 --seeds 1 \
  --turn-budget 100 --cost-cap-usd 1.5
```

For a multi-model parallel shakedown across all 7 keepers (validates
plumbing before committing to a 300-turn sweep), use a low
`--turn-budget` against the full config:

```
qed_swe_bench benchmark --config benchmarks/v8.yaml \
  --envs v8-e01 --seeds 1 \
  --turn-budget 30 --max-parallel 7 --cost-cap-usd 15
```

GPT-5.5 is no longer cyber_policy-blocked for our org as of
2026-05-02 (Trusted Access for Cyber enrollment confirmed) — see
v8.yaml header comment.
