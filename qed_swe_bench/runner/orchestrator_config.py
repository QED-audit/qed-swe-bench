"""Benchmark configuration types + YAML/JSON parser.

Carved out of `orchestrator.py` so the file split keeps each piece
focused. Pure data classes + a parser; no I/O, no docker, no async.

Imported by:
- `runner.orchestrator` — uses `BenchmarkConfig` to drive a sweep.
- `cli.py` — calls `parse_config` on YAML-loaded benchmark files.
- tests — import the dataclasses to seed fixtures.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import yaml as _yaml

# Per-tuple wallclock cap (seconds). Defaults generous enough for V8
# episodes at the standard 300-turn budget; long-horizon configs can
# raise it via `episode_timeout_s` in the benchmark YAML. See
# `runner/budget.py` for the orthogonal turn/token/context caps that
# bound work *within* a tuple.
DEFAULT_EPISODE_TIMEOUT_S = 1800  # 30 min


class NudgeKind(str, Enum):
    """Mid-episode prompt-engineering interventions the loop can inject."""

    # 50 turns since the last grade(): asks the model to grade its current
    # PoC instead of spinning on a single approach.
    STUCK = "stuck"
    # At 75% of turn_budget, once: tells the model how many turns remain
    # so it can focus on the highest still-reachable capability.
    WRAPUP = "wrapup"
    # Model emitted no tool calls before exhausting budget: re-prompts to
    # keep working toward higher capabilities.
    VOLUNTARY = "voluntary"


ALL_NUDGES: frozenset[NudgeKind] = frozenset(NudgeKind)


def parse_nudges(value: Any) -> frozenset[NudgeKind]:
    """Coerce the YAML `nudges` value to a `frozenset[NudgeKind]`.

    Accepts:
      true / false / null / missing  → all-on / all-off
      list of kind names             → that subset; element "all" expands
                                       to every kind
    """
    if value is True:
        return ALL_NUDGES
    if value is False or value is None:
        return frozenset()
    if isinstance(value, list):
        out: set[NudgeKind] = set()
        for item in value:
            if item == "all":
                return ALL_NUDGES
            try:
                out.add(NudgeKind(item))
            except ValueError as exc:
                valid = [k.value for k in NudgeKind] + ["all"]
                raise ValueError(
                    f"unknown nudge kind {item!r}; expected one of {valid}"
                ) from exc
        return frozenset(out)
    raise ValueError(
        f"nudges: expected bool or list[str], got {type(value).__name__}"
    )


@dataclass(frozen=True)
class EnvSpec:
    id: str
    image: str
    task_type: str = "binary_task"
    # Interface name from `qed_swe_bench.contract.interfaces`. Recorded
    # on every `runs` row for audit. Default matches bench-v8 V8 envs;
    # other task families override.
    interface: str = "rl.mcp.v8_task.v1"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    # Provider-specific completion params merged into the LLM call. Used for
    # things like OpenAI's `reasoning_effort` for the gpt-5 family. Empty
    # frozen dict default — keep deterministic and avoid mutable-default
    # pitfalls. Pass `None` for unsupported keys to *remove* a default that
    # the client would otherwise add (e.g. `temperature` on gpt-5*).
    params: Mapping[str, Any] = field(default_factory=dict)
    reasoning_replay: str = "auto"


@dataclass(frozen=True)
class Budgets:
    """Per-episode hard limits. `turn_budget` is the only fairness anchor
    for v8 (turn-as-effort, decisions.md); `token_budget` and
    `context_budget` are *optional with always-report* — set to None in
    YAML to disable enforcement and `Budget` will still track
    `weighted_tokens_used` / `peak_per_turn_context` for `score.json`.
    """

    turn_budget: int = 300
    token_budget: int | None = 2_500_000
    context_budget: int | None = 180_000
    max_tokens: int = 16_384


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_id: str
    models: list[ModelSpec]
    envs: list[EnvSpec]
    seeds: list[int]
    budgets: Budgets = field(default_factory=Budgets)
    max_parallel: int = 2
    # The container's MCP `setup()` carries all bug-specific framing; this
    # is the first user turn that sends the agent into the loop. Optional —
    # falls back to `runner.prompts.default_init_prompt()`.
    init_prompt: str | None = None
    init_prompt_path: Path | None = None
    # Optional supplement appended to the resolved init prompt with a blank
    # line between. Cheap experimental knob: framing tweaks, capability
    # hints, etc. Inline string OR file path.
    init_prompt_hint: str | None = None
    init_prompt_hint_path: Path | None = None
    # Resilience knobs. Generous defaults preserve current behavior.
    episode_timeout_s: int = DEFAULT_EPISODE_TIMEOUT_S
    cost_cap_usd: float | None = None

    # Mid-episode prompt-engineering interventions. See `NudgeKind` above
    # for the available kinds. YAML accepts `true` / `false` (all-on /
    # all-off) or a list (e.g. `[stuck, voluntary]`) for fine-grained
    # control. Defaults to empty — clean evaluation. Always set this
    # explicitly in benchmark YAMLs so the mode is obvious at a glance.
    nudges: frozenset[NudgeKind] = field(default_factory=frozenset)

    def with_overrides(self, **changes: Any) -> BenchmarkConfig:
        """Return a copy with selected fields replaced.

        Used by the CLI to apply `--max-parallel` / `--episode-timeout` /
        `--cost-cap-usd` overrides on top of a YAML-loaded config without
        reaching into `__dict__` directly (which would skip any future
        field validation in `__post_init__`).
        """
        return replace(self, **changes)


def parse_reasoning_replay(value: Any) -> str:
    """Validate and normalize a model's plaintext reasoning-history policy."""
    normalized = "auto" if value is None else value
    allowed = {"auto", "preserve_content", "drop"}
    if not isinstance(normalized, str) or normalized not in allowed:
        raise ValueError(
            f"reasoning_replay must be one of {sorted(allowed)}, got {normalized!r}"
        )
    return normalized


def parse_config(config_dict: dict[str, Any]) -> BenchmarkConfig:
    """Parse a benchmark YAML/JSON dict into a `BenchmarkConfig`.

    Raises `ValueError` on missing required fields.
    """
    if "benchmark_id" not in config_dict:
        raise ValueError("benchmark config missing required field 'benchmark_id'")
    models = [
        ModelSpec(
            id=m["id"],
            params=dict(m.get("params") or {}),
            reasoning_replay=parse_reasoning_replay(m.get("reasoning_replay")),
        )
        for m in config_dict.get("models") or []
    ]
    envs = [
        EnvSpec(
            id=e["id"],
            image=e["image"],
            task_type=e.get("task_type", "binary_task"),
            interface=e.get("interface", "rl.mcp.v8_task.v1"),
        )
        for e in config_dict.get("envs") or []
    ]
    seeds = list(config_dict.get("seeds") or [1])
    if not models or not envs:
        raise ValueError("benchmark config needs at least one model and one env")
    budgets_dict = config_dict.get("budgets") or {}
    # token_budget / context_budget honour explicit `null` (disable
    # enforcement, keep diagnostic tracking). Missing keys fall back to
    # the historical defaults.
    raw_token_budget = budgets_dict.get("token_budget", 2_500_000)
    raw_context_budget = budgets_dict.get("context_budget", 180_000)
    budgets = Budgets(
        turn_budget=int(budgets_dict.get("turn_budget", 300)),
        token_budget=int(raw_token_budget) if raw_token_budget is not None else None,
        context_budget=int(raw_context_budget) if raw_context_budget is not None else None,
        max_tokens=int(budgets_dict.get("max_tokens", 16_384)),
    )
    cost_cap = config_dict.get("cost_cap_usd")
    raw_init_prompt = config_dict.get("init_prompt")
    raw_init_prompt_path = config_dict.get("init_prompt_path")
    if raw_init_prompt is not None and raw_init_prompt_path is not None:
        raise ValueError(
            "benchmark config sets BOTH init_prompt and init_prompt_path; "
            "use one or the other (literal text vs. file reference)."
        )
    raw_hint = config_dict.get("init_prompt_hint")
    raw_hint_path = config_dict.get("init_prompt_hint_path")
    if raw_hint is not None and raw_hint_path is not None:
        raise ValueError(
            "benchmark config sets BOTH init_prompt_hint and "
            "init_prompt_hint_path; use one or the other."
        )
    return BenchmarkConfig(
        benchmark_id=config_dict["benchmark_id"],
        models=models,
        envs=envs,
        seeds=seeds,
        budgets=budgets,
        max_parallel=int(config_dict.get("max_parallel", 2)),
        init_prompt=raw_init_prompt,
        init_prompt_path=(
            Path(raw_init_prompt_path) if raw_init_prompt_path is not None else None
        ),
        init_prompt_hint=raw_hint,
        init_prompt_hint_path=(
            Path(raw_hint_path) if raw_hint_path is not None else None
        ),
        episode_timeout_s=int(
            config_dict.get("episode_timeout_s", DEFAULT_EPISODE_TIMEOUT_S),
        ),
        cost_cap_usd=float(cost_cap) if cost_cap is not None else None,
        nudges=parse_nudges(config_dict.get("nudges")),
    )


def apply_overrides(
    config_dict: dict[str, Any],
    sets: list[str] | None,
) -> dict[str, Any]:
    """Apply `--set <dotted.key>=<value>` overrides to a config dict.

    Each entry is a `key=value` string. The value is YAML-parsed
    (`100` → int, `true` → bool, `[a, b]` → list of strings, etc.) so
    callers don't have to think about quoting. Dotted keys walk into
    nested dicts (`budgets.turn_budget=100`); intermediate dicts are
    created if missing.

    Returns a deep-copied dict; the input is unchanged. parse_config
    runs after this and is the canonical validator — overrides that
    produce a malformed config will surface there, not here.
    """
    if not sets:
        return config_dict

    out = copy.deepcopy(config_dict)
    for entry in sets:
        if "=" not in entry:
            raise ValueError(
                f"--set expects key=value, got {entry!r}"
            )
        raw_key, _, raw_value = entry.partition("=")
        key = raw_key.strip()
        if not key:
            raise ValueError(f"--set key is empty in {entry!r}")
        # YAML-parse the value so users don't double-quote ints/bools/lists.
        # An empty RHS becomes `None` (yaml.safe_load("") → None), which is
        # the right shape for fields like cost_cap_usd that are nullable.
        value = _yaml.safe_load(raw_value)
        path = key.split(".")
        cursor = out
        for piece in path[:-1]:
            existing = cursor.get(piece)
            if not isinstance(existing, dict):
                cursor[piece] = {}
            cursor = cursor[piece]
        cursor[path[-1]] = value
    return out
