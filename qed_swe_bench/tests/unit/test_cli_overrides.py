"""Tests for `apply_overrides` — the engine behind `--set <dotted>=<value>`.

Pinned semantics:
  - YAML-parsed values (`100` → int, `true` → bool, `[a,b]` → list).
  - Dotted keys walk into nested dicts; missing intermediates are created.
  - Empty/malformed inputs error early.
  - Deep-copy: input dict is never mutated.
  - Result feeds straight into parse_config — overrides surface as
    parse-time errors when they produce a malformed config.
"""

from __future__ import annotations

import copy

import pytest

from qed_swe_bench.runner.orchestrator_config import (
    apply_overrides,
    parse_config,
)


def _base() -> dict:
    return {
        "benchmark_id": "test",
        "models": [{"id": "anthropic/claude-haiku-4-5"}],
        "envs": [{"id": "e", "image": "local/x:latest"}],
        "seeds": [1],
        "budgets": {
            "turn_budget": 300,
            "token_budget": 2_500_000,
            "context_budget": 180_000,
            "max_tokens": 16_384,
        },
    }


# ---------------- value type coercion ----------------


def test_apply_overrides_yaml_parses_int() -> None:
    cfg = apply_overrides(_base(), ["budgets.turn_budget=100"])
    assert cfg["budgets"]["turn_budget"] == 100
    assert isinstance(cfg["budgets"]["turn_budget"], int)


def test_apply_overrides_yaml_parses_float() -> None:
    cfg = apply_overrides(_base(), ["cost_cap_usd=2.5"])
    assert cfg["cost_cap_usd"] == 2.5
    assert isinstance(cfg["cost_cap_usd"], float)


def test_apply_overrides_yaml_parses_bool() -> None:
    cfg = apply_overrides(_base(), ["nudges=true"])
    assert cfg["nudges"] is True


def test_apply_overrides_yaml_parses_list() -> None:
    cfg = apply_overrides(_base(), ["seeds=[1, 2, 3]"])
    assert cfg["seeds"] == [1, 2, 3]


def test_apply_overrides_keeps_strings() -> None:
    cfg = apply_overrides(_base(), ["init_prompt=Use setup() first."])
    assert cfg["init_prompt"] == "Use setup() first."


def test_apply_overrides_empty_value_becomes_none() -> None:
    """yaml.safe_load('') is None — useful for nullable fields like
    cost_cap_usd."""
    cfg = apply_overrides(_base(), ["cost_cap_usd="])
    assert cfg["cost_cap_usd"] is None


# ---------------- dotted-path traversal ----------------


def test_apply_overrides_walks_nested_dict() -> None:
    cfg = apply_overrides(_base(), ["budgets.max_tokens=8192"])
    assert cfg["budgets"]["max_tokens"] == 8192
    # Sibling fields preserved.
    assert cfg["budgets"]["turn_budget"] == 300


def test_apply_overrides_creates_missing_intermediate_dict() -> None:
    """If the YAML doesn't declare `budgets:` at all, --set creates it."""
    base = _base()
    del base["budgets"]
    cfg = apply_overrides(base, ["budgets.turn_budget=50"])
    assert cfg["budgets"] == {"turn_budget": 50}


def test_apply_overrides_replaces_non_dict_intermediate() -> None:
    """If a path-piece points at a non-dict, it gets replaced with a fresh
    dict (otherwise we'd silently traverse into something nonsensical)."""
    base = _base()
    base["budgets"] = "garbage"  # type: ignore[assignment]
    cfg = apply_overrides(base, ["budgets.turn_budget=50"])
    assert cfg["budgets"] == {"turn_budget": 50}


# ---------------- multiple sets ----------------


def test_apply_overrides_applies_all_in_order() -> None:
    cfg = apply_overrides(
        _base(),
        [
            "budgets.turn_budget=10",
            "cost_cap_usd=1.5",
            "nudges=true",
        ],
    )
    assert cfg["budgets"]["turn_budget"] == 10
    assert cfg["cost_cap_usd"] == 1.5
    assert cfg["nudges"] is True


def test_apply_overrides_later_set_wins() -> None:
    cfg = apply_overrides(
        _base(),
        ["budgets.turn_budget=10", "budgets.turn_budget=20"],
    )
    assert cfg["budgets"]["turn_budget"] == 20


# ---------------- error cases ----------------


def test_apply_overrides_rejects_missing_equals() -> None:
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(_base(), ["budgets.turn_budget100"])


def test_apply_overrides_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="key is empty"):
        apply_overrides(_base(), ["=10"])


# ---------------- non-mutating ----------------


def test_apply_overrides_does_not_mutate_input() -> None:
    base = _base()
    snapshot = copy.deepcopy(base)
    apply_overrides(base, ["budgets.turn_budget=100"])
    assert base == snapshot


def test_apply_overrides_no_sets_returns_input_unchanged() -> None:
    base = _base()
    assert apply_overrides(base, None) is base
    assert apply_overrides(base, []) is base


# ---------------- integration with parse_config ----------------


def test_apply_overrides_then_parse_config_picks_up_changes() -> None:
    """The full intended pipeline: --set → apply_overrides → parse_config."""
    cfg_dict = apply_overrides(
        _base(),
        [
            "budgets.turn_budget=50",
            "cost_cap_usd=10",
            "init_prompt_hint=cheap caps first",
        ],
    )
    bench = parse_config(cfg_dict)
    assert bench.budgets.turn_budget == 50
    assert bench.cost_cap_usd == 10.0
    assert bench.init_prompt_hint == "cheap caps first"


def test_apply_overrides_malformed_value_surfaces_at_parse_config() -> None:
    """If --set produces a config that parse_config can't validate, the
    error surfaces there. apply_overrides itself doesn't validate."""
    cfg_dict = apply_overrides(_base(), ["models=anthropic/claude-haiku-4-5"])
    # `models` is now a string, not a list-of-dicts.
    with pytest.raises((TypeError, AttributeError, KeyError)):
        parse_config(cfg_dict)
