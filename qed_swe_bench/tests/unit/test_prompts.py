"""Tests for the init-prompt resolution path.

The container's MCP setup() carries all bug-specific framing; the runner
only sends an init prompt (first user turn) and an optional hint
appended after it. This file pins:

  - parse_config accepts inline init_prompt OR init_prompt_path
  - parse_config accepts inline init_prompt_hint OR init_prompt_hint_path
  - both are optional (no required-prompt rule anymore — the container is
    authoritative for content, the init prompt is just a default pointer)
  - run_benchmark's resolution chain reads the bench fields and appends
    the hint with a blank line between
  - default_init_prompt() is a short pointer at setup() / grade()
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qed_swe_bench.runner.orchestrator_config import parse_config
from qed_swe_bench.runner.prompts import default_init_prompt


# ---------------- parse_config: init_prompt accepted both ways ----------------


def _base_config(**overrides: object) -> dict:
    """Minimal valid YAML dict; overrides plug or override fields."""
    base: dict = {
        "benchmark_id": "test-init",
        "models": [{"id": "anthropic/claude-haiku-4-5"}],
        "envs": [{"id": "e", "image": "local/x:latest"}],
        "seeds": [1],
    }
    base.update(overrides)
    return base


def test_parse_config_accepts_inline_init_prompt() -> None:
    cfg = parse_config(_base_config(init_prompt="hello"))
    assert cfg.init_prompt == "hello"
    assert cfg.init_prompt_path is None


def test_parse_config_accepts_init_prompt_path(tmp_path: Path) -> None:
    template = tmp_path / "init.template"
    template.write_text("path-loaded init", encoding="utf-8")
    cfg = parse_config(_base_config(init_prompt_path=str(template)))
    assert cfg.init_prompt is None
    assert cfg.init_prompt_path == template


def test_parse_config_omits_init_prompt_entirely() -> None:
    """No required-prompt rule: defaults are fine. Container's setup()
    carries the bug content; init prompt falls through to the runner
    default."""
    cfg = parse_config(_base_config())
    assert cfg.init_prompt is None
    assert cfg.init_prompt_path is None
    assert cfg.init_prompt_hint is None
    assert cfg.init_prompt_hint_path is None


def test_parse_config_rejects_both_init_prompt_forms() -> None:
    """Inline AND path is ambiguous — reject."""
    with pytest.raises(ValueError, match="BOTH init_prompt and init_prompt_path"):
        parse_config(_base_config(init_prompt="x", init_prompt_path="/tmp/y"))


# ---------------- parse_config: init_prompt_hint accepted both ways -----------


def test_parse_config_accepts_inline_init_prompt_hint() -> None:
    cfg = parse_config(_base_config(init_prompt_hint="extra hint"))
    assert cfg.init_prompt_hint == "extra hint"
    assert cfg.init_prompt_hint_path is None


def test_parse_config_accepts_init_prompt_hint_path(tmp_path: Path) -> None:
    template = tmp_path / "hint.template"
    template.write_text("path-loaded hint", encoding="utf-8")
    cfg = parse_config(_base_config(init_prompt_hint_path=str(template)))
    assert cfg.init_prompt_hint is None
    assert cfg.init_prompt_hint_path == template


def test_parse_config_rejects_both_hint_forms() -> None:
    with pytest.raises(
        ValueError, match="BOTH init_prompt_hint and init_prompt_hint_path"
    ):
        parse_config(
            _base_config(
                init_prompt_hint="x", init_prompt_hint_path="/tmp/y"
            )
        )


# ---------------- default_init_prompt -----------------------------------------


def test_default_init_prompt_points_at_setup_and_grade() -> None:
    out = default_init_prompt()
    assert "setup()" in out
    assert "grade(" in out
    assert len(out) < 500
