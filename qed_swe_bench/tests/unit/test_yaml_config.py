"""Benchmark configs parse from YAML (and JSON, since YAML is a superset)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qed_swe_bench.runner.orchestrator import parse_config

SAMPLES = Path(__file__).resolve().parent.parent.parent.parent / "examples" / "benchmarks"


@pytest.mark.parametrize(
    "yaml_path",
    sorted(SAMPLES.glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_example_yaml_configs_parse(yaml_path: Path) -> None:
    obj = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    bench = parse_config(obj)
    assert bench.benchmark_id
    assert bench.models, f"{yaml_path.name} declared no models"
    assert bench.envs, f"{yaml_path.name} declared no envs"
    assert bench.seeds, f"{yaml_path.name} declared no seeds"


def test_yaml_with_comments_parses() -> None:
    """Comments are the whole point of switching from JSON. Confirm they survive."""
    text = """\
# top-level comment
benchmark_id: t  # inline comment
models:
  - id: anthropic/claude-haiku-4-5
envs:
  - id: e
    image: local/e:latest          # local-image ref
    task_type: binary_task
seeds:
  - 1
  - 2
"""
    bench = parse_config(yaml.safe_load(text))
    assert bench.benchmark_id == "t"
    assert bench.seeds == [1, 2]


def test_json_input_still_works() -> None:
    """yaml.safe_load handles JSON too; backwards-compat for any old configs."""
    json_text = (
        '{"benchmark_id": "t", '
        '"models": [{"id": "m"}], "envs": [{"id": "e", "image": "i"}], "seeds": [1]}'
    )
    bench = parse_config(yaml.safe_load(json_text))
    assert bench.benchmark_id == "t"
