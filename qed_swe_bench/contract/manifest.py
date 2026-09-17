"""Manifest record + loader.

A *manifest* is the per-env contract: which interface it implements, where
the image is, what task_type, expected_capabilities, integrity baseline,
how to grade. The runner consults the manifest at episode planning time;
the validator checks it at registration time and again before each
benchmark run.

Authored:
- by hand for V8 envs (one YAML per bug under `manifests/v8/`);
- by an adapter for envs imported from existing repos.

Canonical form is the JSON Schema at `contract/schemas/manifest.schema.json`.
This module loads YAML or JSON, validates against that schema, and
cross-checks the `interface` field against the registry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from qed_swe_bench.contract import interfaces

# ---------------------------------------------------------------------------
# Schema loading (one-shot, package-relative)
# ---------------------------------------------------------------------------

_SCHEMA_RESOURCE = ("qed_swe_bench.contract.schemas", "manifest.schema.json")


def load_schema() -> dict[str, Any]:
    """Load the manifest JSON Schema from package resources."""
    pkg, name = _SCHEMA_RESOURCE
    with resources.files(pkg).joinpath(name).open("r", encoding="utf-8") as f:
        return json.load(f)


# Build the validator once; jsonschema validators are reusable + thread-safe.
_VALIDATOR: Draft202012Validator | None = None


def _validator() -> Draft202012Validator:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = Draft202012Validator(load_schema())
    return _VALIDATOR


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ManifestError(Exception):
    """Base for manifest problems."""


class ManifestSchemaError(ManifestError):
    """Manifest doesn't validate against the JSON schema."""

    def __init__(self, errors: list[ValidationError]):
        # ValidationError.message + .json_path; aggregate for a useful str.
        lines = [f"  - {e.json_path}: {e.message}" for e in errors]
        super().__init__("manifest schema violations:\n" + "\n".join(lines))
        self.errors = errors


class ManifestInterfaceError(ManifestError):
    """Manifest references an interface not in the registry, or a tool
    surface inconsistent with the declared interface."""


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Manifest:
    """A validated, normalized manifest record.

    The dict form (``raw``) is the source of truth; this dataclass exists
    so the runner can pluck common fields without dict-spelunking.
    """

    env_id: str
    interface: str
    interface_flavor: str
    task_type: str
    image_ref: str
    image_digest: str | None
    project: str | None
    bug_id: str | None
    capability_class: str | None
    expected_capabilities: tuple[str, ...]
    expected_findings: tuple[dict[str, Any], ...]
    episode_tools: tuple[str, ...]
    evaluation_tools: tuple[str, ...]
    evaluate: dict[str, Any]
    integrity_baseline: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def grader_kind(self) -> str:
        return self.evaluate["kind"]

    @property
    def grader_command(self) -> tuple[str, ...] | None:
        if self.grader_kind != "cli_oneshot":
            return None
        return tuple(self.evaluate["command"])

    @property
    def grader_tool(self) -> str | None:
        if self.grader_kind != "mcp_tool":
            return None
        return self.evaluate["tool"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_text(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        loaded = yaml.safe_load(text)
    elif suffix == ".json":
        loaded = json.loads(text)
    else:
        raise ManifestError(f"unsupported manifest extension: {path}")
    if not isinstance(loaded, dict):
        raise ManifestError(f"manifest must be a mapping at top level: {path}")
    return loaded


def _validate_schema(raw: dict[str, Any]) -> None:
    errors = sorted(_validator().iter_errors(raw), key=lambda e: list(e.absolute_path))
    if errors:
        raise ManifestSchemaError(errors)


def _check_interface_consistency(raw: dict[str, Any]) -> None:
    """Cross-check fields the JSON schema can't enforce on its own."""
    iface_name = raw["interface"]
    iface = interfaces.lookup(iface_name)
    if iface is None:
        raise ManifestInterfaceError(
            f"interface '{iface_name}' is not in the registry "
            f"(known: {', '.join(interfaces.all_names())})"
        )

    # Episode tools must be a subset of the interface's declared tools.
    declared_tools = set(iface.mcp_tools)
    mcp = raw.get("mcp") or {}
    episode_tools = set(mcp.get("episode_tools") or ())
    if episode_tools and declared_tools and not episode_tools.issubset(declared_tools):
        extra = sorted(episode_tools - declared_tools)
        raise ManifestInterfaceError(
            f"interface {iface_name} does not declare these episode tools: {extra}"
        )

    # Evaluation tools must NOT overlap episode tools (reward-hacking guard).
    evaluation_tools = set(mcp.get("evaluation_tools") or ())
    overlap = episode_tools & evaluation_tools
    if overlap:
        raise ManifestInterfaceError(
            f"evaluation_tools must be disjoint from episode_tools; overlap: {sorted(overlap)}"
        )

    # If the interface uses cli_oneshot grading, the manifest must too,
    # because the in-MCP grade tool is the reward-hacking surface we're
    # specifically forbidding for those interfaces.
    if iface.grader_kind == "cli_oneshot" and raw["evaluate"]["kind"] != "cli_oneshot":
        raise ManifestInterfaceError(
            f"interface {iface_name} requires evaluate.kind='cli_oneshot' "
            f"(got '{raw['evaluate']['kind']}')"
        )

    # Expected capabilities must be a subset of the interface's bitmap.
    expected = set(raw.get("expected_capabilities") or ())
    declared_caps = set(iface.capability_flags)
    if declared_caps and expected and not expected.issubset(declared_caps):
        extra = sorted(expected - declared_caps)
        raise ManifestInterfaceError(
            f"expected_capabilities references unknown flags for {iface_name}: {extra}"
        )


def from_dict(raw: dict[str, Any]) -> Manifest:
    """Validate `raw` and project into a Manifest dataclass."""
    _validate_schema(raw)
    _check_interface_consistency(raw)

    image = raw["image"]
    mcp = raw.get("mcp") or {}
    return Manifest(
        env_id=raw["env_id"],
        interface=raw["interface"],
        interface_flavor=raw.get("interface_flavor", "bench_v8"),
        task_type=raw["task_type"],
        image_ref=image["ref"],
        image_digest=image.get("digest"),
        project=raw.get("project"),
        bug_id=raw.get("bug_id"),
        capability_class=raw.get("capability_class"),
        expected_capabilities=tuple(raw.get("expected_capabilities") or ()),
        expected_findings=tuple(raw.get("expected_findings") or ()),
        episode_tools=tuple(mcp.get("episode_tools") or ()),
        evaluation_tools=tuple(mcp.get("evaluation_tools") or ()),
        evaluate=dict(raw["evaluate"]),
        integrity_baseline=dict(raw.get("integrity_baseline") or {}),
        metadata=dict(raw.get("metadata") or {}),
        raw=raw,
    )


def load(path: str | Path) -> Manifest:
    """Read a manifest file (YAML or JSON) and return a validated Manifest."""
    return from_dict(_load_text(Path(path)))
