"""Tests for the provider-agnostic helpers in `runner/llm/base.py` —
specifically the model-match logic that backstops silent provider rerouting.
"""

from __future__ import annotations

import pytest

from qed_swe_bench.runner.llm.base import (
    ModelMismatchError,
    served_matches_requested,
)


# ---------------- served_matches_requested ----------------


@pytest.mark.parametrize(
    "requested,served",
    [
        # Exact match (after stripping prefix on both sides).
        ("openai/gpt-5.5", "gpt-5.5"),
        ("anthropic/claude-haiku-4-5", "claude-haiku-4-5"),
        ("mock/test", "mock/test"),  # mock client keeps prefix on both sides
        # Dated snapshot extension on a hyphen boundary.
        ("openai/gpt-5.5", "gpt-5.5-2026-04-23"),
        ("anthropic/claude-haiku-4-5", "claude-haiku-4-5-20250101"),
        # Stacked prefixes (e.g. openrouter wrapping openai).
        ("openrouter/openai/gpt-5.5", "gpt-5.5"),
        # Empty served string is treated as match (test stubs / older gateways).
        ("openai/gpt-5.5", ""),
    ],
)
def test_served_matches_requested_accepts(requested: str, served: str) -> None:
    assert served_matches_requested(requested=requested, served=served) is True


@pytest.mark.parametrize(
    "requested,served",
    [
        # The documented OpenAI cyber-policy downgrade.
        ("openai/gpt-5.5", "gpt-5.2"),
        ("openai/gpt-5.5", "gpt-5.2-2026-01-15"),
        # Different family entirely — silent reroute we'd want to hear about.
        ("openai/gpt-5.5", "gpt-4o"),
        # Prefix-shares-string trap: gpt-5 is NOT gpt-5.5. Hyphen-boundary
        # rule must reject this.
        ("openai/gpt-5", "gpt-5.5-2026-04-23"),
        # Anthropic family swap.
        ("anthropic/claude-sonnet-4-6", "claude-haiku-4-5-20250101"),
    ],
)
def test_served_matches_requested_rejects(requested: str, served: str) -> None:
    assert served_matches_requested(requested=requested, served=served) is False


# ---------------- ModelMismatchError ----------------


def test_model_mismatch_error_carries_both_ids() -> None:
    """The loop logs requested + served via str(exc) into failure_reason; both
    must appear so post-hoc audit can identify what was downgraded to what."""
    err = ModelMismatchError(requested="openai/gpt-5.5", served="gpt-5.2")
    assert err.requested == "openai/gpt-5.5"
    assert err.served == "gpt-5.2"
    s = str(err)
    assert "openai/gpt-5.5" in s
    assert "gpt-5.2" in s
