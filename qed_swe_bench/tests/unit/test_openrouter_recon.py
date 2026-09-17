"""Unit tests for runner/openrouter_recon.py.

Covers the post-episode reconciliation flow that turns OR
`gen-...` ids in transcript.jsonl into authoritative billed costs
via OR's /api/v1/generation endpoint. HTTP is stubbed; no real
network calls in tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from qed_swe_bench.runner.openrouter_recon import (
    GenerationCost,
    ReconciliationSummary,
    collect_or_generation_ids,
    fetch_generation_cost,
    reconcile_run_dir,
)


# ---------------- transcript parsing ----------------


def _write_transcript(path: Path, ai_entries: list[dict]) -> None:
    """Helper: write a transcript.jsonl with the given AI entries."""
    with path.open("w", encoding="utf-8") as f:
        for entry in ai_entries:
            entry.setdefault("role", "ai")
            f.write(json.dumps(entry) + "\n")


def test_collect_only_returns_or_gen_ids(tmp_path: Path) -> None:
    """Filter to ids matching `gen-...` so direct-provider ids
    (Anthropic msg_..., OpenAI chatcmpl_...) don't get sent to OR."""
    t = tmp_path / "transcript.jsonl"
    _write_transcript(t, [
        {"provider_response_id": "gen-aaa111"},
        {"provider_response_id": "msg_anthropic_bbb"},  # filter out
        {"provider_response_id": "chatcmpl-openai-ccc"},  # filter out
        {"provider_response_id": "gen-ddd222"},
        {"provider_response_id": None},  # filter out
        {},  # no field at all → filter out
    ])
    ids = collect_or_generation_ids(t)
    assert ids == ["gen-aaa111", "gen-ddd222"]


def test_collect_skips_non_ai_rows(tmp_path: Path) -> None:
    """Only AI turns can have provider_response_id; human/tool entries are
    skipped."""
    t = tmp_path / "transcript.jsonl"
    with t.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"role": "human", "content": "go"}) + "\n")
        f.write(json.dumps({"role": "ai", "provider_response_id": "gen-1"}) + "\n")
        f.write(json.dumps({"role": "tool", "name": "exec", "content": "ok"}) + "\n")
    assert collect_or_generation_ids(t) == ["gen-1"]


def test_collect_handles_missing_file(tmp_path: Path) -> None:
    assert collect_or_generation_ids(tmp_path / "nope.jsonl") == []


def test_collect_handles_malformed_lines(tmp_path: Path) -> None:
    """A bad JSON line shouldn't crash the whole pass."""
    t = tmp_path / "transcript.jsonl"
    with t.open("w", encoding="utf-8") as f:
        f.write("not valid json {{{\n")
        f.write(json.dumps({"role": "ai", "provider_response_id": "gen-good"}) + "\n")
    assert collect_or_generation_ids(t) == ["gen-good"]


# ---------------- fetch_generation_cost (HTTP-stubbed) ----------------


def test_fetch_returns_parsed_cost_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: OR returns a `data` envelope with cost fields."""
    body = {
        "data": {
            "total_cost": 0.0042,
            "cache_discount": 0.0008,
            "upstream_inference_cost": 0.0030,
            "provider_name": "DeepInfra",
            "native_tokens_cached": 12000,
        }
    }

    def fake_get(url, *, params, headers, timeout):
        assert "openrouter.ai" in url
        assert params == {"id": "gen-abc"}
        assert headers["Authorization"] == "Bearer test-key"
        return _MockResp(200, body)

    with patch("qed_swe_bench.runner.openrouter_recon.httpx.get", side_effect=fake_get):
        gc = fetch_generation_cost("gen-abc", api_key="test-key")
    assert gc is not None
    assert gc.generation_id == "gen-abc"
    assert gc.total_cost == 0.0042
    assert gc.cache_discount == 0.0008
    assert gc.upstream_inference_cost == 0.0030
    assert gc.provider_name == "DeepInfra"
    assert gc.native_tokens_cached == 12000


def test_fetch_returns_none_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 4xx/5xx response → None (best-effort, log + skip)."""
    def fake_get(*args, **kwargs):
        return _MockResp(404, {"error": "not found"})

    with patch("qed_swe_bench.runner.openrouter_recon.httpx.get", side_effect=fake_get):
        assert fetch_generation_cost("gen-missing", api_key="k") is None


def test_fetch_returns_none_on_network_error() -> None:
    """A connection error / timeout → None, no exception escapes."""
    def fake_get(*args, **kwargs):
        raise httpx.ConnectError("network down")

    with patch("qed_swe_bench.runner.openrouter_recon.httpx.get", side_effect=fake_get):
        assert fetch_generation_cost("gen-x", api_key="k") is None


def test_fetch_handles_payload_without_data_envelope() -> None:
    """OR's docs wrap response in `data`, but be defensive in case the
    envelope shape changes."""
    body = {"total_cost": 0.005}  # flat, no `data` key

    def fake_get(*args, **kwargs):
        return _MockResp(200, body)

    with patch("qed_swe_bench.runner.openrouter_recon.httpx.get", side_effect=fake_get):
        gc = fetch_generation_cost("gen-flat", api_key="k")
    assert gc is not None
    assert gc.total_cost == 0.005


# ---------------- reconcile_run_dir (full path, HTTP-stubbed) ----------------


def test_reconcile_returns_none_when_no_or_turns(tmp_path: Path) -> None:
    """A run-dir with no OR `gen-...` ids in its transcript returns
    None — caller skips the cost.json write entirely."""
    _write_transcript(tmp_path / "transcript.jsonl", [
        {"provider_response_id": "msg_anthropic_xyz"},
    ])
    assert reconcile_run_dir(tmp_path, api_key="k") is None


def test_reconcile_aggregates_costs_across_turns(tmp_path: Path) -> None:
    """Sums total_cost / cache_discount / upstream_inference_cost per
    turn into a single ReconciliationSummary."""
    _write_transcript(tmp_path / "transcript.jsonl", [
        {"provider_response_id": "gen-1"},
        {"provider_response_id": "gen-2"},
        {"provider_response_id": "gen-3"},
    ])
    payloads = {
        "gen-1": {"total_cost": 0.001, "cache_discount": 0.0001,
                   "upstream_inference_cost": 0.0009, "provider_name": "DeepInfra"},
        "gen-2": {"total_cost": 0.002, "cache_discount": 0.0002,
                   "upstream_inference_cost": 0.0018, "provider_name": "DeepInfra"},
        "gen-3": {"total_cost": 0.003, "cache_discount": 0.0003,
                   "upstream_inference_cost": 0.0027, "provider_name": "Together"},
    }

    def fake_get(url, *, params, headers, timeout):
        return _MockResp(200, {"data": payloads[params["id"]]})

    with patch("qed_swe_bench.runner.openrouter_recon.httpx.get", side_effect=fake_get):
        summary = reconcile_run_dir(tmp_path, api_key="k")

    assert summary is not None
    assert summary.n_turns_recorded == 3
    assert summary.n_turns_resolved == 3
    assert summary.n_turns_skipped == 0
    assert not summary.incomplete
    assert summary.cost_usd_authoritative == pytest.approx(0.006)
    assert summary.cache_discount_total == pytest.approx(0.0006)
    assert summary.upstream_cost_total == pytest.approx(0.0054)
    # providers_seen accumulates per-turn (with duplicates); to_dict()
    # is what dedupes.
    assert sorted(set(summary.providers_seen)) == ["DeepInfra", "Together"]
    assert summary.to_dict()["providers_seen"] == ["DeepInfra", "Together"]


def test_reconcile_marks_incomplete_on_partial_failure(tmp_path: Path) -> None:
    """If some turns succeed and some fail, summary records
    `incomplete=True` so downstream tooling can flag the cell as having
    incomplete authoritative cost data."""
    _write_transcript(tmp_path / "transcript.jsonl", [
        {"provider_response_id": "gen-ok"},
        {"provider_response_id": "gen-bad"},
    ])
    call_count = {"n": 0}

    def fake_get(url, *, params, headers, timeout):
        call_count["n"] += 1
        if params["id"] == "gen-ok":
            return _MockResp(200, {"data": {"total_cost": 0.005}})
        return _MockResp(500, {"error": "upstream"})

    with patch("qed_swe_bench.runner.openrouter_recon.httpx.get", side_effect=fake_get):
        summary = reconcile_run_dir(tmp_path, api_key="k")

    assert summary is not None
    assert summary.n_turns_resolved == 1
    assert summary.n_turns_skipped == 1
    assert summary.incomplete is True
    assert summary.cost_usd_authoritative == pytest.approx(0.005)


def test_reconcile_skips_when_no_api_key(tmp_path: Path, monkeypatch) -> None:
    """No OPENROUTER_API_KEY → return summary with all turns skipped +
    incomplete=True. We don't want to silently hide that recon was
    intended but couldn't run."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _write_transcript(tmp_path / "transcript.jsonl", [
        {"provider_response_id": "gen-x"},
    ])
    summary = reconcile_run_dir(tmp_path, api_key=None)
    assert summary is not None
    assert summary.n_turns_recorded == 1
    assert summary.n_turns_skipped == 1
    assert summary.incomplete is True


def test_summary_to_dict_round_trip() -> None:
    """The dict shape is what gets serialized into cost.json."""
    s = ReconciliationSummary(
        n_turns_recorded=5,
        n_turns_resolved=4,
        n_turns_skipped=1,
        cost_usd_authoritative=0.123456789,
        cache_discount_total=0.012345,
        upstream_cost_total=0.111,
        providers_seen=["DeepInfra", "Together", "DeepInfra"],
        incomplete=True,
    )
    d = s.to_dict()
    assert d["n_turns_recorded"] == 5
    assert d["n_turns_resolved"] == 4
    assert d["n_turns_skipped"] == 1
    # Rounded to 6 decimals
    assert d["cost_usd_authoritative"] == 0.123457
    assert d["cache_discount_total"] == 0.012345
    assert d["upstream_cost_total"] == 0.111
    # Deduplicated + sorted
    assert d["providers_seen"] == ["DeepInfra", "Together"]
    assert d["incomplete"] is True


# ---------------- helpers ----------------


class _MockResp:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=None,  # type: ignore[arg-type]
            )

    def json(self) -> dict:
        return self._body
