"""Translation from LiteLLM/OpenAI-shape responses to NormalizedResponse.

These do NOT call litellm.completion; they exercise _to_normalized against
realistic stubs. Real-API tests are slow tier.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from qed_swe_bench.runner.llm.litellm_client import LiteLLMClient


def _stub_response(
    *,
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    finish_reason: str = "stop",
    model: str = "openai/gpt-5",
    response_id: str | None = None,
    reasoning_content: str | None = None,
):
    """Build a SimpleNamespace shaped like a litellm/openai ChatCompletion."""
    tc_objs = []
    for tc in tool_calls or []:
        tc_objs.append(
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=tc["arguments"]
                    if isinstance(tc["arguments"], str)
                    else json.dumps(tc["arguments"]),
                ),
            )
        )
    msg = SimpleNamespace(
        content=content, tool_calls=tc_objs or None,
        reasoning_content=reasoning_content,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens)
        if cached_tokens
        else None,
    )
    return SimpleNamespace(
        choices=[choice], usage=usage, model=model, id=response_id,
    )


def test_text_only_response() -> None:
    resp = _stub_response(content="hello", prompt_tokens=10, completion_tokens=5)
    norm = LiteLLMClient._to_normalized(resp)
    assert norm.text == "hello"
    assert norm.tool_calls == ()
    assert norm.usage.input_tokens == 10
    assert norm.usage.output_tokens == 5
    assert norm.usage.cache_read_tokens == 0
    assert norm.stop_reason == "stop"


def test_tool_call_with_well_formed_json() -> None:
    resp = _stub_response(
        tool_calls=[
            {"id": "call_1", "name": "exec", "arguments": {"cmd": "ls -la"}},
        ],
        finish_reason="tool_calls",
    )
    norm = LiteLLMClient._to_normalized(resp)
    assert len(norm.tool_calls) == 1
    assert norm.tool_calls[0].id == "call_1"
    assert norm.tool_calls[0].name == "exec"
    assert norm.tool_calls[0].arguments == {"cmd": "ls -la"}


def test_tool_call_with_malformed_json_fallback() -> None:
    """OSS models behind a gateway sometimes emit invalid JSON; we don't crash."""
    resp = _stub_response(
        tool_calls=[
            {
                "id": "call_x",
                "name": "exec",
                "arguments": "{not valid json",  # malformed string
            },
        ],
        finish_reason="tool_calls",
    )
    norm = LiteLLMClient._to_normalized(resp)
    assert len(norm.tool_calls) == 1
    assert norm.tool_calls[0].arguments == {}


def test_tool_call_with_null_arguments() -> None:
    resp = _stub_response(
        tool_calls=[{"id": "c", "name": "setup", "arguments": ""}],
    )
    norm = LiteLLMClient._to_normalized(resp)
    assert norm.tool_calls[0].arguments == {}


def test_tool_calls_with_list_arg_falls_back_to_dict() -> None:
    """A model that returns a JSON array as the arguments value is malformed
    for tool-call purposes. We coerce to {} rather than crash."""
    resp = _stub_response(
        tool_calls=[{"id": "c", "name": "setup", "arguments": "[1,2,3]"}],
    )
    norm = LiteLLMClient._to_normalized(resp)
    assert norm.tool_calls[0].arguments == {}


def test_cache_read_tokens_populated_from_prompt_tokens_details() -> None:
    """LiteLLM reports prompt_tokens INCLUSIVE of cached_tokens (the
    OpenAI/LiteLLM convention — verified live across gpt-5.5, Gemini 3.1
    Pro, GLM, Kimi, MiniMax on 2026-05-13). NormalizedUsage uses the
    DISJOINT convention (Anthropic-native style: input_tokens = base only,
    cache_read separate) so compute_cost can sum the three categories
    without subtraction regardless of upstream provider. The populator
    subtracts cache_read here so downstream code sees a consistent
    disjoint shape."""
    resp = _stub_response(
        content="ok", prompt_tokens=120, completion_tokens=10, cached_tokens=80
    )
    norm = LiteLLMClient._to_normalized(resp)
    # 120 prompt - 80 cached = 40 base
    assert norm.usage.input_tokens == 40
    assert norm.usage.cache_read_tokens == 80


def test_input_tokens_clamped_to_zero_if_cache_exceeds_prompt() -> None:
    """Defensive: if a provider reports cached_tokens > prompt_tokens (shouldn't
    happen under the inclusive convention, but the data has occasionally
    been observed off-by-one), clamp at 0 rather than emit negative."""
    resp = _stub_response(
        content="ok", prompt_tokens=50, completion_tokens=5, cached_tokens=80
    )
    norm = LiteLLMClient._to_normalized(resp)
    assert norm.usage.input_tokens == 0
    assert norm.usage.cache_read_tokens == 80


def test_reasoning_tokens_populated_from_completion_tokens_details() -> None:
    """OpenAI reasoning models (gpt-5, gpt-5.5, o-series) report reasoning
    tokens under completion_tokens_details. Surface them so xhigh-tier
    requests leave evidence they actually fired."""
    msg = SimpleNamespace(content="ok", tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=2048,
        prompt_tokens_details=None,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=1900),
    )
    resp = SimpleNamespace(
        choices=[choice], usage=usage, model="gpt-5.5-2026-04-23"
    )
    norm = LiteLLMClient._to_normalized(resp)
    assert norm.usage.reasoning_tokens == 1900
    assert norm.model == "gpt-5.5-2026-04-23"


def test_reasoning_tokens_default_zero_when_field_absent() -> None:
    """Non-reasoning models / older API shapes: parser must default to 0."""
    resp = _stub_response(content="ok", prompt_tokens=10, completion_tokens=5)
    norm = LiteLLMClient._to_normalized(resp)
    assert norm.usage.reasoning_tokens == 0


def test_provider_response_id_captured_when_present() -> None:
    """OpenRouter returns its `gen-...` id in the response body; we expose
    it on NormalizedResponse so the OR cost reconciler can look it up
    via /api/v1/generation."""
    resp = _stub_response(content="x", response_id="gen-12345abcdef")
    norm = LiteLLMClient._to_normalized(resp)
    assert norm.provider_response_id == "gen-12345abcdef"


def test_provider_response_id_none_when_absent() -> None:
    """Direct providers may or may not include `id`; default is None."""
    resp = _stub_response(content="x", response_id=None)
    norm = LiteLLMClient._to_normalized(resp)
    assert norm.provider_response_id is None


def test_reasoning_content_captured_when_present() -> None:
    """Moonshot's API returns the model's thinking trace in
    `message.reasoning_content`. _to_normalized must surface it on
    NormalizedResponse so messages.py can replay it on the next turn
    (otherwise Moonshot 400s and LiteLLM injects a placeholder)."""
    resp = _stub_response(
        content="visible answer",
        reasoning_content="step 1: parse the patch\nstep 2: locate the function\n...",
    )
    norm = LiteLLMClient._to_normalized(resp)
    assert norm.reasoning_content == (
        "step 1: parse the patch\nstep 2: locate the function\n..."
    )


def test_reasoning_content_none_when_absent() -> None:
    """Most providers (gpt-5, Gemini, GLM, MiniMax via OR) don't surface
    a separate reasoning_content field — their reasoning is server-side
    or counted via reasoning_tokens. Empty string and missing both
    coerce to None so the message builder can `if resp.reasoning_content:`
    without surprises."""
    resp = _stub_response(content="x")  # no reasoning_content
    norm = LiteLLMClient._to_normalized(resp)
    assert norm.reasoning_content is None

    resp_empty = _stub_response(content="x", reasoning_content="")
    norm_empty = LiteLLMClient._to_normalized(resp_empty)
    assert norm_empty.reasoning_content is None  # empty string → None


def test_route_is_litellm_without_gateway() -> None:
    client = LiteLLMClient(model="openai/gpt-5")
    assert client.route == "litellm"
    assert client.api_base is None


def test_route_is_litellm_gateway_with_api_base() -> None:
    client = LiteLLMClient(model="openai/llama-3.3-70b", api_base="https://gw/v1")
    assert client.route == "litellm_gateway"
    assert client.api_base == "https://gw/v1"


# ---------------- request-timeout resolution ----------------

from qed_swe_bench.runner.llm.litellm_client import (  # noqa: E402
    DEFAULT_RATE_LIMIT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT_S,
    _is_rate_limit,
    _parse_retry_after,
    _rate_limit_max_retries,
    _request_timeout_s,
)


def test_request_timeout_default_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LITELLM_REQUEST_TIMEOUT_S", raising=False)
    assert _request_timeout_s() == DEFAULT_REQUEST_TIMEOUT_S
    assert DEFAULT_REQUEST_TIMEOUT_S >= 60, (
        "default must be generous enough for multi-turn gateway calls; "
        "see soak-resilience TimeoutError diagnosis"
    )


def test_request_timeout_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_REQUEST_TIMEOUT_S", "120")
    assert _request_timeout_s() == 120


def test_request_timeout_garbage_env_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LITELLM_REQUEST_TIMEOUT_S", "not-a-number")
    with caplog.at_level("WARNING"):
        assert _request_timeout_s() == DEFAULT_REQUEST_TIMEOUT_S
    assert any("not a positive int" in r.message for r in caplog.records)


def test_request_timeout_zero_or_negative_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for bad in ("0", "-30"):
        monkeypatch.setenv("LITELLM_REQUEST_TIMEOUT_S", bad)
        assert _request_timeout_s() == DEFAULT_REQUEST_TIMEOUT_S


# ---------------- rate-limit retry plumbing ----------------


def test_rate_limit_default_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", raising=False)
    assert _rate_limit_max_retries() == DEFAULT_RATE_LIMIT_MAX_RETRIES


def test_rate_limit_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", "10")
    assert _rate_limit_max_retries() == 10


def test_rate_limit_zero_disables_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", "0")
    assert _rate_limit_max_retries() == 0


def test_rate_limit_garbage_env_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", "junk")
    with caplog.at_level("WARNING"):
        assert _rate_limit_max_retries() == DEFAULT_RATE_LIMIT_MAX_RETRIES
    assert any("non-negative int" in r.message for r in caplog.records)


def test_litellm_passes_num_retries_to_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """litellm.completion should receive num_retries from the env-resolved
    helper, so its built-in rate-limit / 5xx backoff actually runs."""
    captured: list[dict] = []

    def fake_completion(**kwargs):
        captured.append(kwargs)
        return _stub_response(content="ok")

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", "7")

    from qed_swe_bench.runner.llm.litellm_client import LiteLLMClient
    client = LiteLLMClient(model="openai/gpt-5.5")
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=10,
    )
    assert captured[0].get("num_retries") == 7


# ---------------- retry-after parsing ----------------


def test_parse_retry_after_seconds_format() -> None:
    msg = (
        "Rate limit reached for gpt-5.5 ... "
        "Please try again in 5.247s. Visit https://..."
    )
    assert _parse_retry_after(msg) == 5.247


def test_parse_retry_after_milliseconds_format() -> None:
    msg = "...rate_limit_exceeded... Please try again in 921ms. Visit ..."
    # 921ms == 0.921s
    assert _parse_retry_after(msg) == pytest.approx(0.921)


def test_parse_retry_after_returns_none_when_absent() -> None:
    assert _parse_retry_after("some unrelated error message") is None


def test_parse_retry_after_prefers_ms_over_seconds_when_both_present() -> None:
    """Defensive: if a future provider message accidentally contains both
    units, the ms form is the one that comes first in OpenAI's wording
    ('try again in NNNms'). Pin which we honor so the behavior is stable."""
    msg = "try again in 200ms (or roughly 0.2s)"
    assert _parse_retry_after(msg) == pytest.approx(0.2)


# ---------------- _is_rate_limit duck-typing ----------------


def test_is_rate_limit_matches_on_class_name() -> None:
    class FakeRateLimitError(Exception):
        pass
    assert _is_rate_limit(FakeRateLimitError("anything")) is True


def test_is_rate_limit_matches_on_message_body() -> None:
    class GenericException(Exception):
        pass
    assert _is_rate_limit(GenericException("rate_limit_exceeded for foo")) is True
    assert _is_rate_limit(GenericException("Rate limit reached for gpt-5.5")) is True


def test_is_rate_limit_rejects_unrelated_errors() -> None:
    class UnrelatedError(Exception):
        pass
    assert _is_rate_limit(UnrelatedError("connection refused")) is False
    assert _is_rate_limit(ValueError("bad input")) is False


# ---------------- end-to-end retry behavior ----------------


def test_complete_retries_on_rate_limit_with_provider_supplied_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry loop should sleep the parsed retry-after, not tenacity's
    default exponential. Mock time.sleep to capture the actual wait."""
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)

    import qed_swe_bench.runner.llm.litellm_client as mod
    monkeypatch.setattr(mod.time, "sleep", fake_sleep)

    # First call: rate-limit with retry-after 0.5s. Second call: success.
    call_count = {"n": 0}

    def fake_completion(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise type("RateLimitError", (Exception,), {})(
                "rate_limit_exceeded ... Please try again in 0.5s."
            )
        return _stub_response(content="ok after retry")

    monkeypatch.setattr(mod.litellm, "completion", fake_completion)
    monkeypatch.setenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", "3")

    client = mod.LiteLLMClient(model="openai/gpt-5.5")
    resp = client.complete(messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=10)

    assert resp.text == "ok after retry"
    assert call_count["n"] == 2
    # First sleep should be 0.5s + jitter (0-0.5s), so 0.5 ≤ s < 1.0.
    assert len(sleeps) == 1
    assert 0.5 <= sleeps[0] < 1.05


def test_complete_falls_back_to_exponential_when_no_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the rate-limit message lacks a parsable retry-after, fall back
    to exponential backoff (1, 2, 4, ...)."""
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)

    import qed_swe_bench.runner.llm.litellm_client as mod
    monkeypatch.setattr(mod.time, "sleep", fake_sleep)

    call_count = {"n": 0}

    def fake_completion(**kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise type("RateLimitError", (Exception,), {})("rate_limit_exceeded; no time hint")
        return _stub_response(content="ok")

    monkeypatch.setattr(mod.litellm, "completion", fake_completion)
    monkeypatch.setenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", "5")

    client = mod.LiteLLMClient(model="openai/gpt-5.5")
    client.complete(messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=10)

    # Two retries → two sleeps. First should be ~2^0=1 + jitter, second ~2^1=2 + jitter.
    assert len(sleeps) == 2
    assert 1.0 <= sleeps[0] < 1.55
    assert 2.0 <= sleeps[1] < 2.55


def test_complete_does_not_retry_non_rate_limit_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-rate-limit error should propagate immediately, no sleep, no retry."""
    sleeps: list[float] = []
    monkeypatch.setattr("qed_swe_bench.runner.llm.litellm_client.time.sleep", sleeps.append)

    call_count = {"n": 0}

    def fake_completion(**kwargs):
        call_count["n"] += 1
        raise RuntimeError("some 500 error")

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", "5")

    from qed_swe_bench.runner.llm.litellm_client import LiteLLMClient
    client = LiteLLMClient(model="openai/gpt-5.5")
    with pytest.raises(RuntimeError, match="some 500 error"):
        client.complete(messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=10)

    assert call_count["n"] == 1
    assert sleeps == []


def test_complete_exhausts_retries_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every attempt rate-limits, the last exception propagates after
    max_retries+1 total attempts."""
    sleeps: list[float] = []
    monkeypatch.setattr("qed_swe_bench.runner.llm.litellm_client.time.sleep", sleeps.append)

    call_count = {"n": 0}

    def fake_completion(**kwargs):
        call_count["n"] += 1
        raise type("RateLimitError", (Exception,), {})(
            "rate_limit_exceeded; try again in 0.1s"
        )

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setenv("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES", "2")

    from qed_swe_bench.runner.llm.litellm_client import LiteLLMClient
    client = LiteLLMClient(model="openai/gpt-5.5")
    with pytest.raises(Exception, match="rate_limit_exceeded"):
        client.complete(messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=10)

    # max_retries=2 → 1 initial + 2 retries = 3 attempts, 2 sleeps.
    assert call_count["n"] == 3
    assert len(sleeps) == 2


# ---------------- safety_identifier forwarding ----------------


def _capture_litellm_call(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace litellm.completion with a no-op that records its kwargs.
    Returns the list it appends to."""
    import litellm

    captured: list[dict] = []

    def fake_completion(**kwargs):
        captured.append(kwargs)
        return _stub_response(content="ok", prompt_tokens=10, completion_tokens=2)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    return captured


def test_safety_identifier_forwarded_for_openai_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI documents `safety_identifier` as the per-user lever that scopes
    cyber_policy revocations. Must reach litellm.completion verbatim."""
    captured = _capture_litellm_call(monkeypatch)
    client = LiteLLMClient(model="openai/gpt-5.5")
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=10,
        safety_identifier="qed_swe_bench-run-abc123",
    )
    assert captured[0].get("safety_identifier") == "qed_swe_bench-run-abc123"


def test_safety_identifier_dropped_for_non_openai_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Other providers don't recognize the kwarg; litellm may 400 on it. Gate
    forwarding behind an openai/* check."""
    captured = _capture_litellm_call(monkeypatch)
    client = LiteLLMClient(model="gemini/gemini-2.5-flash")
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=10,
        safety_identifier="qed_swe_bench-run-abc123",
    )
    assert "safety_identifier" not in captured[0]


def test_safety_identifier_omitted_when_caller_passes_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No identifier supplied → don't add the kwarg at all (avoid sending
    empty strings or None to the provider)."""
    captured = _capture_litellm_call(monkeypatch)
    client = LiteLLMClient(model="openai/gpt-5.5")
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=10,
        safety_identifier=None,
    )
    assert "safety_identifier" not in captured[0]


# ---------------- seed-param gating ----------------


def test_seed_forwarded_for_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI accepts seed; forward it for what little reproducibility we get."""
    captured = _capture_litellm_call(monkeypatch)
    client = LiteLLMClient(model="openai/gpt-5.5")
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=10,
        seed=42,
    )
    assert captured[0].get("seed") == 42


def test_seed_dropped_for_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini's API rejects `seed` — LiteLLM 400s with
    UnsupportedParamsError unless we drop it. Per-call gate avoids
    global drop_params side-effects."""
    captured = _capture_litellm_call(monkeypatch)
    client = LiteLLMClient(model="gemini/gemini-3.1-pro-preview")
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=10,
        seed=42,
    )
    assert "seed" not in captured[0]


def test_seed_dropped_for_zai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Z.ai also rejects `seed`. Same per-call gate as Gemini."""
    captured = _capture_litellm_call(monkeypatch)
    client = LiteLLMClient(model="zai/glm-5.1")
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=10,
        seed=42,
    )
    assert "seed" not in captured[0]


def test_temperature_dropped_for_moonshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Moonshot Kimi K2.x rejects `temperature=0` (only 1 is allowed); same
    quirk as OpenAI's gpt-5 family. Drop the default rather than 400."""
    captured = _capture_litellm_call(monkeypatch)
    client = LiteLLMClient(model="moonshot/kimi-k2.6")
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=10,
    )
    assert "temperature" not in captured[0]


def test_temperature_dropped_for_gemini3_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini 3.x with `temperature < 1.0` triggers infinite loops and empty
    output (LiteLLM emits a warning + we see HTTP-level hangs in production)."""
    captured = _capture_litellm_call(monkeypatch)
    client = LiteLLMClient(model="gemini/gemini-3.1-pro-preview")
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=10,
    )
    assert "temperature" not in captured[0]


def test_temperature_dropped_for_gemini3_via_other_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same `gemini-3` family quirk applies regardless of routing prefix
    (OpenRouter, LiteLLM proxy, Vertex AI, gateway double-prefix)."""
    for model_id in (
        "openrouter/google/gemini-3.1-pro-preview",
        "litellm_proxy/gemini-3.1-pro-preview",
        "vertex_ai/gemini-3-flash",
        "openai/gemini/gemini-3.1-pro-preview",
    ):
        captured = _capture_litellm_call(monkeypatch)
        client = LiteLLMClient(model=model_id)
        client.complete(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            max_tokens=10,
        )
        assert "temperature" not in captured[0], f"failed for {model_id}"


def test_temperature_kept_for_gemini2(monkeypatch: pytest.MonkeyPatch) -> None:
    """The temperature-quirk is documented for Gemini 3; older Gemini
    families accept temperature=0 and shouldn't be affected."""
    captured = _capture_litellm_call(monkeypatch)
    client = LiteLLMClient(model="gemini/gemini-2.5-pro")
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=10,
    )
    assert captured[0]["temperature"] == 0
