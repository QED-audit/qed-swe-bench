"""MockClient replays canned tool-call sequence."""

from __future__ import annotations

from qed_swe_bench.runner.llm.mock import DEFAULT_SEQUENCE, MockClient


def _call(client: MockClient):
    return client.complete(
        messages=[],
        tools=[],
        max_tokens=4096,
    )


def test_default_sequence_replays_in_order() -> None:
    client = MockClient(model="mock/test")
    seen = []
    for _ in range(len(DEFAULT_SEQUENCE)):
        resp = _call(client)
        assert len(resp.tool_calls) == 1
        seen.append(resp.tool_calls[0].name)
    assert seen == [step["name"] for step in DEFAULT_SEQUENCE]


def test_after_sequence_returns_text_no_calls() -> None:
    client = MockClient(model="mock/test", sequence=[])
    resp = _call(client)
    assert resp.tool_calls == ()
    assert resp.text == "mock done"
    assert resp.stop_reason == "end_turn"


def test_custom_sequence() -> None:
    custom = [{"name": "grade", "arguments": {"path": "/x.js"}}]
    client = MockClient(model="mock/test", sequence=custom)
    resp = _call(client)
    assert resp.tool_calls[0].name == "grade"
    assert resp.tool_calls[0].arguments == {"path": "/x.js"}
    # Next call: no more steps.
    resp2 = _call(client)
    assert resp2.tool_calls == ()


def test_tool_call_ids_increment() -> None:
    client = MockClient(model="m", sequence=[
        {"name": "a", "arguments": {}},
        {"name": "b", "arguments": {}},
    ])
    r1 = _call(client)
    r2 = _call(client)
    assert r1.tool_calls[0].id != r2.tool_calls[0].id


def test_route_is_mock() -> None:
    assert MockClient(model="m").route == "mock"
