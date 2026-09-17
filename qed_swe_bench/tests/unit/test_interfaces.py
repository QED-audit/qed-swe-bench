"""Interface registry sanity."""

from __future__ import annotations

from qed_swe_bench.contract import interfaces


def test_registry_contains_expected_interfaces() -> None:
    expected = {
        "rl.mcp.base.v1",
        "rl.mcp.v8_task.v1",
        "rl.mcp.binary_task.v1",  # reserved for bountybench / CTF-shaped envs
        "rl.mcp.lifecycle.v1",
    }
    assert set(interfaces.REGISTRY) == expected


def test_lookup_v8_task() -> None:
    iface = interfaces.lookup("rl.mcp.v8_task.v1")
    assert iface is not None
    assert iface.grader_tool == "grade"
    assert "ace" in iface.capability_flags
    assert iface.submission_kind == "task"


def test_generic_binary_task_is_reserved() -> None:
    """Reserved for bountybench; capability_flags intentionally empty until filled."""
    iface = interfaces.lookup("rl.mcp.binary_task.v1")
    assert iface is not None
    assert iface.capability_flags == ()
    assert "Reserved" in iface.notes


def test_lookup_unknown_returns_none() -> None:
    assert interfaces.lookup("rl.mcp.totally-fake.v99") is None
    assert not interfaces.is_known("rl.mcp.totally-fake.v99")


def test_v8_task_capability_set_matches_extractor() -> None:
    """V8's capability_flags must agree with capabilities.CAPABILITY_FLAGS."""
    from qed_swe_bench.runner.capabilities import CAPABILITY_FLAGS

    iface = interfaces.lookup("rl.mcp.v8_task.v1")
    assert set(iface.capability_flags) == set(CAPABILITY_FLAGS)


def test_each_non_base_inherits_from_base() -> None:
    """Sanity: every concrete interface points at a registered parent."""
    for name, iface in interfaces.REGISTRY.items():
        if iface.base is None:
            continue
        assert iface.base in interfaces.REGISTRY, (
            f"{name} declares base={iface.base} which is not registered"
        )


def test_all_names_sorted() -> None:
    assert interfaces.all_names() == sorted(interfaces.REGISTRY)
