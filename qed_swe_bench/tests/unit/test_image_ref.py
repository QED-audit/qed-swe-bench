"""Image-ref classification + resolution. Subprocess calls are stubbed out."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from qed_swe_bench.runner.image_ref import (
    ImageRefError,
    ResolvedImage,
    is_buildable_path,
    is_digest,
    is_local,
    resolve,
)

# ---------- classification ----------

def test_is_digest() -> None:
    assert is_digest("ghcr.io/x/y@sha256:" + "0" * 64)
    assert is_digest("foo@sha256:" + "a" * 64)
    assert not is_digest("ghcr.io/x/y:v1")
    assert not is_digest("local/x:latest")


def test_is_local() -> None:
    assert is_local("local/sample-stack-bof:latest")
    assert is_local("sample-stack-bof:latest")
    assert is_local("foo")
    assert not is_local("ghcr.io/forallsecure/v8/cve-2024-4761:v1")
    assert not is_local("registry.example.com:5000/foo:tag")
    assert not is_local("ghcr.io/x/y@sha256:" + "0" * 64)


def test_is_buildable_path() -> None:
    assert is_buildable_path("./bench-v8/bugs/sample-stack-bof/")
    assert is_buildable_path("../foo/")
    assert is_buildable_path("/abs/path/")
    assert is_buildable_path("./foo/Dockerfile")
    assert not is_buildable_path("local/sample:latest")
    assert not is_buildable_path("ghcr.io/x:tag")


# ---------- resolve ----------

class _FakeProc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_inspect_stdout(image_id: str = "sha256:" + "a" * 64) -> str:
    return json.dumps([{"Id": image_id, "RepoTags": ["foo:bar"]}])


@pytest.fixture
def fake_subprocess(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def stub_run(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProc:
        calls.append(list(cmd))
        # pull → success
        if "pull" in cmd:
            return _FakeProc(0)
        # inspect → return canned info
        if "inspect" in cmd:
            return _FakeProc(0, stdout=_make_inspect_stdout())
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "run", stub_run)
    return calls


def test_resolve_rejects_buildable_path() -> None:
    with pytest.raises(ImageRefError, match="filesystem path"):
        resolve("./bench-v8/bugs/sample-stack-bof/")


def test_resolve_local_no_pull(fake_subprocess) -> None:
    res = resolve("local/sample-stack-bof:latest")
    assert isinstance(res, ResolvedImage)
    assert res.image_ref == "local/sample-stack-bof:latest"
    assert res.image_digest == "sha256:" + "a" * 64
    assert res.pulled is False
    # Inspect, no pull.
    assert any("inspect" in c for c in fake_subprocess)
    assert not any("pull" in c for c in fake_subprocess)


def test_resolve_registry_tag_present_locally_no_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the image is already cached locally for the requested tag, don't
    re-pull. Avoids wedging on expired-token or offline runs (CLAUDE.md
    notes ECR tokens lapse every ~12h)."""
    def stub_run(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProc:
        if "pull" in cmd:
            pytest.fail("should not pull when image is already cached locally")
        if "inspect" in cmd:
            return _FakeProc(0, stdout=_make_inspect_stdout())
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "run", stub_run)
    res = resolve("ghcr.io/forallsecure/v8/cve-2024-4761:v1")
    assert res.pulled is False
    assert res.image_digest == "sha256:" + "a" * 64


def test_resolve_registry_tag_absent_locally_triggers_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    """First time we see a registry tag (no local cache), pull, then inspect."""
    seen: list[str] = []

    def stub_run(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProc:
        seen.append(cmd[1])  # 'image' or 'pull'
        if cmd[:2] == ["docker", "image"] and "inspect" in cmd:
            if seen.count("image") == 1:
                return _FakeProc(1, stderr="No such image")
            return _FakeProc(0, stdout=_make_inspect_stdout())
        if "pull" in cmd:
            return _FakeProc(0)
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "run", stub_run)
    res = resolve("ghcr.io/forallsecure/v8/cve-2024-4761:v1")
    assert res.pulled is True
    assert res.image_digest == "sha256:" + "a" * 64


def test_resolve_registry_tag_force_pull_env_overrides_local_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`QED_SWE_BENCH_FORCE_PULL=1` opts back into always-pull behavior even
    when the image is present locally — for the rare case of verifying a
    tag is current against the registry."""
    monkeypatch.setenv("QED_SWE_BENCH_FORCE_PULL", "1")
    pull_seen = []

    def stub_run(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProc:
        if "pull" in cmd:
            pull_seen.append(cmd)
            return _FakeProc(0)
        if "inspect" in cmd:
            return _FakeProc(0, stdout=_make_inspect_stdout())
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "run", stub_run)
    res = resolve("ghcr.io/forallsecure/v8/cve-2024-4761:v1")
    assert res.pulled is True
    assert pull_seen, "QED_SWE_BENCH_FORCE_PULL should have forced a pull"


def test_resolve_digest_present_locally_no_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    """For a digest-pinned ref like ghcr.io/x/y@sha256:<MD>, the bare manifest
    digest from the ref is NOT the same SHA as the local image Id (a real
    registry-manifest digest hashes the manifest blob; the local Id hashes
    the image config). resolve() must inspect the loaded image and return
    its Id — `docker run sha256:<local-Id>` works, `docker run sha256:<MD>`
    does not."""
    manifest_digest = "sha256:" + "b" * 64    # appears in the ref
    local_id = "sha256:" + "0" * 63 + "1"     # different — what inspect returns
    ref = f"ghcr.io/x/y@{manifest_digest}"

    inspect_calls = []

    def stub_run(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProc:
        if "pull" in cmd:
            pytest.fail("should not pull when image is already present")
        if "inspect" in cmd:
            inspect_calls.append(cmd)
            return _FakeProc(0, stdout=_make_inspect_stdout(image_id=local_id))
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "run", stub_run)
    res = resolve(ref)
    assert res.pulled is False
    # Critical: image_digest is the LOCAL Id, not the registry manifest digest.
    assert res.image_digest == local_id
    assert res.image_digest != manifest_digest
    # Inspect is called to discover the local Id.
    assert any("inspect" in c for c in inspect_calls)


def test_resolve_digest_absent_locally_triggers_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same registry-manifest-digest vs local-Id distinction; this time the
    image isn't cached so a pull happens before the inspect."""
    manifest_digest = "sha256:" + "c" * 64
    local_id = "sha256:" + "f" * 63 + "0"
    ref = f"ghcr.io/x/y@{manifest_digest}"
    seen: list[str] = []

    def stub_run(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProc:
        seen.append(cmd[1])  # 'image' or 'pull'
        # First inspect (presence check) → fail. Second (after pull) → succeed.
        if cmd[:2] == ["docker", "image"] and "inspect" in cmd:
            if seen.count("image") == 1:
                return _FakeProc(1, stderr="No such image")
            return _FakeProc(0, stdout=_make_inspect_stdout(image_id=local_id))
        if "pull" in cmd:
            return _FakeProc(0)
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "run", stub_run)
    res = resolve(ref)
    assert res.pulled is True
    assert res.image_digest == local_id
    assert res.image_digest != manifest_digest


def test_resolve_pull_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the image isn't present locally and pull fails (network /
    registry / auth), surface the failure as ImageRefError."""
    def stub_run(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProc:
        if "pull" in cmd:
            return _FakeProc(1, stderr="manifest unknown")
        # Inspect fails → triggers the pull path.
        return _FakeProc(1, stderr="No such image")

    monkeypatch.setattr(subprocess, "run", stub_run)
    with pytest.raises(ImageRefError, match="manifest unknown"):
        resolve("ghcr.io/x/y:missing")


def test_resolve_inspect_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def stub_run(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProc:
        return _FakeProc(1, stderr="No such image")

    monkeypatch.setattr(subprocess, "run", stub_run)
    with pytest.raises(ImageRefError, match="not loaded"):
        resolve("local/never-built:latest")


def test_resolve_inspect_returns_no_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def stub_run(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeProc:
        return _FakeProc(0, stdout=json.dumps([{"RepoTags": ["x"]}]))  # no Id

    monkeypatch.setattr(subprocess, "run", stub_run)
    with pytest.raises(ImageRefError, match="missing valid Id sha256"):
        resolve("local/foo:latest")
