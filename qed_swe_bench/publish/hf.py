"""HuggingFace API adapter for the publish pipeline.

Pure adapter: no business logic, no policy. The CLI handles
permission gates (`--allow-public`, `--license`); this module just
wraps `huggingface_hub` calls and surfaces useful errors.

`huggingface_hub` is imported lazily — it lives in the `[publish]`
optional extra so EC2 runners don't pay for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class HFError(Exception):
    """Raised when an HF interaction fails for an actionable reason."""


@dataclass(frozen=True)
class PushResult:
    repo_id: str
    revision: str
    private: bool
    upload_commit_url: str | None
    tag_created: bool


def _import_hf():
    try:
        from huggingface_hub import HfApi, create_repo  # type: ignore[import-not-found]
        from huggingface_hub.errors import HfHubHTTPError  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "publish push requires the [publish] extras. Install with: "
            "pip install -e '.[publish]' (adds huggingface_hub + pyarrow + "
            "zstandard)."
        ) from exc
    return HfApi, create_repo, HfHubHTTPError


def revision_tag_exists(repo_id: str, revision: str) -> bool:
    """Return True if the revision tag already exists on the remote.

    Used as the publish CLI's idempotence check: if the tag is there,
    we've already published this methodology snapshot and should
    refuse the second push (unless --force-retag).
    """
    HfApi, _create_repo, HfHubHTTPError = _import_hf()
    api = HfApi()
    try:
        refs = api.list_repo_refs(repo_id=repo_id, repo_type="dataset")
    except HfHubHTTPError as exc:
        # 404 → repo doesn't exist yet → no tag possible. Anything else
        # is a real error (auth, network, etc.) that the caller should
        # see verbatim.
        if exc.response is not None and exc.response.status_code == 404:
            return False
        raise
    for tag in refs.tags or []:
        if tag.name == revision:
            return True
    return False


def push(
    bundle_dir: Path,
    repo_id: str,
    *,
    revision: str,
    private: bool,
    force_retag: bool = False,
    commit_message: str | None = None,
) -> PushResult:
    """Create the dataset repo if missing, upload the bundle, tag the revision.

    Single atomic upload (one commit on `main`); no PRs, no branches.
    The tag is the immutability layer — future readers cite the
    revision by name and get a stable snapshot.
    """
    HfApi, create_repo, HfHubHTTPError = _import_hf()
    api = HfApi()

    # 1. ensure the dataset repo exists
    try:
        create_repo(
            repo_id, repo_type="dataset", private=private, exist_ok=True
        )
    except HfHubHTTPError as exc:
        owner = repo_id.split("/", 1)[0]
        name = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id
        raise HFError(
            f"create_repo failed for {repo_id}: {exc}. "
            f"Token may lack 'Create repos' permission for org `{owner}`. "
            f"Workaround: pre-create at https://huggingface.co/new-dataset "
            f"(owner={owner}, name={name}, private={private}), then re-run."
        ) from exc

    # 2. upload everything in bundle_dir to repo root in a single commit
    try:
        commit_info = api.upload_folder(
            folder_path=str(bundle_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message
            or f"Publish revision {revision}",
            create_pr=False,
        )
    except HfHubHTTPError as exc:
        raise HFError(f"upload_folder failed: {exc}") from exc

    # 3. tag the commit. If the tag already exists and we're not
    # force-retagging, the create_tag call would 409; we check first
    # so we can give a clearer message.
    tag_existed = revision_tag_exists(repo_id, revision)
    tag_created = False
    if tag_existed and not force_retag:
        # Upload landed (a new main commit) but the tag still points
        # at the prior commit. That's the conservative behavior — the
        # revision tag is *immutable* by default.
        pass
    else:
        if tag_existed and force_retag:
            try:
                api.delete_tag(
                    repo_id=repo_id, repo_type="dataset", tag=revision
                )
            except HfHubHTTPError as exc:
                raise HFError(
                    f"delete_tag failed (force-retag): {exc}"
                ) from exc
        try:
            api.create_tag(
                repo_id=repo_id,
                repo_type="dataset",
                tag=revision,
                tag_message=f"Methodology revision {revision}",
                exist_ok=False,
            )
            tag_created = True
        except HfHubHTTPError as exc:
            raise HFError(f"create_tag failed: {exc}") from exc

    return PushResult(
        repo_id=repo_id,
        revision=revision,
        private=private,
        upload_commit_url=getattr(commit_info, "commit_url", None),
        tag_created=tag_created,
    )


def delete_repo(repo_id: str) -> None:
    """Delete a dataset repo. Used by smoke-test teardown only.

    Raises HFError on any failure — this is destructive, so the caller
    must see the error.
    """
    HfApi, _create_repo, HfHubHTTPError = _import_hf()
    api = HfApi()
    try:
        api.delete_repo(repo_id=repo_id, repo_type="dataset")
    except HfHubHTTPError as exc:
        raise HFError(f"delete_repo failed for {repo_id}: {exc}") from exc
