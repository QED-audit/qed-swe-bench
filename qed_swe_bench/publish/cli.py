"""CLI orchestrator for `qed_swe_bench publish`.

The function `publish_command` is registered as a top-level Typer
command in `qed_swe_bench/cli.py` (one command, no sub-verbs).

Pipeline: select_canonical → audit gate → bundle → write card +
manifests → optional push.

Privacy: the only mechanism is `--exclude-model <id>` (repeatable).
The flag value never enters any uploaded artifact — it appears only
in the local audit-trail manifest and the dry-run console output.
See docs/decisions.md D-15.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from qed_swe_bench.config import Config
from qed_swe_bench.publish import audit_gate, bundle, card, hf
from qed_swe_bench.publish.revision import compute_revision
from qed_swe_bench.publish.selection import (
    CellRecord,
    find_orphan_run_dirs,
    select_canonical,
)

console = Console()


def _print_dry_run_summary(
    *,
    db_path: Path,
    benchmark_id: str,
    revision: str,
    repo_id: str,
    license_id: str | None,
    excluded_models: list[str],
    cells: list[CellRecord],
    gate_result: audit_gate.GateResult,
    push_will_be_public: bool,
    push_will_run: bool,
) -> None:
    console.print()
    console.print("[bold]qed-swe-bench publish — DRY RUN[/]" if not push_will_run
                  else "[bold]qed-swe-bench publish — PRE-PUSH SUMMARY[/]")
    console.print(f"  Source DB:       {db_path}")
    console.print(f"  Benchmark:       {benchmark_id}")
    console.print(f"  Revision:        {revision}")
    visibility = "PUBLIC" if push_will_be_public else "private"
    console.print(
        f"  Target repo:     {repo_id}  (license: {license_id or '<unset>'}, "
        f"visibility: {visibility})"
    )
    if excluded_models:
        console.print(
            f"  Excluded models: {', '.join(excluded_models)}   (--exclude-model)"
        )
    else:
        console.print(
            "  Excluded models: [yellow](none — all models in the DB will be "
            "published)[/]"
        )
    console.print()

    # Per-model breakdown
    by_model: dict[str, dict[str, int]] = {}
    for cell in cells:
        slot = by_model.setdefault(
            cell.model, {"cells": 0, "succeeded": 0, "model_failed": 0}
        )
        slot["cells"] += 1
        if cell.status == "succeeded":
            slot["succeeded"] += 1
        elif cell.status == "model_failed":
            slot["model_failed"] += 1

    console.print("[bold]Will publish:[/]")
    if not by_model:
        console.print("  [red](no cells — selection is empty)[/]")
    else:
        # Right-pad model names so cell counts align visually
        max_model_len = max(len(m) for m in by_model)
        for model, counts in sorted(by_model.items()):
            console.print(
                f"  {model.ljust(max_model_len)}  "
                f"{counts['cells']:>4} cells   "
                f"({counts['succeeded']} succeeded, "
                f"{counts['model_failed']} model_failed)"
            )
        console.print(f"  {'-' * max_model_len}  ----")
        console.print(
            f"  {'Total'.ljust(max_model_len)}  {len(cells):>4} cells"
        )
    console.print()

    h = gate_result.counts.get("high", 0)
    m = gate_result.counts.get("medium", 0)
    i = gate_result.counts.get("info", 0)
    console.print(f"[bold]Audit:[/]  {h} HIGH, {m} MEDIUM, {i} INFO")
    console.print(
        "        [dim](audit findings flag runs for manual review; "
        "they are not definitive — false positives are expected. "
        "Inspect audit.json before deciding.)[/]"
    )
    if gate_result.high_run_ids:
        console.print(
            f"        HIGH run_ids: {', '.join(gate_result.high_run_ids[:6])}"
            + ("…" if len(gate_result.high_run_ids) > 6 else "")
        )
    if gate_result.medium_run_ids:
        console.print(
            f"        MEDIUM run_ids: {', '.join(gate_result.medium_run_ids[:6])}"
            + ("…" if len(gate_result.medium_run_ids) > 6 else "")
        )
    console.print()


def publish_command(
    benchmark_id: Annotated[str, typer.Option(
        "--benchmark-id", "-b",
        help="Benchmark to publish (column in the runs table). Default: v8.",
    )] = "v8",
    exclude_model: Annotated[list[str] | None, typer.Option(
        "--exclude-model",
        help="Drop all rows whose model id matches (exact string match). "
             "Repeatable. The only mechanism for keeping NDA-private models "
             "out of the publish path; private names never enter any "
             "committed file.",
    )] = None,
    repo_id: Annotated[str, typer.Option(
        "--repo-id",
        help="HF dataset repo id. Default: qed_swe_bench/v8.",
    )] = "qed_swe_bench/v8",
    revision: Annotated[str | None, typer.Option(
        "--revision",
        help="Override the auto-derived revision tag "
             "(default: v8-<sha7(v8.yaml)>-pt<sha7(prompt-template)>).",
    )] = None,
    license_id: Annotated[str | None, typer.Option(
        "--license",
        help="SPDX license id (e.g. cc-by-4.0). Required for public push.",
    )] = None,
    push: Annotated[bool, typer.Option(
        "--push",
        help="Actually upload to HuggingFace. Without this flag, only the "
             "local bundle under dist/ is built.",
    )] = False,
    allow_public: Annotated[bool, typer.Option(
        "--allow-public",
        help="Acknowledge a public push. Without this, --push always uses "
             "private visibility regardless of --private.",
    )] = False,
    private: Annotated[bool, typer.Option(
        "--private",
        help="Force private visibility on push. Default if --allow-public "
             "is not set.",
    )] = False,
    allow_high_audit: Annotated[bool, typer.Option(
        "--allow-high-audit",
        help="Permit publishing even when the audit gate finds HIGH "
             "severity issues. The audit is a manual-review aid, not a "
             "definitive judgment — false positives are common. Use "
             "after triaging audit.json and confirming each HIGH "
             "finding is benign or expected.",
    )] = False,
    force_retag: Annotated[bool, typer.Option(
        "--force-retag",
        help="Delete and re-create the revision tag if it already exists "
             "on the remote. Default: refuse the second push.",
    )] = False,
    dry_run: Annotated[bool, typer.Option(
        "--dry-run",
        help="Build the bundle locally (skipping all network calls) and "
             "print the publish plan. Implies the same selection + audit + "
             "bundling work as a real push.",
    )] = False,
    repo_root: Annotated[Path | None, typer.Option(
        "--repo-root",
        help="Override the repo root used for revision computation and the "
             "dist/ output dir. Defaults to the current working directory.",
    )] = None,
) -> None:
    """Bundle canonical runs for a benchmark and publish them to HuggingFace.

    Two methodology rules drive this command (see docs/decisions.md D-15):

      1. HuggingFace is the academic record. Reward-hack cells, failures,
         and full transcripts all ship. Display-time filtering is the
         website's concern — this command does not read
         `website/data/exclusions.json`.

      2. Private model names never enter a committed file. Pass
         `--exclude-model <id>` once per private model on each invocation.
         The names appear only in the local audit-trail manifest under
         `dist/` and in the console output — never in anything uploaded.

    A typical public push:

        qed_swe_bench publish --push --allow-public --license cc-by-4.0 \\
            --exclude-model anthropic/some-private-preview
    """
    excluded = sorted(set(exclude_model or []))
    cfg = Config.from_env()
    db_path = cfg.db_path
    runs_root = cfg.runs_dir
    root = (repo_root or Path.cwd()).resolve()

    if not db_path.exists():
        console.print(
            f"[red]DB not found: {db_path}. Run `qed_swe_bench import runs/` "
            f"first.[/]"
        )
        raise typer.Exit(code=2)

    rev = revision or compute_revision(root)

    # 1. Selection
    cells = select_canonical(
        db_path, benchmark_id, exclude_models=excluded
    )
    if not cells:
        console.print(
            f"[red]No publishable cells for benchmark_id={benchmark_id!r} "
            f"after applying {len(excluded)} exclusion(s). Refusing to "
            f"build an empty dataset.[/]"
        )
        raise typer.Exit(code=2)

    # Warn (don't fail) if there are run_dirs on disk without DB rows —
    # the publish path uses the DB exclusively, so orphans get silently
    # left out. Operators usually want to know.
    orphans = find_orphan_run_dirs(runs_root, benchmark_id, cells)
    if orphans:
        console.print(
            f"[yellow]warning: {len(orphans)} run dir(s) under "
            f"runs/{benchmark_id}/ are not in the DB and will not ship. "
            f"Run `qed_swe_bench import runs/` to backfill.[/]"
        )

    # 2. Audit gate (always run; HIGH only blocks the push)
    gate_result = audit_gate.gate(c.run_dir for c in cells)

    # Compute push intent for the dry-run summary
    push_will_be_public = push and allow_public and not private
    push_will_run = push and not dry_run

    _print_dry_run_summary(
        db_path=db_path,
        benchmark_id=benchmark_id,
        revision=rev,
        repo_id=repo_id,
        license_id=license_id,
        excluded_models=excluded,
        cells=cells,
        gate_result=gate_result,
        push_will_be_public=push_will_be_public,
        push_will_run=push_will_run,
    )

    # HIGH findings block the *push* (not the bundle). Dry-run always
    # gets to build so the operator can inspect dist/ before deciding
    # how to triage. We re-check this gate after the bundle, just before
    # the push step.

    # 3. Bundle (always — dry-run still writes locally so the operator
    # can inspect `dist/` before any push)
    dist_root = root / "dist" / repo_id.replace("/", "__") / rev
    console.print(f"  Bundling → {dist_root.relative_to(root)}/")
    stats = bundle.build(cells, dist_root)
    console.print(
        f"  ✓ {stats.n_cells} cells, "
        f"{stats.sidecar_bytes / (1024 * 1024):.1f} MB sidecars"
    )

    # 4. Card + manifests + audit
    card.write_card(
        dist_root,
        repo_id=repo_id,
        revision=rev,
        cells=cells,
        stats=stats,
        gate=gate_result,
        license_id=license_id,
    )
    # Local-only manifest (records excluded_models for operator audit trail)
    card.write_manifest(
        dist_root,
        repo_id=repo_id,
        revision=rev,
        cells=cells,
        stats=stats,
        gate=gate_result,
        license_id=license_id,
        excluded_models=excluded,
        for_upload=False,
        filename="manifest.local.json",
    )
    # Upload manifest (no excluded_models field; this is what ships)
    card.write_manifest(
        dist_root,
        repo_id=repo_id,
        revision=rev,
        cells=cells,
        stats=stats,
        gate=gate_result,
        license_id=license_id,
        excluded_models=excluded,
        for_upload=True,
        filename="manifest.json",
    )
    card.write_audit_json(dist_root, gate=gate_result)

    if not push or dry_run:
        console.print()
        if gate_result.has_high:
            console.print(
                f"[yellow]Note: {gate_result.counts['high']} HIGH audit "
                f"finding(s) would block a real push. Triage audit.json, "
                f"then re-run with --push (add --allow-high-audit to "
                f"override after triage).[/]"
            )
        console.print(
            "[bold]Local-only build complete.[/] To push: re-run with "
            "--push (and --allow-public --license ... for public)."
        )
        return

    # 5. Push — re-gate on HIGH findings before we cross the network
    if gate_result.has_high and not allow_high_audit:
        console.print(
            f"[red]Refusing to push: {gate_result.counts['high']} HIGH "
            f"audit finding(s). Triage audit.json, then re-run with "
            f"--allow-high-audit to override.[/]"
        )
        raise typer.Exit(code=3)

    if not allow_public and not private:
        # User passed --push without --allow-public or --private; force
        # private so they can't accidentally publish a public dataset.
        console.print(
            "[yellow]No --allow-public flag; pushing as PRIVATE.[/]"
        )
        will_be_private = True
    elif allow_public and private:
        console.print(
            "[red]--private and --allow-public are mutually exclusive.[/]"
        )
        raise typer.Exit(code=2)
    else:
        will_be_private = private

    if not will_be_private and not license_id:
        console.print(
            "[red]Refusing public push without --license <spdx-id>. "
            "Use --license cc-by-4.0 (or another SPDX id).[/]"
        )
        raise typer.Exit(code=2)

    # Idempotence: if the revision tag already exists on the remote and
    # --force-retag is not set, exit 0 with a "already published" note.
    if not force_retag and hf.revision_tag_exists(repo_id, rev):
        console.print(
            f"[yellow]Revision tag {rev!r} already exists on "
            f"{repo_id}; nothing to publish. Pass --force-retag to "
            f"overwrite.[/]"
        )
        raise typer.Exit(code=0)

    # The directory we upload is dist_root, but it contains a local-only
    # manifest.local.json the operator's audit trail wants to keep on
    # disk and never push. Move it sideways before upload, restore after.
    local_only = dist_root / "manifest.local.json"
    sidecar = dist_root.parent / f".manifest.local.{rev}.json"
    if local_only.exists():
        local_only.rename(sidecar)

    try:
        result = hf.push(
            dist_root,
            repo_id,
            revision=rev,
            private=will_be_private,
            force_retag=force_retag,
        )
    except hf.HFError as exc:
        console.print(f"[red]Push failed: {exc}[/]")
        raise typer.Exit(code=1) from exc
    finally:
        if sidecar.exists():
            sidecar.rename(local_only)

    visibility = "private" if result.private else "PUBLIC"
    console.print()
    console.print(
        f"[green]✓ Published {repo_id} @ {result.revision} ({visibility})[/]"
    )
    if result.upload_commit_url:
        console.print(f"  commit: {result.upload_commit_url}")
    if result.tag_created:
        console.print(f"  tag created: {result.revision}")
    else:
        console.print(
            f"  tag {result.revision} already existed; commit landed on "
            "main but the tag still points at the prior commit"
        )


def main() -> int:
    """Allow running this module directly: `python -m qed_swe_bench.publish.cli`."""
    typer.run(publish_command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
