"""qed_swe_bench CLI entry point.

Commands:
  doctor          Verify env / docker / deps / provider keys
  benchmark       Run a multi-model benchmark from a YAML config
  summary         Per-benchmark spend / status table (with --watch)
  smoke           Per-model tool-call fidelity probe
  aggregate       Emit results table from the DB
  publish         Bundle canonical runs and push to HuggingFace
  import-eval     Ingest external eval/ trees as historical rows
  register-dir    Catalog all envs under a directory of manifests
  validate-image  Run the validator suite against a single image
  list-interfaces Print the registered MCP interface contracts
  api             Serve the FastAPI read endpoints
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from qed_swe_bench.config import Config
from qed_swe_bench.db.schema import connect, init_db
from qed_swe_bench.publish.cli import publish_command

app = typer.Typer(
    name="qed_swe_bench",
    help="Multi-model V8 coding-benchmark pipeline.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

# `qed_swe_bench publish` — bundle canonical runs into HuggingFace shape
# and (optionally) push. Implemented in qed_swe_bench/publish/cli.py;
# registered here as a top-level command.
app.command(name="publish")(publish_command)


@app.callback()
def _root(
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show INFO-level progress (image pulls, per-tuple results, "
             "benchmark-done summary). Default surfaces only warnings/errors.",
    ),
    debug: bool = typer.Option(
        False, "--debug",
        help="Show DEBUG-level detail (everything --verbose shows, plus "
             "internal state). Implies --verbose.",
    ),
) -> None:
    """Top-level CLI options. Runs before any subcommand."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Check env vars, docker, disk, and bench-v8 paths.

    Exits 0 if all checks pass, 1 otherwise. Prints a human-readable table.
    """
    cfg = Config.from_env()
    table = Table(title="qed_swe_bench doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    failures: list[str] = []

    # --- DB path ---
    try:
        db_path = init_db()
        with connect(db_path) as con:
            row = con.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
        version = row["value"] if row else "?"
        table.add_row("database", "[green]ok[/]", f"{db_path} (schema v{version})")
    except Exception as exc:
        failures.append(f"database: {exc}")
        table.add_row("database", "[red]fail[/]", str(exc))

    # --- docker ---
    docker_path = shutil.which("docker")
    if docker_path:
        try:
            result = subprocess.run(
                [docker_path, "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                table.add_row(
                    "docker", "[green]ok[/]", f"daemon up, version {result.stdout.strip()}"
                )
            else:
                failures.append("docker daemon not responding")
                table.add_row(
                    "docker", "[red]fail[/]", f"daemon error: {result.stderr.strip()}"
                )
        except Exception as exc:
            failures.append(f"docker: {exc}")
            table.add_row("docker", "[red]fail[/]", str(exc))
    else:
        failures.append("docker binary not in PATH")
        table.add_row("docker", "[red]fail[/]", "binary not in PATH")

    # --- disk ---
    try:
        usage = shutil.disk_usage(cfg.runs_dir.parent if cfg.runs_dir.parent.exists() else "/")
        free_gb = usage.free / (1024**3)
        if free_gb >= 100:
            table.add_row("disk", "[green]ok[/]", f"{free_gb:.1f} GB free")
        elif free_gb >= 20:
            table.add_row("disk", "[yellow]warn[/]", f"{free_gb:.1f} GB free (V8 builds want ≥100)")
        else:
            failures.append(f"disk: only {free_gb:.1f} GB free")
            table.add_row("disk", "[red]fail[/]", f"{free_gb:.1f} GB free")
    except Exception as exc:
        table.add_row("disk", "[yellow]warn[/]", str(exc))

    # --- .env permissions ---
    # Default umask 022 makes a touched .env world-readable (644). With
    # provider keys inside, that's a quiet leak to any local user. Warn
    # if too permissive; `make install` chmods 600 for new clones.
    env_path = Path(".env")
    if env_path.is_file():
        import stat as _stat
        mode = env_path.stat().st_mode & 0o777
        if mode & 0o077:  # any group/other bits set
            table.add_row(
                ".env perms",
                "[yellow]warn[/]",
                f"{oct(mode)} (world/group readable; recommend `chmod 600 .env`)",
            )
        else:
            table.add_row(".env perms", "[green]ok[/]", oct(mode))
    else:
        table.add_row(".env perms", "[dim]missing[/]", "no .env in CWD (env vars from shell only)")

    # --- provider keys (warning only; not all needed for every run) ---
    keys = {
        "ANTHROPIC_API_KEY": cfg.anthropic_api_key,
        "OPENAI_API_KEY": cfg.openai_api_key,
        "GEMINI_API_KEY": cfg.gemini_api_key,
        "OPENROUTER_API_KEY": cfg.openrouter_api_key,
    }
    for name, val in keys.items():
        if val:
            table.add_row(name, "[green]ok[/]", f"set ({len(val)} chars)")
        else:
            table.add_row(name, "[yellow]unset[/]", "models for this provider will fail")

    # --- gateway ---
    if cfg.openai_api_base:
        table.add_row(
            "OPENAI_API_BASE",
            "[green]set[/]",
            f"{cfg.openai_api_base} (all openai/* routed here)",
        )
    else:
        table.add_row(
            "OPENAI_API_BASE",
            "[dim]unset[/]",
            "openai/* will hit api.openai.com directly",
        )

    # --- bench-v8 paths ---
    # Two distinct things to check:
    #   1. bench-v8 SOURCES (the submodule itself). If missing, it means
    #      `git submodule update --init` was never run — a real setup gap.
    #   2. bench-v8/bugs/ — regenerable BUILD OUTPUT (gitignored, produced
    #      by bench-v8/dataset-bootstrap.py). Only needed if you intend
    #      to BUILD V8 docker images on this host. Run-only hosts that
    #      pull images from ECR don't need it.
    bench_v8_sources = cfg.bench_v8_bugs.parent / "README.md"
    if bench_v8_sources.is_file():
        table.add_row(
            "bench-v8 sources",
            "[green]ok[/]",
            f"{cfg.bench_v8_bugs.parent} (submodule initialized)",
        )
    else:
        table.add_row(
            "bench-v8 sources",
            "[red]missing[/]",
            f"{cfg.bench_v8_bugs.parent} — run "
            "`git submodule update --init benchmarks/bench-v8`",
        )

    if cfg.bench_v8_bugs.is_dir():
        bug_count = sum(1 for p in cfg.bench_v8_bugs.iterdir() if p.is_dir())
        table.add_row(
            "bench-v8 build outputs",
            "[green]ok[/]",
            f"{cfg.bench_v8_bugs} ({bug_count} bug dirs)",
        )
    else:
        table.add_row(
            "bench-v8 build outputs",
            "[dim]not built[/]",
            f"{cfg.bench_v8_bugs} — only needed if BUILDING V8 images "
            "on this host (run-from-ECR doesn't need this)",
        )

    # --- python deps (cheap import probe) ---
    for mod in ("anthropic", "litellm", "mcp", "typer", "pydantic"):
        try:
            __import__(mod)
            table.add_row(f"py:{mod}", "[green]ok[/]", "importable")
        except ImportError as exc:
            failures.append(f"missing python dep {mod}")
            table.add_row(f"py:{mod}", "[red]fail[/]", str(exc))

    console.print(table)
    if failures:
        console.print(f"\n[bold red]{len(failures)} check(s) failed.[/]")
        raise typer.Exit(code=1)
    console.print("\n[bold green]all checks passed.[/]")


# ---------------------------------------------------------------------------
# benchmark / smoke / aggregate / import-eval — stubs filled in later days
# ---------------------------------------------------------------------------


@app.command()
def benchmark(
    config: Path | None = typer.Option(None, help="Path to benchmark JSON config"),
    test: bool = typer.Option(False, "--test", help="Real LLM × sample-stack-bof × 1 seed (~$0.10)"),
    mock_llm: bool = typer.Option(False, "--mock-llm", help="Stub LLM × sample-stack-bof × 1 seed ($0)"),
    max_parallel: int | None = typer.Option(None, "--max-parallel", "-p"),
    resume: bool = typer.Option(False, "--resume", help="Skip rows already in DB"),
    retry_failed: bool = typer.Option(
        False, "--retry-failed",
        help="Delete prior infra_failed/model_failed rows for this benchmark_id "
             "so the sweep retries them. succeeded rows are never touched.",
    ),
    resume_failed: bool = typer.Option(
        False, "--resume-failed",
        help="Pick up resumable failures (transient timeouts, episode "
             "wallclock, orchestrator crashes) by replaying their tool "
             "sequence and continuing the agent loop where it died. Doesn't "
             "schedule new tuples. Mutually exclusive with --retry-failed.",
    ),
    episode_timeout: int | None = typer.Option(
        None, "--episode-timeout",
        help="Per-tuple wallclock cap in seconds (overrides config; defaults "
             "to BenchmarkConfig.episode_timeout_s = 1800).",
    ),
    cost_cap_usd: float | None = typer.Option(
        None, "--cost-cap-usd",
        help="Abort scheduling further tuples once running spend reaches this "
             "USD total. Tuples that don't run are recorded as infra_failed "
             "with reason cost_cap_exceeded; --retry-failed brings them back.",
    ),
    models: list[str] | None = typer.Option(
        None, "--models", "-m",
        help="Filter the config's models list to those provided. Comma-"
             "separated or repeat the flag. Each entry must exactly match a "
             "model id declared in the config (typo guard). Lets one ground-"
             "truth config (e.g. benchmarks/v8.yaml) be run per-model "
             "without forking copies.",
    ),
    envs: list[str] | None = typer.Option(
        None, "--envs", "-e",
        help="Filter the config's envs list to those env ids provided. "
             "Same shape and typo-guard as --models. Lets v8.yaml be run "
             "against a single bug from the matrix without forking.",
    ),
    seeds: list[str] | None = typer.Option(
        None, "--seeds",
        help="Filter the config's seeds list to those values provided "
             "(comma-separated or repeat). Each entry is parsed as int and "
             "must appear in the config's seeds.",
    ),
    set_overrides: list[str] | None = typer.Option(
        None, "--set",
        help="Generic YAML field override: --set <dotted.key>=<value>. "
             "Repeat or comma-separate. Value is YAML-parsed (`100` → int, "
             "`true` → bool, `[a,b]` → list). Examples: "
             "`--set budgets.turn_budget=100`, "
             "`--set init_prompt_hint_path=benchmarks/prompts/init-v2-hint.template`. "
             "Applies before parse_config so overrides go through the "
             "same validation as the YAML. Use --models/--envs/--seeds "
             "to filter list-of-dicts; --set is for everything else.",
    ),
    turn_budget: int | None = typer.Option(
        None, "--turn-budget",
        help="Override the config's budgets.turn_budget for this run. "
             "Use for cheap shakedowns against an otherwise-flagship "
             "config (e.g. --turn-budget 30 across all 7 keepers to "
             "validate plumbing before committing to a 300-turn sweep) "
             "without forking the YAML.",
    ),
    nudges: str | None = typer.Option(
        None, "--nudges",
        help="Override the config's nudges setting: 'true' enables stuck/"
             "wrapup/voluntary nudges (matches imported-opus historical "
             "baseline); 'false' is clean evaluation. The runtime form "
             "supports the same true/false/list shapes as the YAML.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print planned tuples + resolved digests; don't run"),
) -> None:
    """Run a multi-model benchmark."""
    import asyncio
    import time
    from datetime import UTC, datetime

    from qed_swe_bench.runner.orchestrator import (
        BenchmarkConfig,
        Budgets,
        EnvSpec,
        ModelSpec,
        parse_config,
        run_benchmark,
    )

    if sum([config is not None, test, mock_llm]) != 1:
        console.print(
            "[red]Pick exactly one of: --config <path>, --test, or --mock-llm[/]"
        )
        raise typer.Exit(code=2)

    if retry_failed and resume_failed:
        console.print(
            "[red]--retry-failed and --resume-failed are mutually exclusive: "
            "the first deletes prior failures, the second continues them.[/]"
        )
        raise typer.Exit(code=2)

    # --set is YAML-shaped and only meaningful against a YAML config.
    # The --test / --mock-llm paths build their BenchmarkConfig in-process.
    if set_overrides and config is None:
        console.print(
            "[red]--set requires --config; the --test / --mock-llm paths "
            "build their BenchmarkConfig in-process and don't accept "
            "YAML overrides.[/]"
        )
        raise typer.Exit(code=2)

    if config is not None:
        # YAML is a superset of JSON for our purposes — yaml.safe_load handles
        # both .yaml and .json files identically. Comments are the win.
        import yaml as _yaml
        from qed_swe_bench.runner.orchestrator_config import apply_overrides

        config_dict = _yaml.safe_load(config.read_text(encoding="utf-8"))
        # Apply --set overrides BEFORE parse_config so they go through the
        # same validation as the YAML (and surface as the same errors).
        if set_overrides:
            flat_sets: list[str] = []
            for entry in set_overrides:
                flat_sets.extend(p.strip() for p in entry.split(",") if p.strip())
            try:
                config_dict = apply_overrides(config_dict, flat_sets)
            except ValueError as exc:
                console.print(f"[red]--set: {exc}[/]")
                raise typer.Exit(code=2) from exc
        bench = parse_config(config_dict)
    else:
        # --test / --mock-llm: hardcoded scenario for the sample-stack-bof env.
        # --test default model: prefer direct Anthropic (caching) when the
        # ANTHROPIC_API_KEY is set; fall back to OpenRouter for non-Anthropic.
        # NEVER route Anthropic models through OpenRouter (loses caching).
        from qed_swe_bench.config import Config as _Config
        _cfg = _Config.from_env()
        # Mock runs don't make a real API call but still need a model id
        # for routing; treat them the same as the Anthropic-direct path
        # since the factory's mock branch only inspects the prefix.
        if mock_llm or _cfg.anthropic_api_key:
            test_model = "anthropic/claude-haiku-4-5"
        elif _cfg.openrouter_api_key:
            test_model = "openrouter/openai/gpt-4o-mini"
        elif _cfg.openai_api_key:
            test_model = "openai/gpt-4o-mini"
        elif _cfg.gemini_api_key:
            test_model = "gemini/gemini-2.5-flash-lite"
        else:
            console.print(
                "[red]--test needs at least one provider key set "
                "(OPENROUTER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY).[/]"
            )
            raise typer.Exit(code=2)

        # --test / --mock-llm builds a non-V8 BenchmarkConfig in-process.
        # All bug-specific framing comes from the container's MCP setup();
        # init_prompt falls through to the runner's default.
        bench = BenchmarkConfig(
            benchmark_id=f"{'mock' if mock_llm else 'test'}-{int(time.time())}",
            models=[ModelSpec(id=test_model)],
            envs=[
                EnvSpec(
                    id="sample-stack-bof",
                    image="local/sample-stack-bof:latest",
                    task_type="binary_task",
                ),
            ],
            seeds=[1],
            budgets=Budgets(turn_budget=20, token_budget=100_000, context_budget=50_000),
            max_parallel=1,
        )

    if models is not None:
        # Accept comma-separated values within a single --models flag in
        # addition to the typer-native repeat-flag form. Both shapes are
        # ergonomic; we don't want to force one or the other.
        wanted: list[str] = []
        for entry in models:
            wanted.extend(part.strip() for part in entry.split(",") if part.strip())
        declared_ids = [m.id for m in bench.models]
        unknown = [w for w in wanted if w not in declared_ids]
        if unknown:
            console.print(
                f"[red]--models referenced ids not in this config: "
                f"{unknown}[/]\nDeclared in {config}: {declared_ids}"
            )
            raise typer.Exit(code=2)
        filtered = [m for m in bench.models if m.id in wanted]
        if not filtered:
            console.print("[red]--models filter left no models to run.[/]")
            raise typer.Exit(code=2)
        bench = bench.with_overrides(models=filtered)

    if envs is not None:
        wanted_env_ids: list[str] = []
        for entry in envs:
            wanted_env_ids.extend(part.strip() for part in entry.split(",") if part.strip())
        declared_env_ids = [e.id for e in bench.envs]
        unknown_envs = [w for w in wanted_env_ids if w not in declared_env_ids]
        if unknown_envs:
            console.print(
                f"[red]--envs referenced ids not in this config: "
                f"{unknown_envs}[/]\nDeclared in {config}: {declared_env_ids}"
            )
            raise typer.Exit(code=2)
        filtered_envs = [e for e in bench.envs if e.id in wanted_env_ids]
        if not filtered_envs:
            console.print("[red]--envs filter left no envs to run.[/]")
            raise typer.Exit(code=2)
        bench = bench.with_overrides(envs=filtered_envs)

    if seeds is not None:
        wanted_seeds: list[int] = []
        for entry in seeds:
            for part in entry.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    wanted_seeds.append(int(part))
                except ValueError:
                    console.print(
                        f"[red]--seeds value {part!r} is not an int[/]"
                    )
                    raise typer.Exit(code=2) from None
        declared_seeds = list(bench.seeds)
        unknown_seeds = [s for s in wanted_seeds if s not in declared_seeds]
        if unknown_seeds:
            console.print(
                f"[red]--seeds referenced values not in this config: "
                f"{unknown_seeds}[/]\nDeclared in {config}: {declared_seeds}"
            )
            raise typer.Exit(code=2)
        filtered_seeds = [s for s in declared_seeds if s in wanted_seeds]
        if not filtered_seeds:
            console.print("[red]--seeds filter left no seeds to run.[/]")
            raise typer.Exit(code=2)
        bench = bench.with_overrides(seeds=filtered_seeds)

    if turn_budget is not None:
        # Reach into the nested Budgets dataclass; replace the budgets
        # field on the BenchmarkConfig with a copy that has the new
        # turn_budget. Other budget axes (token_budget, context_budget,
        # max_tokens) are preserved.
        from dataclasses import replace as _replace
        bench = bench.with_overrides(
            budgets=_replace(bench.budgets, turn_budget=turn_budget)
        )

    if nudges is not None:
        # Reuse the YAML-side parser by mapping the CLI string forms to
        # the same shapes parse_nudges accepts. Keeps behavior identical
        # whether the value comes from YAML or the CLI.
        from qed_swe_bench.runner.orchestrator_config import parse_nudges
        v = nudges.strip().lower()
        if v == "true":
            parsed_nudges = parse_nudges(True)
        elif v == "false":
            parsed_nudges = parse_nudges(False)
        else:
            parsed_nudges = parse_nudges(
                [p.strip() for p in nudges.split(",") if p.strip()]
            )
        bench = bench.with_overrides(nudges=parsed_nudges)

    overrides: dict[str, object] = {}
    if max_parallel is not None:
        overrides["max_parallel"] = max_parallel
    if episode_timeout is not None:
        overrides["episode_timeout_s"] = episode_timeout
    if cost_cap_usd is not None:
        overrides["cost_cap_usd"] = cost_cap_usd
    if overrides:
        bench = bench.with_overrides(**overrides)

    console.print(
        f"[bold]benchmark[/] id={bench.benchmark_id} "
        f"models={len(bench.models)} envs={len(bench.envs)} seeds={len(bench.seeds)} "
        f"parallel={bench.max_parallel} mock_llm={mock_llm} dry_run={dry_run}"
    )

    if dry_run:
        from qed_swe_bench.runner.image_ref import ImageRefError, resolve
        console.print("\n[bold]planned tuples (model × env × seed):[/]")
        for env in bench.envs:
            try:
                resolved = resolve(env.image)
                digest_str = resolved.image_digest[:24]
                pulled = "(pulled)" if resolved.pulled else ""
            except ImageRefError as exc:
                digest_str = "[red]unresolved[/]"
                pulled = f"({exc})"
            console.print(f"  env  {env.id:30}  {env.image:60}  {digest_str} {pulled}")
        for m in bench.models:
            console.print(f"  model {m.id}")
        n_tuples = len(bench.models) * len(bench.envs) * len(bench.seeds)
        console.print(f"\n[bold]would run {n_tuples} episodes[/] (dry-run; nothing executed)")
        return

    start_iso = datetime.now(UTC).isoformat()
    if resume_failed:
        from qed_swe_bench.runner.resume import run_resume_batch
        histogram = asyncio.run(run_resume_batch(bench, mock_llm=mock_llm))
    else:
        histogram = asyncio.run(
            run_benchmark(
                bench,
                mock_llm=mock_llm,
                retry_failed=retry_failed,
                config_path=config,
            ),
        )

    console.print("\n[bold]done.[/] status histogram:")
    for status, count in sorted(histogram.items()):
        console.print(f"  {status:20} {count}")

    # Show what actually went wrong for any non-succeeded outcome. Two cases
    # we have to handle separately, because a histogram-only summary hides
    # both:
    #   1. NEW failures from this invocation: `started_at >= start_iso` and
    #      `status != 'succeeded'`.
    #   2. SKIPPED prior failures: `resumed_skip` means the (model, env, seed)
    #      tuple was already in the DB with a non-succeeded status, so the
    #      orchestrator silently no-op'd. The relevant row predates this
    #      invocation, so it's *not* caught by case 1 — but the user still
    #      saw nothing run and deserves to know why.
    def _print_failure_rows(rows, header: str) -> None:
        if not rows:
            return
        console.print(f"\n[bold red]{header}[/]")
        for r in rows:
            first_line = (r["failure_reason"] or "").strip().splitlines()
            reason_str = first_line[0] if first_line else "(no detail)"
            if len(reason_str) > 200:
                reason_str = reason_str[:200] + "…"
            console.print(
                f"  [red]{r['status']}[/] {r['run_id']}  "
                f"{r['model']} × {r['env_id']}\n"
                f"    exit_reason: {r['exit_reason'] or '(none)'}\n"
                f"    {reason_str}"
            )

    bug_rows: list = []
    if any(s != "succeeded" for s in histogram):
        with connect(Config.from_env().db_path) as con:
            new_rows = con.execute(
                """
                SELECT run_id, model, env_id, status, exit_reason, failure_reason
                FROM runs
                WHERE benchmark_id = ?
                  AND started_at >= ?
                  AND status != 'succeeded'
                ORDER BY started_at
                """,
                (bench.benchmark_id, start_iso),
            ).fetchall()
            skipped_rows = []
            if histogram.get("resumed_skip", 0) > 0:
                skipped_rows = con.execute(
                    """
                    SELECT run_id, model, env_id, status, exit_reason, failure_reason
                    FROM runs
                    WHERE benchmark_id = ?
                      AND started_at < ?
                      AND status != 'succeeded'
                    ORDER BY started_at
                    """,
                    (bench.benchmark_id, start_iso),
                ).fetchall()

        # Split by category. The orchestrator tags exit_reason `bug_<Type>`
        # for exceptions that look like code bugs (FileNotFoundError, KeyError,
        # etc.) vs `infra_<Type>` / `image_unresolved` / `episode_timeout_*` /
        # `cost_cap_exceeded` for the rest. Bugs deserve a louder banner
        # because the operator should investigate them before next run.
        def _is_bug(row) -> bool:
            er = row["exit_reason"] or ""
            return er.startswith("bug_")

        bug_rows = [r for r in new_rows if _is_bug(r)]
        infra_new = [r for r in new_rows if not _is_bug(r)]
        bug_rows += [r for r in skipped_rows if _is_bug(r)]
        infra_skipped = [r for r in skipped_rows if not _is_bug(r)]

        _print_failure_rows(infra_new, "failures (this run):")
        _print_failure_rows(
            infra_skipped,
            "skipped — prior failures still in DB "
            "(re-run with --retry-failed to retry):",
        )
        if bug_rows:
            console.print(
                "\n[bold red on white]  CODE BUGS — please investigate  [/]"
            )
            console.print(
                "[red]An exception escaped that doesn't look like an infra "
                "issue. The most likely cause is a bug in qed_swe_bench, not "
                "the env. See traceback above.[/]"
            )
            _print_failure_rows(bug_rows, "code bugs:")

    console.print(
        f"\n[dim]view results:[/] "
        f"[bold]qed_swe_bench aggregate --benchmark-id {bench.benchmark_id}[/]"
    )

    # Exit nonzero on any failure type. Bugs are the most important signal
    # so they short-circuit the exit code regardless of histogram.
    if bug_rows or histogram.get("infra_failed", 0) > 0:
        raise typer.Exit(code=1)


def _build_summary_view(benchmark_id: str | None):
    """Build the summary's Table + spend footer as a single renderable.

    Pulled out of `summary()` so `--watch` can re-render on a timer
    without duplicating the SQL + formatting code.
    """
    from datetime import datetime

    from rich.console import Group
    from rich.text import Text

    where_clause = "WHERE benchmark_id = ?" if benchmark_id else ""
    params = (benchmark_id,) if benchmark_id else ()

    with connect() as con:
        rows = con.execute(
            f"""
            SELECT benchmark_id, model, provenance, status,
                   count(*) AS n,
                   round(sum(cost_usd), 4) AS total_cost,
                   round(avg(score), 2) AS avg_score
            FROM runs
            {where_clause}
            GROUP BY benchmark_id, model, provenance, status
            ORDER BY benchmark_id, model, provenance, status
            """,
            params,
        ).fetchall()

    title_suffix = f" (benchmark_id={benchmark_id})" if benchmark_id else ""
    table = Table(title=f"qed_swe_bench summary{title_suffix}")
    table.add_column("benchmark")
    table.add_column("model")
    table.add_column("provenance")
    table.add_column("status")
    table.add_column("n", justify="right")
    table.add_column("avg score", justify="right")
    table.add_column("cost ($)", justify="right")

    if not rows:
        # Render an empty table + a note so --watch keeps redrawing
        # cleanly while waiting for the first row to land.
        return Group(
            table,
            Text("no runs yet", style="yellow"),
            Text(f"refreshed {datetime.now().strftime('%H:%M:%S')}",
                 style="dim"),
        )

    real_total = 0.0
    imputed_total = 0.0
    for r in rows:
        cost_disp = f"{r['total_cost']:.4f}" if r["total_cost"] is not None else "-"
        score_disp = f"{r['avg_score']:.2f}" if r["avg_score"] is not None else "-"
        prov = r["provenance"] or "?"
        # Color cue: imputed historical vs real fresh.
        prov_disp = f"[dim]{prov}[/]" if prov.startswith("imported_from_") else prov
        table.add_row(
            r["benchmark_id"],
            r["model"],
            prov_disp,
            r["status"],
            str(r["n"]),
            score_disp,
            cost_disp,
        )
        if r["total_cost"]:
            if prov.startswith("imported_from_"):
                imputed_total += r["total_cost"]
            else:
                real_total += r["total_cost"]
    footer = Text.from_markup(
        f"[bold green]real fresh spend: ${real_total:.4f}[/]"
        f"  (imputed historical: ${imputed_total:.2f})"
    )
    timestamp = Text(
        f"refreshed {datetime.now().strftime('%H:%M:%S')}",
        style="dim",
    )
    return Group(table, "", footer, timestamp)


@app.command()
def summary(
    benchmark_id: str | None = typer.Option(None, "--benchmark-id", "-b", help="Filter to one benchmark"),
    watch: bool = typer.Option(
        False, "--watch", "-w",
        help="Re-render the table on an interval until Ctrl+C "
             "(useful during a long-running sweep).",
    ),
    interval: float = typer.Option(
        5.0, "--interval",
        help="Seconds between refreshes when --watch is set (default 5).",
        min=0.5, max=300,
    ),
) -> None:
    """Spend / status table per (benchmark, model, provenance, status).

    Provenance distinguishes fresh runs ('native', 'mock') from imported
    sources (any 'imported_from_*' — currently 'imported_from_legacy' for
    vr-agent, 'imported_from_codex' for codex CLI sweeps, 'imported_from_eval'
    for the older bench-v8 eval/ path). Imported runs have IMPUTED cost
    (computed from token counts via the local pricing table); real spend
    is the sum over non-imported provenances.
    """
    if not watch:
        console.print(_build_summary_view(benchmark_id))
        return

    # Live-redraw mode. `screen=True` uses the alt-screen so the table
    # redraws in place (htop-like) rather than scrolling history. Ctrl+C
    # exits via the KeyboardInterrupt catch in the Live context manager.
    import time

    from rich.live import Live

    try:
        with Live(
            _build_summary_view(benchmark_id),
            console=console,
            refresh_per_second=4,
            screen=True,
            transient=False,
        ) as live:
            while True:
                time.sleep(interval)
                live.update(_build_summary_view(benchmark_id))
    except KeyboardInterrupt:
        # Print once on exit so the operator sees the final state in
        # their normal terminal scrollback, not just the alt-screen.
        console.print(_build_summary_view(benchmark_id))


@app.command()
def audit(
    benchmark_id: str | None = typer.Option(
        None, "--benchmark-id", "-b",
        help="Audit every run dir under runs/<benchmark_id>/. "
             "Mutually exclusive with --run-id.",
    ),
    run_id: str | None = typer.Option(
        None, "--run-id",
        help="Audit a single run by run_id (looked up in the DB to find "
             "its run_dir). Mutually exclusive with --benchmark-id.",
    ),
    detail: bool = typer.Option(
        False, "--detail",
        help="Print the offending excerpt for every finding.",
    ),
    output_format: str = typer.Option(
        "table", "--format", "-f",
        help="'table' (default, human-readable) or 'json' (machine).",
    ),
    reproduce: bool = typer.Option(
        False, "--reproduce",
        help="Replay each grade() call from the run against a fresh "
             "container and compare the recorded capabilities to what "
             "the grader emits this time. Catches PoCs that hardcode "
             "addresses (won't repro) and any forged GRADER_RESULT_FD "
             "output (a real re-grade re-fires the actual grader).",
    ),
) -> None:
    """Scan run transcripts for red flags (refusal language, off-workspace
    writes, hardcoded addresses in PoCs, etc.).

    **The audit focuses manual review; it is not a definitive judgment.**
    Checks are grep-shaped on purpose so they themselves remain
    auditable, which means false positives are expected (especially in
    C1's substring matching). A finding flags a run *for human
    inspection*; it does not establish that anything wrong
    happened. Treat HIGH/MEDIUM/INFO as "how loudly to look," not
    "how guilty."

    See `qed_swe_bench/audit/transcripts.py` for the 11 checks
    (C1–C11) and per-check rationale. C10/C11 verify model identity —
    that the provider actually served the requested model and didn't
    silently downgrade reasoning effort.

    Use this whenever a run completes — `make audit BENCHMARK_ID=<id>`
    is sugar for the same thing — and before sharing audit-bundle
    tarballs with reward-hacking auditors.
    """
    from qed_swe_bench.audit import Severity, audit_runs
    from qed_swe_bench.config import Config as _Config

    if (benchmark_id is None) == (run_id is None):
        console.print(
            "[red]Pass exactly one of --benchmark-id or --run-id.[/]"
        )
        raise typer.Exit(code=2)

    cfg = _Config.from_env()
    from qed_swe_bench.db.schema import connect

    if benchmark_id is not None:
        # Resolve via the DB (canonical pointer) — works for both legacy
        # `runs/<benchmark_id>/<utc-iso>__<run_id>/` and the new D-10
        # `runs/<benchmark_id>/<host>/<datetime>/<run_id>/` layout. Also
        # picks up imported / cross-host run-dirs whose absolute path
        # might not live under cfg.runs_dir at all.
        with connect(cfg.db_path) as con:
            rows = con.execute(
                "SELECT run_dir FROM runs WHERE benchmark_id = ? "
                "AND run_dir IS NOT NULL "
                "ORDER BY started_at",
                (benchmark_id,),
            ).fetchall()
        if not rows:
            console.print(
                f"[red]No runs in DB for benchmark_id={benchmark_id!r}. "
                f"Did you `qed_swe_bench import runs/`?[/]"
            )
            raise typer.Exit(code=2)
        run_dirs = [Path(r["run_dir"]) for r in rows]
    else:
        # --run-id: look up the run_dir from the DB.
        with connect(cfg.db_path) as con:
            row = con.execute(
                "SELECT run_dir, benchmark_id FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            console.print(f"[red]run_id {run_id!r} not in the runs DB.[/]")
            raise typer.Exit(code=2)
        run_dirs = [Path(row["run_dir"])]

    reports = audit_runs(run_dirs)

    if reproduce:
        # Reproduction is async + docker-bound; run it after the static
        # audit so a fast clean/fail signal lands first, then the
        # expensive replay either confirms or contradicts.
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, TextColumn,
            TimeElapsedColumn, TimeRemainingColumn,
        )

        from qed_swe_bench.audit import reproduce_run_sync

        repro_reports = []
        for run_dir in run_dirs:
            console.print(f"[bold]reproducing[/] {run_dir.name}...")
            with Progress(
                TextColumn("  [cyan]replay[/]"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[dim]{task.fields[tool]}[/]"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console, transient=True,
            ) as progress:
                task_id = progress.add_task("replay", total=None, tool="…")

                def _on_progress(idx: int, total: int | None, tool: str, dur: float) -> None:
                    if total is not None and progress.tasks[task_id].total != total:
                        progress.update(task_id, total=total)
                    progress.update(task_id, completed=idx + 1, tool=tool)
                try:
                    rr = reproduce_run_sync(run_dir, on_progress=_on_progress)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [red]error[/]: {type(exc).__name__}: {exc}")
                    continue
            repro_reports.append(rr)
            if rr.error:
                console.print(f"  [red]session error[/]: {rr.error}")
                continue
            for c in rr.comparisons:
                if c.matches:
                    rec_caps = sorted(k for k, v in c.recorded.items() if v)
                    console.print(
                        f"  [green]✓ match[/] grade #{c.grade_call_index + 1} "
                        f"({c.poc_path}): {rec_caps}"
                    )
                else:
                    if c.error:
                        console.print(
                            f"  [red]✗ error[/] grade #{c.grade_call_index + 1} "
                            f"({c.poc_path}): {c.error}"
                        )
                    else:
                        console.print(
                            f"  [yellow]⚠ divergence[/] grade #{c.grade_call_index + 1} "
                            f"({c.poc_path})"
                        )
                        if c.only_recorded:
                            console.print(
                                f"    in recorded only: {list(c.only_recorded)}"
                            )
                        if c.only_reproduced:
                            console.print(
                                f"    in reproduced only: {list(c.only_reproduced)}"
                            )
        all_match = all(rr.overall_match for rr in repro_reports) if repro_reports else False
        if all_match:
            console.print(f"\n[green]✓ all {len(repro_reports)} run(s) reproduced their recorded capabilities[/]")
        else:
            console.print(f"\n[yellow]⚠ at least one run diverged from its recording[/]")
        # Reproduction output is human-only; --format json with --reproduce
        # is supported via the static-audit path above + the per-run prints.
        # Skip the default table render if both flags were passed; the
        # reproduce log above is the operator-relevant view.
        return

    if output_format == "json":
        import json
        out = []
        for r in reports:
            out.append({
                "run_id": r.run_id,
                "run_dir": str(r.run_dir),
                "status": r.status,
                "highest_severity": r.highest_severity.value if r.highest_severity else None,
                "findings": [
                    {
                        "check_id": f.check_id,
                        "name": f.name,
                        "severity": f.severity.value,
                        "detail": f.detail,
                        "excerpt": f.excerpt,
                    }
                    for f in r.findings
                ],
            })
        sys.stdout.write(json.dumps(out, indent=2) + "\n")
        return

    # Default 'table' format.
    from rich.table import Table

    if not reports:
        console.print("[yellow]No runs to audit.[/]")
        return

    # Header summary
    total = len(reports)
    clean = sum(1 for r in reports if r.clean)
    high = sum(1 for r in reports if r.highest_severity == Severity.HIGH)
    med = sum(1 for r in reports if r.highest_severity == Severity.MEDIUM)
    info = sum(1 for r in reports if r.highest_severity == Severity.INFO)
    console.print(
        f"[bold]audit[/] {total} run(s): {clean} clean, "
        f"{high} HIGH, {med} MEDIUM, {info} INFO"
    )

    # Per-check check_ids in fixed order so rendering is stable.
    check_ids = [f"C{i}" for i in range(1, 12)]

    table = Table(show_header=True, header_style="bold")
    table.add_column("run_id", overflow="fold")
    for cid in check_ids:
        table.add_column(cid, justify="center")

    for r in reports:
        row = [r.run_id]
        by_check: dict[str, str] = {}
        for f in r.findings:
            mark = {"HIGH": "[red]✗[/]", "MEDIUM": "[yellow]⚠[/]", "INFO": "[cyan]ⓘ[/]"}[f.severity.value]
            # Worst severity wins per check_id (HIGH > MEDIUM > INFO).
            existing = by_check.get(f.check_id)
            if existing is None:
                by_check[f.check_id] = mark
            elif "✗" in existing:
                pass
            elif "⚠" in existing and mark == "[red]✗[/]":
                by_check[f.check_id] = mark
            elif existing == "[cyan]ⓘ[/]" and mark != "[cyan]ⓘ[/]":
                by_check[f.check_id] = mark
        for cid in check_ids:
            row.append(by_check.get(cid, "[green]✓[/]"))
        table.add_row(*row)

    console.print(table)

    if detail:
        for r in reports:
            if r.clean:
                continue
            console.print(f"\n[bold]{r.run_id}[/]:")
            for f in r.findings:
                tag = {"HIGH": "[red]HIGH[/]", "MEDIUM": "[yellow]MED [/]", "INFO": "[cyan]INFO[/]"}[f.severity.value]
                console.print(f"  {tag} {f.check_id} {f.name}: {f.detail}")
                if f.excerpt:
                    indented = "\n    ".join(f.excerpt.splitlines())
                    console.print(f"    [dim]{indented}[/]")


@app.command(name="register-dir")
def register_dir_cmd(
    bugs_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    interface: str = typer.Option(
        "rl.mcp.v8_task.v1", "--interface",
        help="Interface name from contract/interfaces.py for each env.",
    ),
) -> None:
    """Walk a bench-v8 bugs/ directory and register each subdir as an env."""
    from qed_swe_bench.catalog import register_dir
    histogram = register_dir(bugs_dir, interface=interface)
    console.print("[bold]register-dir[/] histogram:")
    for status, count in sorted(histogram.items()):
        console.print(f"  {status:24} {count}")


@app.command(name="api")
def serve_api(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev)"),
) -> None:
    """Start the FastAPI JSON backend that serves the webui."""
    import uvicorn
    uvicorn.run(
        "qed_swe_bench.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command(name="list-interfaces")
def list_interfaces() -> None:
    """List all registered RL env interfaces (the contract registry)."""
    from qed_swe_bench.contract import interfaces as iface_mod

    table = Table(title="qed_swe_bench RL env interfaces (rl.mcp.<task>.<version>)")
    table.add_column("interface", overflow="fold")
    table.add_column("base", overflow="fold")
    table.add_column("grader")
    table.add_column("submission")
    table.add_column("flags", justify="right")
    table.add_column("description", overflow="fold")
    for name in iface_mod.all_names():
        i = iface_mod.lookup(name)
        if i is None:
            continue
        table.add_row(
            i.name,
            i.base or "-",
            i.grader_kind,
            i.submission_kind or "-",
            str(len(i.capability_flags)) if i.capability_flags else "-",
            i.description.split(".")[0] + ".",  # first sentence
        )
    console.print(table)


@app.command()
def resume(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False,
        help="Path to a run dir on disk (e.g. runs/v8/host/.../<run-id>).",
    ),
    episode_timeout: int = typer.Option(
        18000, "--episode-timeout",
        help="Per-tuple wallclock cap in seconds for the resumed run.",
    ),
    mock_llm: bool = typer.Option(
        False, "--mock-llm", help="Resume against MockClient instead of the real provider.",
    ),
) -> None:
    """Resume one failed/partial run from its run dir.

    Replays the run's tool sequence against a fresh container to rebuild
    filesystem state, rehydrates the LLM message history, then continues
    `run_episode` from where it died. Per-model params and budgets come
    from `<run_dir>/config_snapshot.yaml` (D-13 byte-identical bijection
    with runs.config_snapshot). Original run dir + DB row are re-used
    (transcript files appended to).
    """
    import asyncio
    from qed_swe_bench.runner.resume import resume_one_run

    # Resume must mirror the original run's limits — never invent its
    # own. The run-dir's config_snapshot.yaml is the bijection contract
    # (D-13). If it's missing or unparseable we cannot faithfully
    # continue; bail loudly rather than silently using fabricated
    # defaults.
    model_params: dict = {}
    reasoning_replay = "auto"
    snapshot = run_dir / "config_snapshot.yaml"
    if not snapshot.is_file():
        console.print(
            f"[red]config_snapshot.yaml missing in {run_dir}; cannot "
            "resume without the original run's budgets. Re-run from "
            "scratch with the source YAML, or write a snapshot first.[/]"
        )
        raise typer.Exit(code=2)
    try:
        from qed_swe_bench.runner.orchestrator import parse_config
        bench = parse_config(snapshot)
        budgets = bench.budgets
        job = json.loads((run_dir / "job.json").read_text())
        model = str(job.get("model", ""))
        for m in bench.models:
            if m.id == model:
                model_params = m.params or {}
                reasoning_replay = m.reasoning_replay
                break
    except (OSError, ValueError) as exc:
        console.print(
            f"[red]config_snapshot.yaml in {run_dir} failed to parse: "
            f"{exc}; cannot resume without the original run's budgets.[/]"
        )
        raise typer.Exit(code=2)

    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn,
    )
    progress_state: dict = {"progress": None, "task_id": None}

    def _on_progress(idx: int, total: int | None, tool: str, _dur: float) -> None:
        if progress_state["progress"] is None:
            p = Progress(
                TextColumn("  [cyan]replay[/]"), BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[dim]{task.fields[tool]}[/]"),
                TimeElapsedColumn(), TimeRemainingColumn(),
                console=console, transient=True,
            )
            p.start()
            progress_state["progress"] = p
            progress_state["task_id"] = p.add_task("replay", total=total, tool=tool or "…")
        p = progress_state["progress"]
        if total is not None and p.tasks[progress_state["task_id"]].total != total:
            p.update(progress_state["task_id"], total=total)
        p.update(progress_state["task_id"], completed=idx, tool=tool or "…")
        if total is not None and idx >= total:
            p.stop()
            progress_state["progress"] = None

    from qed_swe_bench.runner.resilience import HeartbeatTracker

    async def _run_with_heartbeat():
        # Same lifecycle as run_resume_batch: start tracker, hand it to
        # resume_one_run, stop on exit. Keeps the row's last_heartbeat
        # fresh so the stale-queued sweep doesn't reap a slow resume.
        tracker = HeartbeatTracker()
        tracker.start()
        try:
            return await resume_one_run(
                run_dir,
                model_params=model_params,
                mock_llm=mock_llm,
                episode_timeout_s=episode_timeout,
                budgets=budgets,
                on_replay_progress=_on_progress,
                heartbeat=tracker,
                reasoning_replay=reasoning_replay,
            )
        finally:
            await tracker.stop()

    outcome = asyncio.run(_run_with_heartbeat())
    console.print(
        f"[bold]resume[/] {outcome.run_id}: "
        f"status={outcome.status} exit_reason={outcome.exit_reason} "
        f"runtime={outcome.runtime_s:.1f}s turns={outcome.turns_total}"
    )
    if outcome.error:
        console.print(f"[red]error:[/] {outcome.error}")
        raise typer.Exit(code=1)
    if outcome.status not in ("succeeded",):
        raise typer.Exit(code=1)


@app.command()
def smoke(
    models: list[str] = typer.Option(..., "--models", "-m", help="Model id(s) to probe"),
    check: str | None = typer.Option(
        None, "--check", help="Special check: 'anthropic-cache'"
    ),
) -> None:
    """Per-model tool-call fidelity probe."""
    console.print("[yellow]smoke: not yet implemented.[/]")
    console.print(f"  models={models} check={check}")
    raise typer.Exit(code=1)


@app.command()
def aggregate(
    benchmark_id: str = typer.Option(..., "--benchmark-id"),
    output_format: str = typer.Option("markdown", "--format", "-f"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    compare_with: str | None = typer.Option(
        None, "--compare-with",
        help=(
            "Side-by-side comparison: render --benchmark-id (primary) vs the"
            " given --compare-with (secondary) regime. Per-cell mean score,"
            " delta, and capabilities acquired only in the secondary. Use"
            " for the scaffold-effect study (e.g. --benchmark-id v8"
            " --compare-with v8-nudged)."
        ),
    ),
) -> None:
    """Emit a results table for a benchmark_id.

    With `--compare-with <other_benchmark_id>`, emits a regime-comparison
    table instead — useful for the nudges-on/nudges-off scaffold study.
    """
    from qed_swe_bench.aggregate import aggregate as _aggregate

    text = _aggregate(
        benchmark_id,
        output_format=output_format,
        output=output,
        compare_with=compare_with,
    )
    if output:
        console.print(f"[green]wrote[/] {output}")
    else:
        # Use sys.stdout so pipes work cleanly (rich's print adds wrapping).
        sys.stdout.write(text)


@app.command(name="import-eval")
def import_eval(
    eval_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    benchmark_id: str = typer.Option("imported-opus", "--benchmark-id"),
    default_model: str = typer.Option(
        "anthropic/claude-opus-4-6", "--default-model",
        help="Used when a run dir's config.json doesn't record a model.",
    ),
    nudges: str | None = typer.Option(
        None, "--nudges",
        help="REQUIRED. Whether the source benchmark ran with mid-episode "
             "scaffolding nudges enabled. Pass 'true' for the imported-opus "
             "baseline (bench-v8 ran nudges ON, per benchmarks/v8.yaml:225-232). "
             "Pass 'false' if the source ran without nudges. The vr-agent eval "
             "dir does not record this, so the importer cannot infer it.",
    ),
) -> None:
    """Ingest bench-v8/eval/ as historical rows."""
    from qed_swe_bench.historical import import_eval_dir

    if nudges is None:
        raise typer.BadParameter(
            "--nudges is required: pass 'true' if the source benchmark ran "
            "with nudges ON (typical for the imported-opus baseline), or "
            "'false' if it ran with nudges OFF. The vr-agent eval/ tree does "
            "not record this so the importer cannot infer it.",
            param_hint="--nudges",
        )
    v = nudges.strip().lower()
    if v == "true":
        nudges_used = True
    elif v == "false":
        nudges_used = False
    else:
        raise typer.BadParameter(
            f"--nudges must be 'true' or 'false', got {nudges!r}",
            param_hint="--nudges",
        )

    histogram = import_eval_dir(
        eval_dir,
        benchmark_id=benchmark_id,
        default_model=default_model,
        nudges_used=nudges_used,
    )
    console.print("[bold]import-eval[/] histogram:")
    for status, count in sorted(histogram.items()):
        console.print(f"  {status:24} {count}")


@app.command(name="import-vr-agent")
def import_vr_agent(
    path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Root of a vr-agent results tree (with scores/ and "
             "transcripts/ subdirectories). Typically lives under tmp/ "
             "and is gitignored — the dataset is sensitive and should "
             "never be checked in.",
    ),
    benchmark_id: str = typer.Option(
        "imported-vr-agent", "--benchmark-id",
        help="Benchmark id to attach to the imported rows. Default keeps "
             "vr-agent imports separate from live `v8` sweeps so the two "
             "appear as distinct columns in `qed_swe_bench summary`.",
    ),
) -> None:
    """Ingest the legacy vr-agent results tree.

    Reads `<path>/transcripts/<model>/<bug>/rep<N>/result.json` per run
    and `<path>/scores/manifest.csv` for the `excluded` flag. Inserts
    one row per (model, env, seed) cell with provenance='imported_from_legacy'
    so these runs are distinguishable from native sweeps and the
    bench-v8 imported-opus historicals.

    The source tree is treated as read-only — nothing under <path> is
    modified or copied into runs/. The DB row's `run_dir` column points
    back at the source path under tmp/ for traceability, but that
    sensitive path never flows into the public snapshot
    (`scripts/build_public_snapshot.py` aggregates cells without
    emitting per-run paths).

    Idempotent: re-runs against the same source are a no-op for
    already-imported rows.
    """
    from qed_swe_bench.historical import import_legacy_vr_agent_dir

    histogram = import_legacy_vr_agent_dir(path, benchmark_id=benchmark_id)
    console.print(f"[bold]import-vr-agent[/] from {path}:")
    for status, count in sorted(histogram.items()):
        console.print(f"  {status:24} {count}")


@app.command(name="import")
def import_cmd(
    path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Run-dir tree to import (e.g. ./runs/, /mnt/shared/runs/, "
             "or a path you've rsync'd from a remote host).",
    ),
) -> None:
    """Walk a run-dir tree and import every native run into the runs DB.

    The bijection contract (D-10): every run-dir is self-contained
    flat text. This command rebuilds the DB from the filesystem.
    Idempotent — re-imports of the same logical (benchmark_id, model,
    env_id, seed) tuple are silently skipped.

    Multi-host workflow:
      # On each host, after benchmark cells finish:
      aws s3 sync runs/ s3://qed_swe_bench-runs/<host>/

      # On the central machine, after rsync from S3 / NFS / scp:
      qed_swe_bench import ./runs-from-all-hosts/
      qed_swe_bench summary
    """
    from qed_swe_bench.historical import import_native_runs

    histogram = import_native_runs(path)
    console.print(f"[bold]import[/] from {path}:")
    for status, count in sorted(histogram.items()):
        console.print(f"  {status:24} {count}")


@app.command(name="export")
def export_cmd(
    target: Path = typer.Argument(
        ...,
        file_okay=False,
        help="Target root directory; per-run dirs are written under "
             "<target>/<benchmark_id>/exported/<run_id>/.",
    ),
    benchmark_id: str | None = typer.Option(
        None, "--benchmark-id",
        help="Filter to one benchmark; omit to export all rows missing a run-dir.",
    ),
    all_rows: bool = typer.Option(
        False, "--all",
        help="Export all matching rows even when their run-dir already "
             "exists on disk. Use to refresh artifacts after a writer "
             "format change.",
    ),
) -> None:
    """Write flat-text artifacts for DB rows whose run-dir is missing.

    Symmetric to `import` (D-10 bijection): import reads run-dirs into
    the DB; export writes DB rows back out as run-dirs. By default
    only fills gaps — rows that already have a populated run-dir on
    disk are skipped — which makes export idempotent for the common
    case.
    """
    from qed_swe_bench.historical import export_native_runs

    histogram = export_native_runs(
        target,
        benchmark_id=benchmark_id,
        only_missing_run_dir=not all_rows,
    )
    console.print(f"[bold]export[/] to {target}:")
    for status, count in sorted(histogram.items()):
        console.print(f"  {status:24} {count}")


@app.command(name="rerun")
def rerun_cmd(
    run_id: str = typer.Argument(
        ...,
        help="The 16-hex run_id to rerun (e.g. from `qed_swe_bench summary` "
             "or the `runs.run_id` column).",
    ),
    cost_cap_usd: float | None = typer.Option(
        None, "--cost-cap-usd",
        help="Override the snapshot's cost_cap_usd for this re-run. Defaults "
             "to the original config's value.",
    ),
    max_parallel: int = typer.Option(
        1, "--max-parallel",
        help="Concurrency for the (always single-tuple) re-run. Default 1 — "
             "you're rerunning one cell, not a sweep.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print the resolved single-tuple BenchmarkConfig and exit "
             "without launching docker / the LLM.",
    ),
) -> None:
    """Re-run a single (model, env, seed) cell from its DB row.

    Reads `runs.config_snapshot` (the verbatim source YAML the row was
    produced under, comments preserved — D-13), narrows to the row's
    (model, env, seed) tuple, and replays through the same code path
    as `qed_swe_bench benchmark`. The DB row is self-sufficient: the
    original `<run_dir>/config_snapshot.yaml` on disk is not required.

    Falls back to reading `<run_dir>/config_snapshot.yaml` for legacy
    rows where the column is NULL (rows imported from run-dirs that
    predate the column).

    NOT a deterministic replay: each invocation produces a fresh
    episode through the live LLM. For deterministic tool-call replay
    against the recorded grader, use `qed_swe_bench audit --reproduce`.
    """
    import asyncio
    import tempfile

    import yaml as _yaml

    from qed_swe_bench.db.schema import transaction
    from qed_swe_bench.runner.orchestrator import (
        parse_config,
        run_benchmark,
    )

    with transaction() as con:
        row = con.execute(
            "SELECT model, env_id, seed, run_dir, config_snapshot, "
            "provenance, repro_cmd "
            "FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
    if row is None:
        console.print(f"[red]reproduce: no run with run_id={run_id!r}[/]")
        raise typer.Exit(code=1)

    # Refuse non-native rows. Imported rows (vr-agent, codex, …) came
    # from foreign agent harnesses; replaying them through qed_swe_bench's
    # loop would produce a structurally different episode. The right
    # rerun path lives in `repro_cmd` for these rows.
    provenance = row["provenance"] or "native"
    if provenance.startswith("imported_from_"):
        repro_cmd = row["repro_cmd"]
        console.print(
            f"[red]reproduce: row {run_id!r} has provenance={provenance!r}; "
            f"qed_swe_bench's native loop can't faithfully reproduce a run "
            f"from a foreign agent harness.[/]"
        )
        if repro_cmd:
            console.print(f"  rerun via: [bold]{repro_cmd}[/]")
        else:
            console.print(
                "  no repro_cmd recorded — see the originating importer "
                "(scripts/import_codex_bench.py, "
                "scripts/import-vr-agent, …) for the right rerun path."
            )
        raise typer.Exit(code=1)

    snapshot = row["config_snapshot"]
    if snapshot is None:
        # Legacy fallback: pre-D-13 rows didn't populate the column;
        # read the on-disk artifact instead. Rows that have neither
        # are unreproducible.
        legacy_path = (
            Path(row["run_dir"]) / "config_snapshot.yaml"
            if row["run_dir"] else None
        )
        if legacy_path is not None and legacy_path.is_file():
            snapshot = legacy_path.read_text(encoding="utf-8")
            console.print(
                f"[yellow]reproduce: row has no config_snapshot column "
                f"(legacy); falling back to {legacy_path}[/]"
            )
        else:
            console.print(
                f"[red]reproduce: run_id={run_id!r} has no config_snapshot "
                f"column and no on-disk fallback at "
                f"<run_dir>/config_snapshot.yaml. This row is not "
                f"reproducible.[/]"
            )
            raise typer.Exit(code=1)

    # parse_config takes a dict; route the YAML through safe_load.
    try:
        config_dict = _yaml.safe_load(snapshot)
        bench = parse_config(config_dict)
    except (_yaml.YAMLError, ValueError) as exc:
        console.print(f"[red]reproduce: snapshot YAML is malformed: {exc}[/]")
        raise typer.Exit(code=1) from exc

    # Narrow to this row's (model, env, seed) tuple. Each filter must
    # find a match in the snapshot — if the snapshot was stored with
    # the same orchestrator that wrote the row, this is guaranteed.
    matching_models = [m for m in bench.models if m.id == row["model"]]
    matching_envs = [e for e in bench.envs if e.id == row["env_id"]]
    if not matching_models:
        console.print(
            f"[red]reproduce: snapshot doesn't declare model={row['model']!r}; "
            f"available={[m.id for m in bench.models]}[/]"
        )
        raise typer.Exit(code=1)
    if not matching_envs:
        console.print(
            f"[red]reproduce: snapshot doesn't declare env={row['env_id']!r}; "
            f"available={[e.id for e in bench.envs]}[/]"
        )
        raise typer.Exit(code=1)

    overrides: dict[str, object] = {
        "models": matching_models,
        "envs": matching_envs,
        "seeds": [int(row["seed"])],
        "max_parallel": max_parallel,
    }
    if cost_cap_usd is not None:
        overrides["cost_cap_usd"] = cost_cap_usd
    bench = bench.with_overrides(**overrides)

    console.print(
        f"[bold]reproduce[/] run_id={run_id} "
        f"model={row['model']} env={row['env_id']} seed={row['seed']}"
    )

    if dry_run:
        console.print("\n[bold]resolved BenchmarkConfig (dry-run):[/]")
        console.print(f"  benchmark_id={bench.benchmark_id}")
        console.print(f"  models={[m.id for m in bench.models]}")
        console.print(f"  envs={[e.id for e in bench.envs]}")
        console.print(f"  seeds={bench.seeds}")
        console.print(f"  budgets={bench.budgets}")
        console.print(f"  nudges={sorted(n.value for n in bench.nudges)}")
        console.print(f"  cost_cap_usd={bench.cost_cap_usd}")
        console.print(f"  max_parallel={bench.max_parallel}")
        return

    # Materialize the snapshot to a tempfile so the orchestrator can
    # use the same `config_path` plumbing it normally does (which
    # writes `<run_dir>/config_snapshot.yaml` per D-10). The tempfile
    # is the "config the operator handed us" in the eyes of the
    # orchestrator — no special-casing.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    ) as tmp:
        tmp.write(snapshot)
        tmp_path = Path(tmp.name)
    try:
        histogram = asyncio.run(
            run_benchmark(
                bench,
                config_path=tmp_path,
            ),
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    console.print("\n[bold]reproduce done.[/] status histogram:")
    for status, count in sorted(histogram.items()):
        console.print(f"  {status:20} {count}")


@app.command(name="migrate-runs")
def migrate_runs_cmd(
    runs_root: Path = typer.Argument(
        Path("runs"),
        exists=True,
        file_okay=False,
        help="Root of the runs/ tree to migrate (default: ./runs).",
    ),
    legacy_host: str = typer.Option(
        "legacy", "--legacy-host",
        help="Host name to assign to runs whose original host is "
             "unknown (the typical case for pre-migration runs).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print what would move without touching disk or DB.",
    ),
) -> None:
    """One-shot migrate legacy run-dirs to the layered D-10 layout.

    Moves runs/<benchmark_id>/<old-shape>/ into
    runs/<benchmark_id>/<host>/<datetime>/<run_id>/. Two-phase: copy
    new → update DB → remove old, so a crash mid-migration leaves
    consistent state.

    Refuses to run if any DB row is queued with a recent heartbeat —
    sibling orchestrators are still working. Wait for them to finish
    or stop them, then re-run.
    """
    from qed_swe_bench.historical import migrate_runs_to_layered_layout

    histogram = migrate_runs_to_layered_layout(
        runs_root,
        legacy_host=legacy_host,
        dry_run=dry_run,
    )
    label = "[dim](dry run)[/]" if dry_run else ""
    console.print(f"[bold]migrate-runs[/] {label}")
    for status, count in sorted(histogram.items()):
        if count > 0:
            console.print(f"  {status:24} {count}")


@app.command(name="validate-image")
def validate_image_cmd(
    env_id: str | None = typer.Option(
        None, "--env-id",
        help="Catalog env_id; manifest path is read from rlenv_images.metadata.manifest_path.",
    ),
    manifest: Path | None = typer.Option(
        None, "--manifest",
        help="Path to a manifest YAML/JSON. Required if --env-id not given.",
    ),
    image_ref: str | None = typer.Option(
        None, "--image-ref",
        help="Override the manifest's image.ref (e.g., to validate a different tag).",
    ),
    skip_container: bool = typer.Option(
        False, "--skip-container",
        help="Run only the manifest_schema check; skip the four container checks.",
    ),
    update_status: bool = typer.Option(
        True, "--update-status/--no-update-status",
        help="Persist the resulting overall status to rlenv_images.validation_status.",
    ),
) -> None:
    """Run the 5-check validator suite against a registered env or a standalone manifest."""
    import asyncio
    from datetime import UTC, datetime

    from qed_swe_bench.validator import run_all
    from qed_swe_bench.validator.runner import CheckStatus

    # Resolve manifest path: explicit --manifest wins; else look up by env_id.
    if manifest is None and env_id is None:
        console.print("[red]error:[/] either --manifest or --env-id is required")
        raise typer.Exit(code=2)

    if manifest is None:
        # Look up the manifest path the catalog recorded for this env.
        with connect() as con:
            row = con.execute(
                "SELECT metadata FROM rlenv_images WHERE env_id=?", (env_id,)
            ).fetchone()
        if row is None:
            console.print(f"[red]error:[/] env_id={env_id!r} not in rlenv_images")
            raise typer.Exit(code=2)
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        mp = meta.get("manifest_path")
        if not mp:
            console.print(
                f"[red]error:[/] env_id={env_id!r} has no manifest_path in metadata; "
                "pass --manifest explicitly"
            )
            raise typer.Exit(code=2)
        manifest = Path(mp)

    report = asyncio.run(
        run_all(
            manifest_path=str(manifest) if manifest else None,
            env_id=env_id,
            image_ref=image_ref,
            skip_container_checks=skip_container,
        )
    )

    table = Table(title=f"validate-image  env={report.env_id}  image={report.image_ref}")
    table.add_column("check")
    table.add_column("status")
    table.add_column("message", overflow="fold")
    for r in report.results:
        color = {
            CheckStatus.PASS: "green",
            CheckStatus.FAIL: "red",
            CheckStatus.SKIP: "dim",
            CheckStatus.UNVERIFIED: "yellow",
        }[r.status]
        table.add_row(r.name, f"[{color}]{r.status.value}[/]", r.message)
    console.print(table)
    console.print(f"[bold]overall:[/] {report.overall_status}")

    if update_status and env_id:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with connect() as con:
            con.execute(
                "UPDATE rlenv_images SET validation_status=?, last_validated_at=? "
                "WHERE env_id=?",
                (report.overall_status, now, env_id),
            )

    if report.overall_status == "fail":
        raise typer.Exit(code=1)


# Allow `python -m qed_swe_bench.cli`
if __name__ == "__main__":
    app()
