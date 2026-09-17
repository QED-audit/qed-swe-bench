.PHONY: help install dev test test-slow test-golden test-all lint format clean db-init smoke summary doctor setup-hooks submodules resync-main audit audit-bundle runbook-local

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "qed_swe_bench targets:"
	@echo "  install       create venv + install package + dev deps"
	@echo "  test          unit + golden tests (no Docker, no API)"
	@echo "  test-slow     slow tests (Docker required, no API)"
	@echo "  test-golden   tier-2 parity tests vs imported-opus eval data"
	@echo "  test-all      run everything"
	@echo "  smoke         build sample env + run --mock-llm"
	@echo "  doctor        print env / docker / deps health check"
	@echo "  summary       per-benchmark spend / status table"
	@echo "  lint / format ruff check / format"
	@echo "  setup-hooks   point git at .githooks/ (refuses commits on main)"
	@echo "  resync-main   hard-align local main with origin/main"
	@echo "  audit-bundle  pack one benchmark's run-dirs + sha256 manifest"
	@echo "                into audit-bundles/<id>-<ts>.tar.gz."
	@echo "                usage: make audit-bundle BENCHMARK_ID=<id>"
	@echo "  runbook-local cp docs/RUNBOOK.md → RUNBOOK.local.md (gitignored)"
	@echo "  clean         delete venv + caches"

install: $(VENV) setup-hooks submodules
	$(PIP) install -e .[dev]
	@# Tighten .env permissions if it exists and is too-permissive (default
	@# umask creates 644 = world-readable, which leaks API keys to any
	@# local user). Idempotent: chmod 600 is a no-op when already 600.
	@if [ -f .env ]; then \
	    chmod 600 .env && echo "tightened .env permissions to 600"; \
	fi

# Pull bench-v8 (and any future submodules). Idempotent — safe to re-run.
submodules:
	@git submodule update --init --recursive

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

dev: install

db-init:
	$(PY) -c "from qed_swe_bench.db.schema import init_db; init_db()"

test:
	$(VENV)/bin/pytest -q -m "not slow"

test-slow:
	$(VENV)/bin/pytest -q -m "slow"

test-golden:
	$(VENV)/bin/pytest -q -m "golden"

test-all:
	$(VENV)/bin/pytest -q

lint:
	$(VENV)/bin/ruff check qed_swe_bench

format:
	$(VENV)/bin/ruff format qed_swe_bench

smoke: $(VENV)
	bash scripts/build_sample_env.sh
	$(VENV)/bin/qed_swe_bench benchmark --mock-llm

summary:
	$(VENV)/bin/qed_swe_bench summary

doctor:
	$(VENV)/bin/qed_swe_bench doctor

# Bootstrap a personal RUNBOOK.local.md from the canonical methodology doc.
# RUNBOOK.local.md is gitignored (*.local.md) — your checkboxes, in-flight
# notes, and operator state stay out of commits. Refresh as needed when
# docs/RUNBOOK.md is updated.
runbook-local:
	@if [ -f RUNBOOK.local.md ]; then \
		echo "RUNBOOK.local.md already exists — refusing to overwrite."; \
		echo "Delete it first if you want to refresh from docs/RUNBOOK.md."; \
		exit 1; \
	fi
	@cp docs/RUNBOOK.md RUNBOOK.local.md
	@echo "Created RUNBOOK.local.md (gitignored). Check items off as you go."

clean:
	rm -rf $(VENV) build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

setup-hooks:
	@# Copy into .git/hooks/ rather than setting core.hooksPath so the
	@# hook keeps working when you `git checkout main` (which doesn't
	@# carry the .githooks/ tree until the hook lands on main itself).
	@mkdir -p .git/hooks
	@cp .githooks/pre-commit .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@# Clear any prior hooksPath override so the static install wins.
	@git config --unset-all core.hooksPath 2>/dev/null || true
	@echo "pre-commit hook installed into .git/hooks/ — refuses commits on main"

# Hard-align local main with origin/main. Safe because per CLAUDE.md we
# never commit to main directly; any divergent commits are obsolete
# pre-rebase predecessors of work that landed on origin/main with new SHAs.
audit:
	@if [ -z "$(BENCHMARK_ID)" ] && [ -z "$(RUN_ID)" ]; then \
	    echo "✗ usage: make audit BENCHMARK_ID=<id>   # audit every run in a benchmark"; \
	    echo "          or: make audit RUN_ID=<id>      # audit one run"; \
	    exit 2; \
	fi
	@if [ -n "$(BENCHMARK_ID)" ]; then \
	    $(VENV)/bin/qed_swe_bench audit --benchmark-id "$(BENCHMARK_ID)" --detail; \
	else \
	    $(VENV)/bin/qed_swe_bench audit --run-id "$(RUN_ID)" --detail; \
	fi

audit-bundle:
	@if [ -z "$(BENCHMARK_ID)" ]; then \
	    echo "✗ usage: make audit-bundle BENCHMARK_ID=<id>"; \
	    echo "  list benchmarks: sqlite3 data/qed_swe_bench.sqlite 'SELECT DISTINCT benchmark_id FROM runs'"; \
	    exit 2; \
	fi
	bash scripts/build_audit_bundle.sh "$(BENCHMARK_ID)"

resync-main:
	@# POSIX `[`, not bash `[[`, so this works under dash (Debian /bin/sh).
	@if [ "$$(git symbolic-ref --short -q HEAD)" = "main" ]; then \
		echo "✗ checkout a feature branch first; resync-main can't run while ON main."; \
		exit 1; \
	fi
	git fetch origin
	git branch -f main origin/main
	@echo "local main → $$(git rev-parse --short main) (matches origin/main)"
