"""Transcript red-flag scanner for qed_swe_bench runs.

Each check is a pure function that walks a run dir's JSONL artifacts
(`tool_calls.jsonl`, `transcript.jsonl`, `grade_calls.jsonl`) and
returns a list of `Finding`s. Findings collect into an `AuditReport`
that the CLI formats as a table or JSON.

The catalog is grep-shaped on purpose — checks must themselves be
auditable, so we don't use LLM-based "does this look weird" judgments.
See the per-check docstring for what each one looks for.
"""

from qed_swe_bench.audit.reproduce import (
    ComparisonResult,
    ReproductionReport,
    reproduce_run,
    reproduce_run_sync,
)
from qed_swe_bench.audit.transcripts import (
    AuditReport,
    Finding,
    Severity,
    audit_run,
    audit_runs,
)

__all__ = [
    "AuditReport",
    "ComparisonResult",
    "Finding",
    "ReproductionReport",
    "Severity",
    "audit_run",
    "audit_runs",
    "reproduce_run",
    "reproduce_run_sync",
]
