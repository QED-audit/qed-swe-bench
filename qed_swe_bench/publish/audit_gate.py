"""Pre-publish audit gate.

Re-runs `qed_swe_bench.audit.transcripts` against the run dirs we're
about to bundle. HIGH findings refuse to publish unless the operator
overrides with `--allow-high-audit`. MEDIUM/INFO findings are
summarized but never block — they exist to be visible.

The audit is a manual-review aid, not a definitive judgment of
cheating. The gate forces a human to triage HIGH findings before the
dataset ships; the human's read is what decides whether to override
or fix. False positives are expected (the checks are grep-shaped
substring scans on tool args; that's the price of keeping the audit
itself auditable).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from qed_swe_bench.audit import AuditReport, Severity, audit_runs


@dataclass(frozen=True)
class GateResult:
    """Aggregate of audit findings across the selected run dirs."""

    counts: dict[str, int] = field(default_factory=dict)
    high_run_ids: tuple[str, ...] = ()
    medium_run_ids: tuple[str, ...] = ()
    info_run_ids: tuple[str, ...] = ()
    reports: tuple[AuditReport, ...] = ()

    @property
    def has_high(self) -> bool:
        return self.counts.get("high", 0) > 0


def gate(run_dirs: Iterable[Path]) -> GateResult:
    """Run audit_runs on each dir; bucket findings by severity.

    Counts are total findings per severity (one run can contribute
    multiple). The `*_run_ids` tuples carry the unique run_ids that
    triggered at least one finding of that severity, so callers can
    print "3 MEDIUM (run_ids: ...)" without re-walking.
    """
    reports = audit_runs(list(run_dirs))
    counts = {"high": 0, "medium": 0, "info": 0}
    high: list[str] = []
    medium: list[str] = []
    info: list[str] = []
    for report in reports:
        seen_for_run: set[Severity] = set()
        for finding in report.findings:
            sev = finding.severity
            if sev is Severity.HIGH:
                counts["high"] += 1
                if Severity.HIGH not in seen_for_run:
                    high.append(report.run_id)
            elif sev is Severity.MEDIUM:
                counts["medium"] += 1
                if Severity.MEDIUM not in seen_for_run:
                    medium.append(report.run_id)
            elif sev is Severity.INFO:
                counts["info"] += 1
                if Severity.INFO not in seen_for_run:
                    info.append(report.run_id)
            seen_for_run.add(sev)
    return GateResult(
        counts=counts,
        high_run_ids=tuple(high),
        medium_run_ids=tuple(medium),
        info_run_ids=tuple(info),
        reports=tuple(reports),
    )
