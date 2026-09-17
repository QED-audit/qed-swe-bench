"""Exception classification for the runner's safety wrap.

When something escapes the per-tuple body, we have to decide:
  - Is this an *infra* problem (docker / network / MCP / LLM provider)?
    The operator should retry or fix their environment.
  - Is this a *bug* in our codebase (FileNotFoundError, KeyError, …)?
    Someone should read the traceback and ship a fix.

The CLI uses this distinction to surface bugs under a separate "CODE BUGS"
banner so they don't hide in the same bucket as transient infra failures.

Two helpers serve different consumers:
  - `format_failure(exc)` — used by the in-body crash path. Returns a
    `crashed: <Type>` exit_reason regardless of category, since the body
    catches at one level for either kind. Predates the bug/infra split.
  - `classify(exc)` and `is_infra_exception(exc)` — used by the outer
    safety wrap. Returns a category prefix (`infra` or `bug`) so the
    exit_reason can encode the distinction.

If you reach for `format_failure` from new code, prefer `classify` — same
information plus the kind. We may unify these eventually.
"""

from __future__ import annotations

import logging

from qed_swe_bench.runner.cli_oneshot_grader import GraderError
from qed_swe_bench.runner.image_ref import ImageRefError

log = logging.getLogger(__name__)


# Exception types we expect from a healthy codebase running against unhealthy
# infra: docker, network, MCP, the LLM provider. An exception of one of these
# types means "the world misbehaved," not "we have a bug."
#
# Keep this list conservative: when in doubt, treat the exception as a code
# bug so the operator gets a loud signal. False positives (real infra issues
# labeled as bugs) are recoverable; false negatives (real bugs labeled as
# infra) hide in the histogram for weeks.
INFRA_EXC_TYPES: tuple[type[BaseException], ...] = (
    TimeoutError,         # asyncio.wait_for, episode_timeout_s, network
    ConnectionError,      # ConnectionResetError, ConnectionRefusedError, etc.
    BrokenPipeError,      # subprocess pipe died (docker / MCP container)
    ImageRefError,        # docker pull / digest resolve
    GraderError,          # cli_oneshot grader failures (already handled,
                          # but listed for outer-wrap fallthrough)
)

# Module name prefixes whose exceptions are always treated as infra.
# httpx / litellm / mcp / anthropic SDKs all live under their own top-level
# modules, so we can duck-type by module without importing each provider's
# exception hierarchy.
_INFRA_MODULES: frozenset[str] = frozenset({"httpx", "litellm", "mcp", "anthropic"})


def unwrap_exception_group(exc: BaseException) -> BaseException:
    """Return the leaf cause inside (possibly nested) BaseExceptionGroups.

    anyio's TaskGroup (and `asyncio.TaskGroup`) wrap any exception
    raised in a child task into a `BaseExceptionGroup`. Python's
    default `BaseExceptionGroup.__str__` collapses to "unhandled
    errors in a TaskGroup (N sub-exceptions)", which is useless for
    diagnosing what actually went wrong. This helper walks the
    `.exceptions` list to surface the real leaf cause for the failure
    record's `exit_reason` / `failure_reason`. If multiple leaves
    exist, the first is returned and a note is logged so the audit
    trail mentions there were siblings.
    """
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        if len(exc.exceptions) > 1:
            log.warning(
                "ExceptionGroup with %d sub-exceptions; reporting the first: %s",
                len(exc.exceptions),
                [type(e).__name__ for e in exc.exceptions],
            )
        exc = exc.exceptions[0]
    return exc


def format_failure(exc: BaseException) -> tuple[str, str]:
    """`(exit_reason, failure_reason)` for a caught mid-episode exception.

    Unwraps ExceptionGroups so the recorded reason names the actual
    leaf cause, not the TaskGroup wrapper. The wrapper's class name is
    preserved as a `(via TaskGroup)` suffix when relevant.
    """
    inner = unwrap_exception_group(exc)
    inner_msg = str(inner) or repr(inner)
    via = f" (via {type(exc).__name__})" if inner is not exc else ""
    return f"crashed: {type(inner).__name__}", f"{inner_msg}{via}"


def is_infra_exception(exc: BaseException) -> bool:
    """Classify an exception as 'infra problem' vs 'likely code bug'.

    Returns True if the exception looks like it came from outside our code
    (docker, network, MCP, LLM provider), False if it looks like a bug in
    qed_swe_bench. Used to pick `infra_<Type>` vs `bug_<Type>` exit_reason
    prefixes so the CLI summary can surface bugs prominently.
    """
    if isinstance(exc, INFRA_EXC_TYPES):
        return True
    module = type(exc).__module__ or ""
    return module.split(".", 1)[0] in _INFRA_MODULES
