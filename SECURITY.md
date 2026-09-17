# Security Policy

## Scope

`qed_swe_bench` is research code for evaluating language models against
historical V8 bug environments. It is intended to be run **locally** on a
researcher's own machine (or a disposable VM) against the bundled benchmark
containers. It is not a hosted service, not a production system, and has no
multi-tenant or trust-boundary properties to defend.

## What this repo is not

- It is not a deployed application; it exposes no network services.
- It is not under a bug bounty program. We do not offer rewards for reports
  against this codebase, and we ask that researchers not treat it as an
  in-scope target for one.
- The benchmark environments deliberately contain known-vulnerable builds of
  V8 (with public CVEs). Work against those builds are the *subject* of
  the benchmark, not vulnerabilities in this project.

## Reporting

If you have spotted a real defect — for example, the harness writing outside
its working directory, a credential being logged, a dependency with a known
CVE we should pin away from, or anything else a user running this locally
should know about — please open a regular GitHub issue, or a PR with a fix.
Pull requests are very welcome.

For anything you would prefer not to discuss in public, email
`contact@qed-audit.example`.

## Supported versions

Only `main` is supported. There are no backports.
