# Contributing

Thanks for your interest. PRs are welcome — small fixes, new model adapters,
new benchmark environments, doc improvements, all of it.

## Before you open a PR

For anything beyond a typo or a small fix, please open an issue first so
we can agree on the approach. This avoids the situation where you put real
work into a change we can't merge for reasons that aren't obvious from the
code.

## Local setup

```bash
make install                 # creates .venv/, editable install
source .venv/bin/activate
qed_swe_bench doctor          # sanity-check env, docker, deps
```

## Before you push

```bash
make lint                    # ruff
make test                    # pytest (fast suite)
make smoke                   # end-to-end mock-LLM run
```

The `slow` and `golden` pytest markers cover tests that need Docker, real
APIs, or significant wall-clock. You don't need to run them for most PRs;
CI does not currently run them either.

## Style

- `ruff` settings live in `pyproject.toml`; let it format and lint.
- Match the style of the file you are editing rather than introducing a new
  pattern in one place.
- Keep commit messages descriptive. We don't enforce conventional commits.

## New benchmark environments

See [`benchmarks/README.md`](./benchmarks/README.md) for the layout and
invocation conventions. New V8 bugs should follow the
the `v8-eNN` env-id scheme and ship the same artifacts as the existing ones.


## License

By contributing you agree that your contribution is licensed under the MIT
License (see [`LICENSE`](./LICENSE)).
