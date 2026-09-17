"""Export the FastAPI OpenAPI spec to stdout.

Used by `webui/scripts/generate-api-client.mjs` so the dashboard's
typed fetch client can be regenerated without a running API server.

Run:
    .venv/bin/python -m qed_swe_bench.api.export_openapi > webui/openapi.json
"""

from __future__ import annotations

import json
import sys

from qed_swe_bench.api.app import app


def main() -> None:
    spec = app.openapi()
    json.dump(spec, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
