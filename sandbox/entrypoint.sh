#!/bin/sh
# Sandbox entrypoint: optionally install dependencies, then run pytest.
#
# If /workspace/requirements.txt exists (written by runner.py), install the
# packages before running the test suite.  The --no-cache-dir and -q flags
# keep output clean and avoid bloating the container layer.

set -e

if [ -f /workspace/requirements.txt ]; then
    pip install --no-cache-dir -q -r /workspace/requirements.txt
fi

exec pytest -v --tb=short \
    --json-report \
    --json-report-file=/workspace/report.json \
    --timeout=55 \
    "$@"
