from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from models.schemas import TestResult

SANDBOX_IMAGE = "codegen-sandbox"
TIMEOUT_SECONDS = 60
MEMORY_LIMIT = "256m"
REPORT_FILENAME = "report.json"


@dataclass
class CodeFile:
    """A single file to be written into the sandbox workspace."""

    filename: str
    content: str


def run_in_sandbox(
    files: list[CodeFile],
    requirements: list[str] | None = None,
) -> TestResult:
    """Write files to a temp directory, run pytest in the Docker sandbox, return TestResult.

    The container is started with --network=none and a 256 MB memory cap. A
    ``requirements.txt`` is written into the workspace when ``requirements`` is
    provided; the container's entrypoint installs it before running pytest.

    Structured JSON output (``pytest-json-report``) is preferred for parsing;
    the function falls back to stdout parsing when the report file is absent or
    corrupt.

    V2 evolution metadata fields (``generation``, ``test_code``) are left as
    ``None``; callers should add them with
    ``result.model_copy(update={...})``.

    Args:
        files: Source and test files to write into ``/workspace``.
        requirements: Optional pip package names to install before running tests.

    Returns:
        A ``TestResult`` with pass/fail counts, errors, and per-test breakdown.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for f in files:
            dest = tmp / f.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(f.content)

        if requirements:
            (tmp / "requirements.txt").write_text("\n".join(requirements))

        try:
            proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--memory",
                    MEMORY_LIMIT,
                    "--memory-swap",
                    MEMORY_LIMIT,  # disable swap to enforce the cap
                    "-v",
                    f"{tmpdir}:/workspace",
                    SANDBOX_IMAGE,
                ],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                passed=False,
                errors=[
                    f"Sandbox timed out after {TIMEOUT_SECONDS}s — "
                    "possible infinite loop in generated code."
                ],
                output="",
            )

        report_path = tmp / REPORT_FILENAME
        if report_path.exists():
            return _parse_json_report(report_path, proc.stdout, proc.stderr)

        return _parse_stdout(proc.returncode, proc.stdout, proc.stderr)


# ── Internal parsers ──────────────────────────────────────────────────────────


def _parse_json_report(report_path: Path, stdout: str, stderr: str) -> TestResult:
    """Parse a ``pytest-json-report`` file into a ``TestResult``.

    Falls back to ``_parse_stdout`` if the report is missing or malformed.
    """
    try:
        data = json.loads(report_path.read_text())
    except (json.JSONDecodeError, OSError):
        return _parse_stdout(1, stdout, stderr)

    summary = data.get("summary", {})
    total: int = summary.get("total", 0)
    passed: int = summary.get("passed", 0)
    # count both outright failures and collection/setup errors
    failed: int = summary.get("failed", 0) + summary.get("error", 0)

    per_test: list[dict] = [
        {
            "name": t.get("nodeid", ""),
            "passed": t.get("outcome") == "passed",
        }
        for t in data.get("tests", [])
    ]

    errors: list[str] = []
    if stderr.strip():
        errors.append(stderr.strip())

    return TestResult(
        passed=failed == 0 and total > 0,
        total_tests=total,
        passed_tests=passed,
        failed_tests=failed,
        errors=errors,
        output=stdout,
        per_test_results=per_test or None,
    )


def _parse_stdout(exit_code: int, stdout: str, stderr: str) -> TestResult:
    """Parse pytest verbose stdout into a ``TestResult`` (fallback path)."""
    total, passed, failed = _extract_counts(stdout)
    per_test = _extract_per_test(stdout) or None

    errors: list[str] = []
    if stderr.strip():
        errors.append(stderr.strip())
    if exit_code != 0 and stdout.strip():
        errors.append(stdout.strip())

    return TestResult(
        passed=exit_code == 0,
        total_tests=total,
        passed_tests=passed,
        failed_tests=failed,
        errors=errors,
        output=stdout,
        per_test_results=per_test,
    )


def _extract_counts(stdout: str) -> tuple[int, int, int]:
    """Return (total, passed, failed) from a pytest summary line."""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", stdout)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", stdout)) else 0
    return passed + failed, passed, failed


def _extract_per_test(stdout: str) -> list[dict]:
    """Parse per-test PASSED / FAILED lines from pytest ``-v`` output."""
    results: list[dict] = []
    for line in stdout.splitlines():
        if " PASSED" in line:
            results.append({"name": line.split(" PASSED")[0].strip(), "passed": True})
        elif " FAILED" in line:
            results.append({"name": line.split(" FAILED")[0].strip(), "passed": False})
    return results
