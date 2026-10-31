#!/usr/bin/env python3
"""Exception-Driven Frontier Debugger Gate.

Runs `pytest --tb=short -q` deterministically (0 tokens).
- If exit code == 0: prints confirmation and exits cleanly with 0 frontier tokens spent.
- If exit code != 0: packages failing test names, tracebacks, and relevant source files
  into `.debug_bundle.md` and optionally invokes a frontier CLI (`claude -p` or `openai`).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / ".debug_bundle.md"

_FAILED_TEST_RE = re.compile(r"^FAILED\s+([^\s]+)", re.MULTILINE)
_FILE_REF_RE = re.compile(r"((?:src|tests)/[a-zA-Z0-9_./\-]+\.py)")


def extract_failing_tests(pytest_output: str) -> List[str]:
    """Extract failing test node IDs from pytest short traceback output."""
    found = _FAILED_TEST_RE.findall(pytest_output)
    return list(dict.fromkeys(found))


def extract_referenced_files(pytest_output: str, root: Path = PROJECT_ROOT) -> List[Path]:
    """Extract existing project source/test files referenced in tracebacks."""
    matches = _FILE_REF_RE.findall(pytest_output)
    resolved: List[Path] = []
    seen: Set[Path] = set()
    for rel_str in matches:
        candidate = (root / rel_str).resolve()
        if candidate.is_file() and candidate not in seen:
            seen.add(candidate)
            resolved.append(candidate)
    return resolved


def build_debug_bundle(
    pytest_output: str,
    exit_code: int,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Create a structured diagnostic prompt file (`.debug_bundle.md`) on test failure."""
    failing_tests = extract_failing_tests(pytest_output)
    referenced_files = extract_referenced_files(pytest_output, root=root)

    sections: List[str] = [
        "# Exception-Driven Frontier Diagnostic Bundle",
        f"- **Generated At:** {datetime.now(timezone.utc).isoformat()}",
        f"- **Pytest Exit Code:** `{exit_code}`",
        f"- **Failing Tests ({len(failing_tests)}):**",
    ]
    if failing_tests:
        for test_node in failing_tests:
            sections.append(f"  - `{test_node}`")
    else:
        sections.append("  - `(Collection or runtime error — see traceback below)`")

    sections.extend(
        [
            "",
            "## 1. Pytest Short Traceback Output",
            "```text",
            pytest_output.strip(),
            "```",
            "",
            "## 2. Relevant Source Files",
        ]
    )

    for file_path in referenced_files:
        rel_path = file_path.relative_to(root)
        content = file_path.read_text(encoding="utf-8", errors="replace")
        sections.extend(
            [
                f"### `{rel_path}`",
                "```python",
                content.strip(),
                "```",
                "",
            ]
        )

    sections.extend(
        [
            "## 3. Repair Instructions for Frontier Debugger",
            "Identify the root cause of the failing deterministic/contract test(s) above and provide "
            "a minimal, surgical fix that preserves all Pydantic V2 contracts and security invariants.",
            "",
        ]
    )

    bundle_path.write_text("\n".join(sections), encoding="utf-8")
    return bundle_path


def invoke_frontier_cli(bundle_path: Path, cli_provider: str) -> int:
    """Optionally invoke `claude -p` or `openai` CLI with the diagnostic bundle."""
    prompt_text = bundle_path.read_text(encoding="utf-8")
    if cli_provider == "claude":
        binary = shutil.which("claude")
        if not binary:
            print("Frontier CLI 'claude' not found in PATH; skipping remote call.", file=sys.stderr)
            return 1
        proc = subprocess.run([binary, "-p", prompt_text], check=False)
        return proc.returncode
    if cli_provider == "openai":
        binary = shutil.which("openai")
        if not binary:
            print("Frontier CLI 'openai' not found in PATH; skipping remote call.", file=sys.stderr)
            return 1
        proc = subprocess.run(
            [binary, "api", "chat.completions.create", "-m", "gpt-4o", "-g", "user", prompt_text],
            check=False,
        )
        return proc.returncode
    return 1


def run_test_gate(
    pytest_args: Iterable[str] | None = None,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    frontier_cli: str | None = None,
) -> int:
    """Execute pytest and only build/invoke frontier diagnostics if exit code != 0."""
    cmd = [sys.executable, "-m", "pytest", "--tb=short", "-q"]
    if pytest_args:
        cmd.extend(pytest_args)

    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")

    if completed.returncode == 0:
        print("All deterministic & contract tests passed. Frontier tokens spent: 0.")
        return 0

    written_path = build_debug_bundle(
        pytest_output=combined_output,
        exit_code=completed.returncode,
        bundle_path=bundle_path,
    )
    print(
        f"Tests failed (exit code {completed.returncode}). "
        f"Diagnostic bundle written to: {written_path}",
        file=sys.stderr,
    )

    if frontier_cli in ("claude", "openai"):
        invoke_frontier_cli(written_path, frontier_cli)

    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pytest suite and only package failure tracebacks for a frontier model on failure."
    )
    parser.add_argument(
        "--frontier-cli",
        choices=["claude", "openai"],
        default=None,
        help="Optionally invoke a frontier CLI with .debug_bundle.md when pytest fails.",
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=DEFAULT_BUNDLE_PATH,
        help="Output path for the failure diagnostic markdown bundle.",
    )
    args, extra_pytest_args = parser.parse_known_args()
    return run_test_gate(
        pytest_args=extra_pytest_args,
        bundle_path=args.bundle_path,
        frontier_cli=args.frontier_cli,
    )


if __name__ == "__main__":
    sys.exit(main())

