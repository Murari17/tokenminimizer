#!/usr/bin/env python3
"""Exception-Driven Frontier Debugger Gate.

Runs `pytest --tb=short -q` deterministically (0 tokens).
- If exit code == 0: prints confirmation and exits cleanly with 0 frontier tokens spent.
- If exit code != 0: packages failing test names, tracebacks, and relevant source files
  into `.debug_bundle.md`, pipes the bundle over `stdin` (avoiding OS `ARG_MAX` limits)
  to a frontier CLI when requested, and can optionally validate & apply unified diff patches.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / ".debug_bundle.md"

_FAILED_TEST_RE = re.compile(r"^FAILED\s+([^\s]+)", re.MULTILINE)
_FILE_REF_RE = re.compile(r"((?:src|tests)/[a-zA-Z0-9_./\-]+\.py)")
_DIFF_BLOCK_RE = re.compile(r"```(?:diff|patch)\n(.*?)```", re.DOTALL)


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


def extract_unified_diff(frontier_output: str) -> Optional[str]:
    """Extract a unified diff fenced block (` ```diff ... ``` `) from frontier model output."""
    match = _DIFF_BLOCK_RE.search(frontier_output)
    if match:
        return match.group(1).strip() + "\n"
    if "--- a/" in frontier_output and "+++ b/" in frontier_output:
        lines = []
        capture = False
        for line in frontier_output.splitlines():
            if line.startswith(("diff --git ", "--- a/")):
                capture = True
            if capture:
                lines.append(line)
        if lines:
            return "\n".join(lines) + "\n"
    return None


def apply_and_verify_patch(diff_text: str, root: Path = PROJECT_ROOT) -> bool:
    """Validate unified diff via `git apply --check`, apply it, and re-run pytest."""
    git_bin = shutil.which("git")
    if not git_bin:
        return False
    check_proc = subprocess.run(
        [git_bin, "apply", "--check", "-"],
        input=diff_text,
        text=True,
        cwd=str(root),
        capture_output=True,
        check=False,
    )
    if check_proc.returncode != 0:
        return False

    apply_proc = subprocess.run(
        [git_bin, "apply", "-"],
        input=diff_text,
        text=True,
        cwd=str(root),
        capture_output=True,
        check=False,
    )
    if apply_proc.returncode != 0:
        return False

    verify_proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--tb=short", "-q"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    return verify_proc.returncode == 0


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
            "Identify the root cause of the failing deterministic/contract test(s) above and output "
            "a minimal unified diff inside a ```diff fenced code block that resolves the failure "
            "while preserving all Pydantic V2 contracts and security invariants.",
            "",
        ]
    )

    bundle_path.write_text("\n".join(sections), encoding="utf-8")
    return bundle_path


def invoke_frontier_cli(
    bundle_path: Path,
    cli_provider: str,
    auto_apply: bool = False,
) -> int:
    """Invoke `claude` or `openai` CLI piping the bundle over `stdin` to avoid OS `ARG_MAX` limits."""
    prompt_text = bundle_path.read_text(encoding="utf-8")
    if cli_provider == "claude":
        binary = shutil.which("claude")
        if not binary:
            print("Frontier CLI 'claude' not found in PATH; skipping remote call.", file=sys.stderr)
            return 1
        cmd = [binary, "-p", "Read the diagnostic bundle from stdin and output a unified diff fix."]
    elif cli_provider == "openai":
        binary = shutil.which("openai")
        if not binary:
            print("Frontier CLI 'openai' not found in PATH; skipping remote call.", file=sys.stderr)
            return 1
        cmd = [binary, "api", "chat.completions.create", "-m", "gpt-4o"]
    else:
        return 1

    proc = subprocess.run(
        cmd,
        input=prompt_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout)
        if auto_apply:
            diff_text = extract_unified_diff(proc.stdout)
            if diff_text and apply_and_verify_patch(diff_text):
                print("Applied frontier patch and verified all pytest checks now pass.")
                return 0
    return proc.returncode


def run_test_gate(
    pytest_args: Iterable[str] | None = None,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    frontier_cli: str | None = None,
    auto_apply: bool = False,
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
        return invoke_frontier_cli(written_path, frontier_cli, auto_apply=auto_apply)

    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pytest suite and only package failure tracebacks for a frontier model on failure."
    )
    parser.add_argument(
        "--frontier-cli",
        choices=["claude", "openai"],
        default=None,
        help="Optionally invoke a frontier CLI (via stdin) with .debug_bundle.md when pytest fails.",
    )
    parser.add_argument(
        "--apply-patch",
        action="store_true",
        help="Automatically validate and apply a unified diff returned by the frontier CLI.",
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
        auto_apply=args.apply_patch,
    )


if __name__ == "__main__":
    sys.exit(main())
