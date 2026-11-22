from __future__ import annotations

import re
import subprocess
import sys


def _extract_count(text: str, *, tool: str) -> int | None:
    """Parse ruff/mypy summary lines, including singular and zero-error forms."""
    if tool == "ruff":
        match = re.search(r"Found (\d+) errors?\.", text)
        if match:
            return int(match.group(1))
        if re.search(r"All checks passed!", text):
            return 0
        return None

    if tool == "mypy":
        match = re.search(r"Found (\d+) errors? in \d+ files?", text)
        if match:
            return int(match.group(1))
        if re.search(r"Success:\s+no issues found", text, re.IGNORECASE):
            return 0
        return None

    raise ValueError(f"unknown tool: {tool}")


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def main() -> int:
    failed = False

    # Hard gate: undefined names and syntax errors are always bugs (they broke
    # augment/balance at runtime after the module-split wave); never baseline.
    hard_code, hard_out = _run([sys.executable, "-m", "ruff", "check", "smartrain", "tests", "--select", "F821,E9"])
    if hard_code != 0:
        failed = True
        print("[ERROR] Undefined names / syntax errors found (F821/E9 must be zero):")
        print(hard_out)

    ruff_limit = 283
    ruff_code, ruff_out = _run([sys.executable, "-m", "ruff", "check", "smartrain", "tests"])
    ruff_count = _extract_count(ruff_out, tool="ruff")
    if ruff_count is None:
        # Successful empty output is still zero (some ruff versions are quiet).
        if ruff_code == 0 and not ruff_out.strip():
            ruff_count = 0
        else:
            print("[ERROR] Unable to parse Ruff error count.")
            print(ruff_out)
            return 1
    print(f"[INFO] Ruff errors: {ruff_count} (baseline <= {ruff_limit})")
    if ruff_count > ruff_limit:
        failed = True
        print("[ERROR] Ruff errors exceeded baseline.")

    mypy_limit = 373
    mypy_code, mypy_out = _run([sys.executable, "-m", "mypy", "smartrain"])
    mypy_count = _extract_count(mypy_out, tool="mypy")
    if mypy_count is None:
        if mypy_code == 0 and not mypy_out.strip():
            mypy_count = 0
        else:
            print("[ERROR] Unable to parse Mypy error count.")
            print(mypy_out)
            return 1
    print(f"[INFO] Mypy errors: {mypy_count} (baseline <= {mypy_limit})")
    if mypy_count > mypy_limit:
        failed = True
        print("[ERROR] Mypy errors exceeded baseline.")

    if failed:
        return 1

    print("[OK] Lint baseline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
