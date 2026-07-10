from __future__ import annotations

import re
import subprocess
import sys


def _extract_count(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1))


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def main() -> int:
    failed = False

    ruff_limit = 511
    ruff_code, ruff_out = _run([sys.executable, "-m", "ruff", "check", "smartrain", "tests"])
    ruff_count = _extract_count(ruff_out, r"Found (\d+) errors\.")
    if ruff_count is None:
        print("[ERROR] Unable to parse Ruff error count.")
        print(ruff_out)
        return 1
    print(f"[INFO] Ruff errors: {ruff_count} (baseline <= {ruff_limit})")
    if ruff_count > ruff_limit:
        failed = True
        print("[ERROR] Ruff errors exceeded baseline.")

    mypy_limit = 306
    mypy_code, mypy_out = _run([sys.executable, "-m", "mypy", "smartrain"])
    mypy_count = _extract_count(mypy_out, r"Found (\d+) errors in \d+ files")
    if mypy_count is None:
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
