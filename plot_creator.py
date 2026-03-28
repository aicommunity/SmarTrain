#!/usr/bin/env python3
"""
Устаревший скрипт. Используйте results_analyzer для сравнения прогонов и графиков.

Пример:
  python3 results_analyzer.py compare --baseline /path/to/run1 --others /path/to/run2 \\
      -o delta.csv --out-png curves.png
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Обёртка: перенаправление на results_analyzer (старый plot_creator устарел)."
    )
    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Аргументы передаются в results_analyzer.py",
    )
    args = parser.parse_args()
    script = Path(__file__).resolve().parent / "results_analyzer.py"
    cmd = [sys.executable, str(script)] + (args.remainder or ["--help"])
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
