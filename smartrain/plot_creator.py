#!/usr/bin/env python3
"""
Устаревший скрипт. Используйте smartrain analyze для сравнения прогонов и графиков.

Пример:
  smartrain analyze compare --baseline /path/to/run1 --others /path/to/run2 \\
      -o delta.csv --out-png curves.png
"""
from __future__ import annotations

import sys

from smartrain.results_analyzer import main as analyze_main


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    analyze_main(argv if argv else ["--help"])


if __name__ == "__main__":
    main()
