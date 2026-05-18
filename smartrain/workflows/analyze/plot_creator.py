#!/usr/bin/env python3
"""
Outdated script. Use smartrain analyze to compare runs and graphs.

Example:
  smartrain analyze compare --baseline /path/to/run1 --others /path/to/run2 \\
      -o delta.csv --out-png curves.png
"""
from __future__ import annotations

import sys

from smartrain.workflows.analyze.analyze_entry import main as analyze_main


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    analyze_main(argv if argv else ["--help"])


if __name__ == "__main__":
    main()
