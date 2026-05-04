from __future__ import annotations

from smartrain import results_analyzer


def build_arg_parser():
    return results_analyzer.build_analyze_arg_parser()


def main(argv: list[str] | None = None):
    return results_analyzer.main(argv)

