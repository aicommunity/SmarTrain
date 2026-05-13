from __future__ import annotations

from smartrain.workflows.analyze import analyze_entry


def build_arg_parser():
    return analyze_entry.build_analyze_arg_parser()


def main(argv: list[str] | None = None):
    return analyze_entry.main(argv)

