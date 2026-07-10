from __future__ import annotations

import argparse

from smartrain.services.visualization.cli_commands import run_vis_cli


def build_vis_arg_parser() -> argparse.ArgumentParser:
    from smartrain.services.visualization.cli_commands import build_vis_arg_parser as _build

    return _build()


def main(argv: list[str] | None = None) -> int:
    return run_vis_cli(argv)

