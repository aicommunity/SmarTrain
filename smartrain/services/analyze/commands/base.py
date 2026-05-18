from __future__ import annotations

import argparse
from abc import ABC, abstractmethod


class AnalyzeCommandHandler(ABC):
    """Analyze CLI subcommand handler."""

    name: str

    @abstractmethod
    def register(self, subparsers: argparse._SubParsersAction, *, common: argparse.ArgumentParser) -> argparse.ArgumentParser:
        ...

    @abstractmethod
    def run(self, args: argparse.Namespace) -> None:
        ...
