from __future__ import annotations

import argparse
from typing import Callable

from smartrain.services.analyze.commands.base import AnalyzeCommandHandler


class _CallableHandler(AnalyzeCommandHandler):
    def __init__(
        self,
        name: str,
        *,
        register_parser: Callable[[argparse._SubParsersAction, argparse.ArgumentParser], argparse.ArgumentParser],
        run: Callable[[argparse.Namespace], None],
    ) -> None:
        self.name = name
        self._register_parser = register_parser
        self._run = run

    def register(self, subparsers: argparse._SubParsersAction, *, common: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return self._register_parser(subparsers, common)

    def run(self, args: argparse.Namespace) -> None:
        self._run(args)


class AnalyzeCommandRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, AnalyzeCommandHandler] = {}

    def register(self, handler: AnalyzeCommandHandler) -> None:
        self._handlers[handler.name] = handler

    def register_callable(
        self,
        name: str,
        *,
        register_parser: Callable[[argparse._SubParsersAction, argparse.ArgumentParser], argparse.ArgumentParser],
        run: Callable[[argparse.Namespace], None],
    ) -> None:
        self.register(_CallableHandler(name, register_parser=register_parser, run=run))

    def add_subparsers(self, sub: argparse._SubParsersAction, *, common: argparse.ArgumentParser) -> None:
        for handler in self._handlers.values():
            handler.register(sub, common=common)

    def dispatch(self, args: argparse.Namespace) -> int:
        cmd = str(getattr(args, "cmd", "") or "")
        handler = self._handlers.get(cmd)
        if handler is None:
            raise SystemExit(f"Unknown analyze command: {cmd!r}")
        handler.run(args)
        return 0
