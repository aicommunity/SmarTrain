"""Shared logging helpers for smartrain modules."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: int = logging.INFO, *, use_rich: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler: logging.Handler
    if use_rich:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=False, show_path=False, markup=False)
            handler.setFormatter(logging.Formatter("%(message)s"))
        except Exception:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
        root.setLevel(level)
    else:
        logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    configure_logging()
    if name:
        return logging.getLogger(name)
    return logging.getLogger("smartrain")


# Alias used by audit remediation docs
setup_logging = configure_logging
