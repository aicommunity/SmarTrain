"""Shared logging helpers for smartrain modules."""

from __future__ import annotations

import logging

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    configure_logging()
    if name:
        return logging.getLogger(name)
    return logging.getLogger("smartrain")
