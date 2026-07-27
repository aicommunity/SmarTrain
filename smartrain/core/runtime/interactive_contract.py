from __future__ import annotations

import os
import sys
from typing import TextIO

INTERACTIVE_ALLOWED_ENV = "SMART_TRAIN_INTERACTIVE_ALLOWED"


def is_interactive_allowed(argv: list[str] | tuple[str, ...] | None = None) -> bool:
    """
    Unified contract:
    - if env flag is set by CLI router, trust it;
    - otherwise (direct module run/tests), allow interactive only when argv is empty.
    """
    env = os.environ.get(INTERACTIVE_ALLOWED_ENV)
    if env is not None:
        return str(env).strip() in {"1", "true", "yes", "on"}
    return not bool(argv)


def _windows_stdin_is_console() -> bool:
    """True when STD_INPUT_HANDLE is a real console (GetConsoleMode succeeds)."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        if not handle or handle == wintypes.HANDLE(-1).value:
            return False
        mode = wintypes.DWORD()
        return bool(kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
    except Exception:
        return False


def stdin_is_tty() -> bool:
    """Reliable interactive-stdin check (Windows: distrust false-positive isatty)."""
    try:
        if not sys.stdin.isatty():
            return False
    except Exception:
        return False
    if sys.platform == "win32":
        return _windows_stdin_is_console()
    return True


class _NonTtyStdinProxy:
    """Wrap stdin so ``isatty()`` is False while delegating all other I/O."""

    def __init__(self, inner: TextIO) -> None:
        self._inner = inner

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def ensure_stdin_isatty_truthful() -> None:
    """
    On Windows, ``sys.stdin.isatty()`` can return True when no console is attached
    (redirected / piped / Cursor-hosted shells). Force a truthful False so all
    ``sys.stdin.isatty()`` call sites agree with GetConsoleMode.
    """
    if sys.platform != "win32":
        return
    try:
        if not sys.stdin.isatty():
            return
    except Exception:
        return
    if _windows_stdin_is_console():
        return
    if isinstance(sys.stdin, _NonTtyStdinProxy):
        return
    sys.stdin = _NonTtyStdinProxy(sys.stdin)  # type: ignore[assignment]
