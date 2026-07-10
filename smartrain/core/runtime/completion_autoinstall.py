"""Best-effort automatic shell completion setup."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


_STATE_ENV = "SMARTRAIN_AUTO_COMPLETION_RUNNING"
_DISABLE_ENV = "SMARTRAIN_DISABLE_AUTO_COMPLETION"


def _state_file() -> Path:
    root = (
        os.environ.get("XDG_STATE_HOME")
        or os.path.join(os.path.expanduser("~"), ".local", "state")
    )
    return Path(root) / "smartrain" / "completion_autoinstall.json"


def _should_skip(argv: list[str]) -> bool:
    if os.environ.get(_DISABLE_ENV, "").strip() in {"1", "true", "yes"}:
        return True
    if os.environ.get(_STATE_ENV, "").strip() == "1":
        return True
    if os.environ.get("_SMARTRAIN_COMPLETE", "").strip():
        return True
    if os.environ.get("CI", "").strip():
        return True
    if "--install-completion" in argv or "--show-completion" in argv:
        return True
    return False


def _already_attempted() -> bool:
    path = _state_file()
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return bool(payload.get("attempted"))
    except Exception:
        return True


def _mark_attempted(success: bool, error: str = "") -> None:
    path = _state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"attempted": True, "success": bool(success), "error": str(error)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_completion_auto_setup(argv: list[str]) -> None:
    """Try one-time completion installation; never raise to caller."""
    if _should_skip(argv):
        return
    if _already_attempted():
        return
    env = os.environ.copy()
    env[_STATE_ENV] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "smartrain", "--install-completion"],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
        _mark_attempted(success=(proc.returncode == 0), error=(proc.stderr or proc.stdout or "").strip()[:400])
    except Exception as exc:
        _mark_attempted(success=False, error=str(exc))
