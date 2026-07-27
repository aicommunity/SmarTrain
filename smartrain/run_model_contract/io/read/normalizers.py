from __future__ import annotations

import os
from pathlib import Path

from smartrain.core.runtime.path_portable import is_abs_like, to_posix


def normalize_path(path: str, *, anchor: str | Path | None = None) -> str:
    """
    Resolve a stored path to an absolute filesystem path.

    Relative paths are joined with ``anchor`` when provided (run/workspace root).
    Legacy absolute paths are resolved as-is.
    """
    raw = to_posix(str(path or "").strip())
    if not raw:
        raise ValueError("Empty path.")
    # Drive-letter / OS-native absolute
    if os.path.isabs(raw) or (len(raw) >= 3 and raw[1] == ":" and raw[2] in "/\\"):
        return str(Path(raw).expanduser().resolve())
    # POSIX absolute that exists on this machine
    if raw.startswith("/") and Path(raw).exists():
        return str(Path(raw).resolve())
    if anchor is not None and not is_abs_like(raw):
        joined = Path(anchor).expanduser().resolve().joinpath(*raw.split("/"))
        return str(joined.resolve())
    if anchor is not None and raw.startswith("/"):
        # Foreign POSIX abs: try as relative suffix under anchor is out of scope;
        # fall through to Path resolve (cwd-relative on Windows).
        pass
    return str(Path(raw).expanduser().resolve())


def normalize_task(task: str | None) -> str:
    value = str(task or "").strip().lower()
    if value in {"detect", "detection", ""}:
        return "detection"
    if value in {"classify", "classification"}:
        return "classification"
    if value in {"segment", "segmentation"}:
        return "segmentation"
    return "detection"


def normalize_backend(backend: str | None) -> str:
    value = str(backend or "").strip().lower()
    if value:
        return value
    return "ultralytics"
