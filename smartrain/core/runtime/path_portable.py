"""
Portable path strings relative to a workspace root.
"""
from __future__ import annotations

import os
from typing import Any


def is_abs_like(value: str) -> bool:
    """True for OS-absolute, POSIX-root (``/…``), or drive-letter paths."""
    v = (value or "").strip()
    if not v:
        return False
    if os.path.isabs(v):
        return True
    if v.startswith("/"):
        return True
    if len(v) >= 3 and v[1] == ":" and v[2] in "/\\":
        return True
    return False


def to_posix(value: str) -> str:
    return (value or "").strip().replace("\\", "/")


def posix_relpath(path: str, start: str) -> str:
    """Like ``os.path.relpath`` but always returns forward-slash separators."""
    ap = os.path.abspath(os.path.expanduser(path))
    sp = os.path.abspath(os.path.expanduser(start))
    return os.path.relpath(ap, sp).replace("\\", "/")


def relativize_if_under(workspace_root: str, path: str | None) -> str | None:
    """
    If ``path`` resolves to a location under ``workspace_root``, return a POSIX-style
    path relative to the workspace root. Otherwise return ``path`` unchanged.
    Workspace root itself becomes ".".
    """
    if path is None or not isinstance(path, str):
        return path
    p = path.strip()
    if not p:
        return path
    wr = os.path.abspath(os.path.expanduser(workspace_root))
    try:
        # Drive-letter / OS abs: resolve normally.
        if os.path.isabs(p) or (len(p) >= 3 and p[1] == ":" and p[2] in "/\\"):
            ap = os.path.abspath(os.path.expanduser(p))
        elif p.startswith("/"):
            # POSIX abs on Windows: abspath would map to drive root incorrectly for
            # under-workspace checks; only relativize when joined form is under WS
            # via suffix is out of scope here — leave unchanged unless native under WS.
            ap = os.path.abspath(os.path.expanduser(p))
        else:
            ap = os.path.abspath(os.path.expanduser(p))
    except (OSError, ValueError):
        return path
    if ap == wr:
        return "."
    if ap.startswith(wr + os.sep):
        return posix_relpath(ap, wr)
    return path


def store_path_under_workspace(workspace_root: str, path: str) -> str:
    """
    Prefer a workspace-relative POSIX path when ``path`` is under the workspace;
    otherwise return the original string (absolute allowed outside the workspace).
    """
    p = (path or "").strip()
    if not p:
        return p
    rel = relativize_if_under(workspace_root, p)
    if rel is None:
        return p
    if rel != p:
        return rel
    # Already relative POSIX under workspace?
    if not is_abs_like(p):
        return to_posix(p)
    return p


def relativize_abs_paths_in_obj(obj: Any, workspace_root: str) -> Any:
    """Recursively replace absolute path strings under workspace_root with relative POSIX paths."""
    if isinstance(obj, dict):
        return {k: relativize_abs_paths_in_obj(v, workspace_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relativize_abs_paths_in_obj(x, workspace_root) for x in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if not s or not is_abs_like(s):
            return obj
        rel = relativize_if_under(workspace_root, s)
        if rel is None or rel == s:
            return obj
        return rel
    return obj


def resolve_stored_path_under_workspace(workspace_root: str, stored: str) -> str:
    """Resolve a path saved by ``relativize_if_under`` (or legacy absolute) against workspace."""
    s = (stored or "").strip()
    if not s:
        raise ValueError("Empty path.")
    wr = os.path.abspath(os.path.expanduser(workspace_root))
    if s == ".":
        return wr
    # Native OS absolute (including drive-letter).
    if os.path.isabs(s) or (len(s) >= 3 and s[1] == ":" and s[2] in "/\\"):
        return os.path.abspath(os.path.expanduser(s))
    # POSIX absolute on Windows: do not treat as workspace-relative join.
    if s.startswith("/"):
        return os.path.abspath(s)
    # Legacy Windows-style relative paths (``datasets\a``) → POSIX then OS join.
    return os.path.abspath(os.path.join(wr, to_posix(s).replace("/", os.sep)))


def resolve_workspace_or_abs_path(workspace_root: str | None, stored: str) -> str:
    """
    Dual-read helper: absolute paths resolve as-is; relative POSIX paths join under workspace.
    When ``workspace_root`` is missing, relative paths resolve against the process cwd.
    """
    s = (stored or "").strip()
    if not s:
        return s
    if workspace_root and not is_abs_like(s):
        return resolve_stored_path_under_workspace(workspace_root, s)
    if is_abs_like(s):
        return os.path.abspath(os.path.expanduser(s))
    return os.path.abspath(s.replace("/", os.sep))
