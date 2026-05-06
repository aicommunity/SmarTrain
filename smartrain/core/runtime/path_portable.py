"""
Portable path strings relative to a workspace root.
"""
from __future__ import annotations

import os
from typing import Any


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
        ap = os.path.abspath(os.path.expanduser(p))
    except (OSError, ValueError):
        return path
    if ap == wr:
        return "."
    if ap.startswith(wr + os.sep):
        rel = os.path.relpath(ap, wr)
        return rel.replace("\\", "/")
    return path


def relativize_abs_paths_in_obj(obj: Any, workspace_root: str) -> Any:
    """Recursively replace absolute path strings under workspace_root with relative POSIX paths."""
    if isinstance(obj, dict):
        return {k: relativize_abs_paths_in_obj(v, workspace_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relativize_abs_paths_in_obj(x, workspace_root) for x in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if not s or not os.path.isabs(s):
            return obj
        return relativize_if_under(workspace_root, s) or obj
    return obj


def resolve_stored_path_under_workspace(workspace_root: str, stored: str) -> str:
    """Resolve a path saved by ``relativize_if_under`` (or legacy absolute) against workspace."""
    s = (stored or "").strip()
    if not s:
        raise ValueError("Empty path.")
    wr = os.path.abspath(os.path.expanduser(workspace_root))
    if os.path.isabs(s):
        return os.path.abspath(s)
    if s == ".":
        return wr
    return os.path.abspath(os.path.join(wr, s.replace("/", os.sep)))
