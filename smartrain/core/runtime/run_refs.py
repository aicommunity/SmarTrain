"""Resolve run directory references (numeric index or filesystem path)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from smartrain.core.runtime.path_portable import is_abs_like, to_posix
from smartrain.core.runtime.run_discovery import find_run_directories


def resolve_run_ref(
    runs_dir: str | Path,
    ref: str,
    *,
    workspace_root: str | Path | None = None,
    exit_on_error: bool = True,
) -> str:
    """Resolve a run reference to an absolute directory path.

    ``ref`` may be a 1-based index into ``find_run_directories(runs_dir)``
    or a filesystem path. Relative paths resolve under ``workspace_root`` when
    provided (workspace-relative POSIX), otherwise under the process CWD.
    """
    s = str(ref).strip()
    runs_root = str(runs_dir)
    if s.isdigit():
        runs = find_run_directories(runs_root)
        i = int(s)
        if i < 1 or i > len(runs):
            msg = f"There is no run with number {i} (in the list {len(runs)})."
            if exit_on_error:
                print(f"[ERROR] {msg}", file=sys.stderr)
                raise SystemExit(1)
            raise IndexError(msg)
        return runs[i - 1]
    if is_abs_like(s):
        return str(Path(s).expanduser().resolve())
    if workspace_root is not None:
        return str((Path(workspace_root) / to_posix(s)).resolve())
    return os.path.abspath(os.path.expanduser(s))


def resolve_run_ref_path(
    runs_dir: str | Path,
    ref: str,
    *,
    workspace_root: str | Path | None = None,
    exit_on_error: bool = True,
) -> Path:
    return Path(
        resolve_run_ref(
            runs_dir,
            ref,
            workspace_root=workspace_root,
            exit_on_error=exit_on_error,
        )
    )
