from __future__ import annotations

import os
from typing import Optional

from smartrain.core.runtime.path_portable import store_path_under_workspace, to_posix


def _normalize_path_for_data_path(path: str, workspace_root: Optional[str]) -> str:
    """Returns the path for data_path: relative POSIX under workspace when possible."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    if workspace_root:
        stored = store_path_under_workspace(workspace_root, abs_path)
        if stored != abs_path:
            return stored
        # Outside workspace: keep absolute with forward slashes for portability of the string form.
        return to_posix(abs_path) if os.name != "nt" else abs_path
    return abs_path
