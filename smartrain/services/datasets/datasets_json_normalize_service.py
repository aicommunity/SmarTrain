from __future__ import annotations

import os
from typing import Optional


def _normalize_path_for_data_path(path: str, workspace_root: Optional[str]) -> str:
    """Returns the path for data_path: relative to workspace, if possible."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    if workspace_root:
        try:
            rel = os.path.relpath(abs_path, workspace_root)
            if not rel.startswith(".."):
                return rel
        except Exception:
            pass
    return abs_path

