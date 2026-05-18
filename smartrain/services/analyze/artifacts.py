from __future__ import annotations

import os
from datetime import datetime

from smartrain.core.runtime.workspace_paths import WorkspaceLayout, resolve_workspace_root


def session_name(raw: str | None) -> str:
    value = (raw or "").strip()
    if value:
        return value
    return f"analyze_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def session_root(workspace_cli: str | None, analytics_session: str | None) -> str:
    try:
        ws = resolve_workspace_root(workspace_cli)
        base = WorkspaceLayout(ws).analytics
    except ValueError:
        base = os.path.join(os.getcwd(), "analytics")
    name = session_name(analytics_session)
    root = os.path.join(base, "analyze-reports", name)
    os.makedirs(root, exist_ok=True)
    return root


def session_artifacts_dir(workspace_cli: str | None, analytics_session: str | None, category: str) -> str:
    root = session_root(workspace_cli, analytics_session)
    out = os.path.join(root, "artifacts", category)
    os.makedirs(out, exist_ok=True)
    return out


def default_relative_output(
    workspace_cli: str | None,
    analytics_session: str | None,
    category: str,
    file_name: str,
    raw: str | None,
) -> str:
    requested = (raw or "").strip()
    if os.path.isabs(requested):
        return os.path.abspath(requested)
    if requested:
        return os.path.join(
            session_artifacts_dir(workspace_cli, analytics_session, category),
            os.path.basename(requested),
        )
    return os.path.join(session_artifacts_dir(workspace_cli, analytics_session, category), file_name)
