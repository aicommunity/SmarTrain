from __future__ import annotations

from smartrain.services.analyze_artifacts import (
    default_relative_output,
    session_artifacts_dir,
    session_name,
    session_root,
)


def resolve_session_name(raw: str | None) -> str:
    return session_name(raw)


def resolve_session_root(workspace_cli: str | None, analytics_session: str | None) -> str:
    return session_root(workspace_cli, analytics_session)


def resolve_session_artifacts_dir(workspace_cli: str | None, analytics_session: str | None, category: str) -> str:
    return session_artifacts_dir(workspace_cli, analytics_session, category)


def resolve_default_relative_output(
    workspace_cli: str | None,
    analytics_session: str | None,
    category: str,
    file_name: str,
    raw: str | None,
) -> str:
    return default_relative_output(workspace_cli, analytics_session, category, file_name, raw)

