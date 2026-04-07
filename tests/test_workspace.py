"""deploy_workspace и WorkspaceLayout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartrain.workspace_paths import (
    WORKSPACE_ENV_VAR,
    WorkspaceLayout,
    deploy_workspace,
    resolve_workspace_root,
)


def test_deploy_workspace_creates_structure(tmp_path: Path) -> None:
    info = deploy_workspace(str(tmp_path))
    assert info["root"] == str(tmp_path.resolve())
    assert "raw_data" in info["created_dirs"]
    assert "datasets" in info["created_dirs"]
    assert "runs" in info["created_dirs"]
    assert "tmp" in info["created_dirs"]
    layout = WorkspaceLayout(str(tmp_path))
    assert Path(layout.source_datasets_info_path()).is_file()
    assert Path(layout.work_datasets_info_path()).is_file()
    assert (Path(layout.raw_data) / "datasets_list.txt").is_file()
    assert (Path(layout.raw_data) / "datasets_list.txt").read_text(encoding="utf-8") == ""
    with open(layout.source_datasets_info_path(), encoding="utf-8") as f:
        assert json.load(f) == {}


def test_deploy_workspace_idempotent(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    b = deploy_workspace(str(tmp_path))
    assert all(s.startswith(("dir:", "file:")) for s in b["skipped"])
    assert b["created_dirs"] == []
    assert b["created_files"] == []


def test_resolve_workspace_root_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    assert resolve_workspace_root(None) == str(tmp_path.resolve())


def test_resolve_workspace_root_cli_over_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    assert resolve_workspace_root(str(other)) == str(other.resolve())
