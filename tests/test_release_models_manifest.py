"""Tests for release models manifest and nested release layout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartrain.core.runtime.workspace_paths import WorkspaceLayout, deploy_workspace
from smartrain.services.models.release_model_rename_service import (
    apply_release_rename,
    build_rename_plan,
    discover_release_models,
)
from smartrain.services.models.release_models_manifest import (
    entry_key_for_pt,
    get_comment,
    get_comment_for_pt,
    load_manifest,
    release_comment_for_run_dir,
    sync_sidecar_comment,
    upsert_entry,
)
from smartrain.workflows.models import model_comment_cli as mcc
from smartrain.workflows.models.model_release_cli import _target_paths


def _write_nested_release_model(
    workspace: Path,
    *,
    dataset: str = "ds1",
    stem: str = "detect_yolov8n_20260115_120000",
    comment: str = "",
) -> Path:
    release_dir = workspace / "models" / dataset / stem
    release_dir.mkdir(parents=True, exist_ok=True)
    pt_path = release_dir / f"{stem}.pt"
    json_path = release_dir / f"{stem}.json"
    (release_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"model": "yolov8n", "dataset": {"name": dataset}}}),
        encoding="utf-8",
    )
    pt_path.write_bytes(b"pt-bytes")
    payload = {
        "comment": comment,
        "source": {
            "source_run": str(workspace / "runs" / "run1"),
            "source_run_relative": "runs/run1",
            "source_weights": "run1.pt",
            "source_sha256": "abc",
            "released_at": "2026-01-15T12:00:00+00:00",
        },
        "training": {"training_info": {"model": "yolov8n", "dataset": {"name": dataset}}},
        "artifacts": {
            "model_path": str(pt_path),
            "json_path": str(json_path),
            "release_dir": str(release_dir),
            "train_copy_dir": str(release_dir / "train"),
            "test_copy_dir": str(release_dir / "test"),
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return pt_path


def test_target_paths_use_nested_layout(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    md = {
        "training_info": {
            "dataset": {"name": "my_ds"},
            "model": "yolov8n",
            "task_type": "detect",
        },
        "timestamps": {"training": {"end": "2026-01-15T12:00:00+00:00"}},
    }
    release_dir, target_pt, target_json = _target_paths(layout, tmp_path / "runs" / "run1", md)
    assert release_dir.name == "detect_yolov8n_20260115_120000"
    assert target_pt.parent == release_dir
    assert target_json.parent == release_dir
    assert target_pt.name == f"{release_dir.name}.pt"


def test_manifest_upsert_and_get_comment(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    pt_path = _write_nested_release_model(tmp_path)

    upsert_entry(
        layout,
        entry_key=entry_key_for_pt(pt_path),
        model_path=pt_path,
        comment="Линия 3",
    )
    assert get_comment(layout, entry_key_for_pt(pt_path)) == "Линия 3"
    manifest = load_manifest(layout)
    assert (tmp_path / "models" / "releases_manifest.json").is_file()
    assert manifest["entries"][entry_key_for_pt(pt_path)]["comment"] == "Линия 3"


def test_release_comment_for_run_dir_nested(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    pt_path = _write_nested_release_model(tmp_path)
    upsert_entry(layout, entry_key=entry_key_for_pt(pt_path), model_path=pt_path, comment="Note")
    release_dir = pt_path.parent
    assert release_comment_for_run_dir(str(release_dir)) == "Note"


def test_discover_release_models_includes_comment(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    pt_path = _write_nested_release_model(tmp_path, comment="Sidecar note")
    upsert_entry(layout, entry_key=entry_key_for_pt(pt_path), model_path=pt_path, comment="Manifest note")

    entry = discover_release_models(layout)[0]
    assert entry.comment == "Manifest note"
    assert entry.release_dir == pt_path.parent


def test_model_comment_cli_updates_manifest_and_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    pt_path = _write_nested_release_model(tmp_path, comment="Old")
    upsert_entry(layout, entry_key=entry_key_for_pt(pt_path), model_path=pt_path, comment="Old")
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    mcc.main(["--workspace", str(tmp_path), "--release", str(pt_path), "--comment", "Новый комментарий"])

    assert get_comment(layout, entry_key_for_pt(pt_path)) == "Новый комментарий"
    payload = json.loads(pt_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["comment"] == "Новый комментарий"


def test_rename_nested_release_updates_manifest(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    old_stem = "detect_yolov8n_20260115_120000"
    new_stem = "my_detector_v2"
    pt_path = _write_nested_release_model(tmp_path, stem=old_stem, comment="Keep me")
    upsert_entry(layout, entry_key=entry_key_for_pt(pt_path), model_path=pt_path, comment="Keep me")

    entry = discover_release_models(layout)[0]
    plan = build_rename_plan(entry, new_stem)
    apply_release_rename(plan, layout=layout)

    new_pt = tmp_path / "models" / "ds1" / new_stem / f"{new_stem}.pt"
    assert new_pt.is_file()
    assert get_comment(layout, f"ds1/{new_stem}") == "Keep me"
    assert get_comment_for_pt(layout, new_pt) == "Keep me"


def test_sync_sidecar_comment(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    pt_path = _write_nested_release_model(tmp_path)
    json_path = pt_path.with_suffix(".json")
    sync_sidecar_comment(json_path, "Updated")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["comment"] == "Updated"
