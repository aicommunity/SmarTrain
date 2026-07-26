"""Tests for model release (move) and unrelease workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartrain.core.runtime.workspace_paths import WorkspaceLayout, deploy_workspace
from smartrain.services.models.release_model_rename_service import discover_release_models
from smartrain.services.models.release_models_manifest import (
    entry_key_for_pt,
    find_release_pt_in_dir,
    is_unified_release_bundle,
    load_manifest,
    release_dir_for_pt,
    remove_entry,
    upsert_entry,
)
from smartrain.workflows.models import model_release_cli as mrc
from smartrain.workflows.models import model_unrelease_cli as muc


def _training_metadata(*, dataset: str = "my_ds", model: str = "yolo11s") -> dict:
    return {
        "training_info": {
            "dataset": {"name": dataset},
            "model": model,
            "task_type": "detect",
            "hyperparameters": {
                "epochs": 400,
                "batch_size": 16,
                "image_size": 640,
            },
        },
        "timestamps": {"training": {"start": "2026-07-06T03:02:58+00:00"}},
    }


def _write_run(
    workspace: Path,
    *,
    dataset: str,
    run_id: str,
    stem: str,
    pt_bytes: bytes = b"canonical-pt",
) -> Path:
    run_dir = workspace / "runs" / dataset / run_id
    models = run_dir / "models"
    models.mkdir(parents=True)
    (models / f"{stem}.pt").write_bytes(pt_bytes)
    (run_dir / "training_metadata.json").write_text(
        json.dumps(_training_metadata(dataset=dataset), ensure_ascii=False),
        encoding="utf-8",
    )
    tests = run_dir / "tests"
    tests.mkdir()
    (tests / "test_metrics.csv").write_text("metric\n0.9\n", encoding="utf-8")
    return run_dir


def _write_legacy_root_release(
    workspace: Path,
    *,
    dataset: str,
    run_id: str,
    stem: str,
) -> Path:
    release_dir = workspace / "models" / dataset / run_id
    release_dir.mkdir(parents=True)
    pt_path = release_dir / f"{stem}.pt"
    json_path = release_dir / f"{stem}.json"
    pt_path.write_bytes(b"legacy-pt")
    (release_dir / f"{stem}.onnx").write_bytes(b"onnx")
    payload = {
        "comment": "",
        "source": {
            "source_run": str(workspace / "runs" / dataset / run_id),
            "source_run_relative": f"runs/{dataset}/{run_id}",
            "source_weights": f"{stem}.pt",
            "source_sha256": "abc",
            "released_at": "2026-01-15T12:00:00+00:00",
        },
        "training": {"training_info": {"model": "yolo11s", "dataset": {"name": dataset}}},
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


def test_release_dir_for_unified_layout(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    stem = "detect_yolo11s_20260706_030258_640px_400epochs_b16"
    run_id = "run-abc"
    release_dir = tmp_path / "models" / "my_ds" / run_id
    models = release_dir / "models"
    models.mkdir(parents=True)
    pt_path = models / f"{stem}.pt"
    json_path = models / f"{stem}.json"
    pt_path.write_bytes(b"pt")
    payload = {
        "comment": "",
        "source": {
            "source_run": str(tmp_path / "runs" / "my_ds" / run_id),
            "source_run_relative": f"runs/my_ds/{run_id}",
            "source_weights": f"{stem}.pt",
            "source_sha256": "abc",
            "released_at": "2026-01-15T12:00:00+00:00",
        },
        "training": {},
        "artifacts": {
            "model_path": str(pt_path),
            "json_path": str(json_path),
            "release_dir": str(release_dir),
            "train_copy_dir": str(release_dir / "train"),
            "test_copy_dir": str(release_dir / "test"),
        },
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    assert release_dir_for_pt(pt_path) == release_dir.resolve()
    assert find_release_pt_in_dir(release_dir) == pt_path.resolve()
    assert is_unified_release_bundle(release_dir)


def test_model_release_moves_run_with_full_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    stem = "detect_yolo11s_20260706_030258_640px_400epochs_b16"
    run_id = "2026-07-14_20-42_ultralytics_yolo11s_640px_400epochs_b16-b1ef93cc"
    run_dir = _write_run(tmp_path, dataset="my_ds", run_id=run_id, stem=stem)
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    mrc.main(["--workspace", str(tmp_path), "--run", str(run_dir), "--comment", "Prod"])

    release_dir = tmp_path / "models" / "my_ds" / run_id
    assert release_dir.is_dir()
    assert not run_dir.exists()
    assert (release_dir / "models" / f"{stem}.pt").read_bytes() == b"canonical-pt"
    assert (release_dir / "models" / f"{stem}.json").is_file()
    assert (release_dir / "tests" / "test_metrics.csv").is_file()

    manifest = load_manifest(layout)
    key = entry_key_for_pt(release_dir / "models" / f"{stem}.pt")
    assert manifest["entries"][key]["comment"] == "Prod"

    entry = discover_release_models(layout)[0]
    assert entry.release_dir == release_dir.resolve()
    assert entry.pt_path == (release_dir / "models" / f"{stem}.pt").resolve()


def test_model_release_idempotent_removes_duplicate_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deploy_workspace(str(tmp_path))
    stem = "detect_yolo11s_20260706_030258_640px_400epochs_b16"
    run_id = "run-dup"
    run_dir = _write_run(tmp_path, dataset="my_ds", run_id=run_id, stem=stem)
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    mrc.main(["--workspace", str(tmp_path), "--run", str(run_dir)])
    duplicate_run = _write_run(tmp_path, dataset="my_ds", run_id=run_id, stem=stem)
    assert duplicate_run.is_dir()

    with pytest.raises(SystemExit) as exc:
        mrc.main(["--workspace", str(tmp_path), "--run", str(duplicate_run)])
    assert exc.value.code == 0
    assert not duplicate_run.exists()


def test_model_unrelease_moves_back_to_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    stem = "detect_yolo11s_20260706_030258_640px_400epochs_b16"
    run_id = "run-unrel"
    run_dir = _write_run(tmp_path, dataset="my_ds", run_id=run_id, stem=stem)
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    mrc.main(["--workspace", str(tmp_path), "--run", str(run_dir)])
    release_dir = tmp_path / "models" / "my_ds" / run_id
    pt_path = release_dir / "models" / f"{stem}.pt"
    key = entry_key_for_pt(pt_path)

    muc.main(["--workspace", str(tmp_path), "--release", str(pt_path), "--yes"])

    runs_dir = tmp_path / "runs" / "my_ds" / run_id
    assert runs_dir.is_dir()
    assert not release_dir.exists()
    assert (runs_dir / "models" / f"{stem}.pt").is_file()
    assert not (runs_dir / "models" / f"{stem}.json").exists()
    assert key not in load_manifest(layout).get("entries", {})


def test_model_unrelease_legacy_root_moves_convert_into_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    stem = "detect_yolo11s_20260706_030258_640px_400epochs_b16"
    run_id = "legacy-run"
    pt_path = _write_legacy_root_release(tmp_path, dataset="my_ds", run_id=run_id, stem=stem)
    upsert_entry(layout, entry_key=entry_key_for_pt(pt_path), model_path=pt_path, comment="")
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    muc.main(["--workspace", str(tmp_path), "--release", str(pt_path), "--yes"])

    runs_dir = tmp_path / "runs" / "my_ds" / run_id
    assert runs_dir.is_dir()
    assert (runs_dir / "models" / f"{stem}.pt").is_file()
    assert (runs_dir / "models" / f"{stem}.onnx").read_bytes() == b"onnx"
    assert discover_release_models(layout) == []


def test_remove_entry(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    key = "ds1/detect_test"
    upsert_entry(
        layout,
        entry_key=key,
        model_path=tmp_path / "models" / "ds1" / "run" / "models" / "detect_test.pt",
        comment="x",
    )
    assert remove_entry(layout, key) is True
    assert remove_entry(layout, key) is False
    assert key not in load_manifest(layout).get("entries", {})


def test_model_unrelease_fails_when_runs_destination_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deploy_workspace(str(tmp_path))
    stem = "detect_yolo11s_20260706_030258_640px_400epochs_b16"
    run_id = "conflict-run"
    run_dir = _write_run(tmp_path, dataset="my_ds", run_id=run_id, stem=stem)
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    mrc.main(["--workspace", str(tmp_path), "--run", str(run_dir)])
    _write_run(tmp_path, dataset="my_ds", run_id=run_id, stem=stem)
    pt_path = tmp_path / "models" / "my_ds" / run_id / "models" / f"{stem}.pt"

    with pytest.raises(SystemExit) as exc:
        muc.main(["--workspace", str(tmp_path), "--release", str(pt_path), "--yes"])
    assert exc.value.code == 1
