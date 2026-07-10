"""Tests for model rename (release catalog)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartrain.core.runtime.workspace_paths import WorkspaceLayout, deploy_workspace
from smartrain.services.models.release_model_naming import is_release_metadata
from smartrain.services.models.release_model_rename_service import (
    ReleaseRenameError,
    apply_release_rename,
    build_rename_plan,
    discover_release_models,
)
from smartrain.workflows.models import model_rename_cli as mrc


def _write_release_model(
    workspace: Path,
    *,
    dataset: str = "ds1",
    stem: str = "detect_yolov8n_20260115_120000",
    model_name: str = "yolov8n",
    task_type: str = "detect",
) -> Path:
    models_dir = workspace / "models" / dataset
    models_dir.mkdir(parents=True, exist_ok=True)
    pt_path = models_dir / f"{stem}.pt"
    json_path = models_dir / f"{stem}.json"
    release_dir = models_dir / stem
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {
                    "model": model_name,
                    "task_type": task_type,
                    "dataset": {"name": dataset},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pt_path.write_bytes(b"pt-bytes")
    payload = {
        "source": {
            "source_run": str(workspace / "runs" / "run1"),
            "source_run_relative": "runs/run1",
            "source_weights": "run1.pt",
            "source_sha256": "abc",
            "released_at": "2026-01-15T12:00:00+00:00",
        },
        "training": {
            "training_info": {
                "model": model_name,
                "task_type": task_type,
                "dataset": {"name": dataset},
            }
        },
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


def test_discover_release_models_skips_registry_bundle(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))

    release_pt = _write_release_model(tmp_path, stem="detect_yolo_20260115_120000")

    bundle = tmp_path / "models" / "promoted_bundle"
    (bundle / "models").mkdir(parents=True)
    (bundle / "model_manifest.json").write_text(
        json.dumps({"friendly_name": "promoted_bundle", "weights_file": "models/run1.pt"}),
        encoding="utf-8",
    )
    registry_pt = bundle / "models" / "run1.pt"
    registry_pt.write_bytes(b"registry")
    (bundle / "models" / "run1.json").write_text(
        json.dumps(
            {
                "source": {"source_run": "/x"},
                "artifacts": {"release_dir": "/x"},
            }
        ),
        encoding="utf-8",
    )

    found = discover_release_models(layout)
    assert len(found) == 1
    assert found[0].pt_path == release_pt.resolve()


def test_rename_release_stem_updates_core_files(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    old_stem = "detect_yolov8n_20260115_120000"
    new_stem = "my_detector_v2"
    pt_path = _write_release_model(tmp_path, stem=old_stem)

    entry = discover_release_models(layout)[0]
    plan = build_rename_plan(entry, new_stem)
    result = apply_release_rename(plan)

    assert not result.skipped
    assert (tmp_path / "models" / "ds1" / f"{new_stem}.pt").is_file()
    assert (tmp_path / "models" / "ds1" / f"{new_stem}.json").is_file()
    assert (tmp_path / "models" / "ds1" / new_stem).is_dir()
    assert not pt_path.exists()
    assert not (tmp_path / "models" / "ds1" / f"{old_stem}.json").exists()

    payload = json.loads((tmp_path / "models" / "ds1" / f"{new_stem}.json").read_text(encoding="utf-8"))
    assert is_release_metadata(payload)
    assert payload["artifacts"]["model_path"].endswith(f"{new_stem}.pt")
    assert payload["artifacts"]["release_dir"].endswith(new_stem)


def test_rename_release_stem_updates_converted_artifacts(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    old_stem = "detect_yolov8n_20260115_120000"
    new_stem = "detect_yolov8s_20260115_120000"
    pt_path = _write_release_model(tmp_path, stem=old_stem)
    parent = pt_path.parent
    onnx_name = f"{old_stem}_imgsz640x640_b1_static_op17_fp32_simplify1_nms0.onnx"
    onnx_path = parent / onnx_name
    onnx_path.write_bytes(b"onnx")
    meta_path = parent / f"{onnx_name}.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "format": "onnx",
                "path": str(onnx_path),
                "filename": onnx_name,
                "source_path": str(pt_path),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    entry = discover_release_models(layout)[0]
    plan = build_rename_plan(entry, new_stem)
    apply_release_rename(plan)

    new_onnx = parent / onnx_name.replace(old_stem, new_stem)
    new_meta = parent / f"{new_onnx.name}.meta.json"
    assert new_onnx.is_file()
    assert new_meta.is_file()
    assert not onnx_path.exists()
    meta_payload = json.loads(new_meta.read_text(encoding="utf-8"))
    assert meta_payload["filename"] == new_onnx.name
    assert meta_payload["path"].endswith(new_onnx.name)
    assert meta_payload["source_path"].endswith(f"{new_stem}.pt")


def test_rename_conflict_raises(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    _write_release_model(tmp_path, stem="detect_a_20260115_120000")
    conflict_pt = tmp_path / "models" / "ds1" / "detect_b_20260115_120000.pt"
    conflict_pt.write_bytes(b"other")

    entry = discover_release_models(layout)[0]
    with pytest.raises(ReleaseRenameError, match="target already exists"):
        build_rename_plan(entry, "detect_b_20260115_120000")


def test_rename_noop_same_stem(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    stem = "detect_yolov8n_20260115_120000"
    _write_release_model(tmp_path, stem=stem)

    entry = discover_release_models(layout)[0]
    plan = build_rename_plan(entry, stem)
    result = apply_release_rename(plan)
    assert result.skipped
    assert result.reason


def test_rename_parses_canonical_stem_updates_training_info(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    old_stem = "detect_yolov8n_20260115_120000"
    new_stem = "detect_newmodel_20260115_120000"
    _write_release_model(tmp_path, stem=old_stem, model_name="yolov8n")

    entry = discover_release_models(layout)[0]
    plan = build_rename_plan(entry, new_stem)
    apply_release_rename(plan)

    release_json = json.loads(
        (tmp_path / "models" / "ds1" / f"{new_stem}.json").read_text(encoding="utf-8")
    )
    assert release_json["training"]["training_info"]["model"] == "newmodel"

    train_meta = json.loads(
        (tmp_path / "models" / "ds1" / new_stem / "training_metadata.json").read_text(encoding="utf-8")
    )
    assert train_meta["training_info"]["model"] == "newmodel"


def test_model_rename_cli_non_interactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    stem = "detect_yolov8n_20260115_120000"
    pt_path = _write_release_model(tmp_path, stem=stem)
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    mrc.main(["--workspace", str(tmp_path), "--release", str(pt_path), "--new-name", "my_detector"])

    assert (tmp_path / "models" / "ds1" / "my_detector.pt").is_file()


def test_model_rename_cli_same_name_exits_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    deploy_workspace(str(tmp_path))
    stem = "detect_yolov8n_20260115_120000"
    pt_path = _write_release_model(tmp_path, stem=stem)
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    with pytest.raises(SystemExit) as exc:
        mrc.main(["--workspace", str(tmp_path), "--release", str(pt_path), "--new-name", stem])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Nothing to do" in out
    assert pt_path.is_file()
