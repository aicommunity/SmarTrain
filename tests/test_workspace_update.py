"""Tests for smartrain update scanner/appliers and unified rename fix."""

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
from smartrain.services.models.release_models_manifest import entry_key_for_pt, upsert_entry
from smartrain.services.update.plan import UpdateCategory, parse_categories
from smartrain.services.update.scanner import scan_workspace
from smartrain.workflows.update import update_cli as uc


def _release_sidecar(workspace: Path, release_dir: Path, pt_path: Path, stem: str) -> None:
    json_path = pt_path.with_suffix(".json")
    payload = {
        "comment": "",
        "source": {
            "source_run": str(workspace / "runs" / "ds" / "r"),
            "source_run_relative": "runs/ds/r",
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
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_parse_categories() -> None:
    assert UpdateCategory.LAYOUT in parse_categories("layout,yaml")
    assert len(parse_categories(None)) == len(UpdateCategory)


def test_scan_finds_legacy_train_and_root_metrics(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    run = tmp_path / "runs" / "ds" / "run1"
    run.mkdir(parents=True)
    (run / "train").mkdir()
    (run / "train" / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
    (run / "training_metadata.json").write_text(
        json.dumps({"training_info": {"model": "yolo11n", "dataset": {"name": "ds"}}}),
        encoding="utf-8",
    )
    (run / "test_metrics.csv").write_text("m\n1\n", encoding="utf-8")
    (run / "models").mkdir()
    (run / "models" / "detect_yolo11n_20260101_000000_640px_1epochs_b1.pt").write_bytes(b"pt")

    plan = scan_workspace(layout)
    actions = {s.action for s in plan.steps}
    assert "migrate_train" in actions or any("train" in s.title for s in plan.steps)
    assert any(s.action == "migrate_tests" for s in plan.steps)


def test_update_dry_run_and_apply_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    run = tmp_path / "runs" / "ds" / "run1"
    run.mkdir(parents=True)
    (run / "train").mkdir()
    (run / "train" / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
    (run / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {
                    "model": "yolo11n",
                    "dataset": {"name": "ds"},
                    "task_type": "detect",
                    "hyperparameters": {"epochs": 1, "batch_size": 1, "image_size": 640},
                },
                "timestamps": {"training": {"start": "2026-01-01T00:00:00+00:00"}},
            }
        ),
        encoding="utf-8",
    )
    (run / "models").mkdir()
    (run / "models" / "detect_yolo11n_20260101_000000_640px_1epochs_b1.pt").write_bytes(b"pt")
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    with pytest.raises(SystemExit) as exc:
        uc.main(["--workspace", str(tmp_path), "--dry-run", "--only", "layout"])
    assert exc.value.code in (0, None) or exc.value.code == 0
    assert (run / "train").is_dir()

    with pytest.raises(SystemExit) as exc2:
        uc.main(["--workspace", str(tmp_path), "--yes", "--only", "layout"])
    assert exc2.value.code in (0, None) or exc2.value.code == 0
    assert (run / "train-ultralytics").is_dir()
    assert not (run / "train").exists()


def test_update_unify_root_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    stem = "detect_yolo11n_20260101_000000_640px_1epochs_b1"
    release_dir = tmp_path / "models" / "ds" / "run1"
    release_dir.mkdir(parents=True)
    pt = release_dir / f"{stem}.pt"
    pt.write_bytes(b"pt")
    (release_dir / f"{stem}.onnx").write_bytes(b"onnx")
    (release_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    _release_sidecar(tmp_path, release_dir, pt, stem)
    upsert_entry(layout, entry_key=f"ds/run1", model_path=pt, comment="note")
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    with pytest.raises(SystemExit) as exc:
        uc.main(["--workspace", str(tmp_path), "--yes", "--only", "releases,manifest"])
    assert exc.value.code in (0, None) or exc.value.code == 0

    nested = release_dir / "models" / f"{stem}.pt"
    assert nested.is_file()
    assert (release_dir / "models" / f"{stem}.onnx").is_file()
    assert not pt.exists()


def test_update_check_fails_on_residual(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    run = tmp_path / "runs" / "ds" / "run1"
    run.mkdir(parents=True)
    (run / "train").mkdir()
    (run / "train" / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
    (run / "training_metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    with pytest.raises(SystemExit) as exc:
        uc.main(["--workspace", str(tmp_path), "--check", "--only", "layout"])
    assert exc.value.code == 1


def test_rename_unified_keeps_files_under_models(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    stem = "detect_yolo11n_20260101_000000_640px_1epochs_b1"
    release_dir = tmp_path / "models" / "ds" / "run_folder"
    models = release_dir / "models"
    models.mkdir(parents=True)
    pt = models / f"{stem}.pt"
    pt.write_bytes(b"pt")
    (models / f"{stem}.onnx").write_bytes(b"onnx")
    (release_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"model": "yolo11n", "dataset": {"name": "ds"}}}),
        encoding="utf-8",
    )
    _release_sidecar(tmp_path, release_dir, pt, stem)
    upsert_entry(layout, entry_key=entry_key_for_pt(pt), model_path=pt, comment="")

    entry = discover_release_models(layout)[0]
    plan = build_rename_plan(entry, "my_detector_v2")
    apply_release_rename(plan, layout=layout)

    new_pt = models / "my_detector_v2.pt"
    assert new_pt.is_file()
    assert (models / "my_detector_v2.onnx").is_file()
    assert not (release_dir / "my_detector_v2.pt").exists()


def test_normalize_yaml_via_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    ds = tmp_path / "datasets" / "ds1"
    ds.mkdir(parents=True)
    (ds / "train" / "images").mkdir(parents=True)
    (ds / "data.yaml").write_text(
        "path: /abs/ds1\ntrain: train/images\nval: train/images\nnames: [a]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    with pytest.raises(SystemExit) as exc:
        uc.main(["--workspace", str(tmp_path), "--yes", "--only", "yaml"])
    assert exc.value.code in (0, None) or exc.value.code == 0
    text = (ds / "data.yaml").read_text(encoding="utf-8")
    assert "path:" not in text


def test_update_removes_duplicate_root_runtime_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    release_dir = tmp_path / "models" / "ds" / "run1"
    (release_dir / "tmp").mkdir(parents=True)
    (release_dir / "models").mkdir(parents=True)
    (release_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    body = "train: train/images\nval: valid/images\nnames: [a]\n"
    (release_dir / "_runtime_data_train.yaml").write_text(body, encoding="utf-8")
    (release_dir / "tmp" / "_runtime_data_train.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    with pytest.raises(SystemExit) as exc:
        uc.main(["--workspace", str(tmp_path), "--yes", "--only", "layout"])
    assert exc.value.code in (0, None) or exc.value.code == 0
    assert not (release_dir / "_runtime_data_train.yaml").exists()
    assert (release_dir / "tmp" / "_runtime_data_train.yaml").is_file()


def test_update_rewrites_posix_abs_sidecar_to_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    stem = "detect_yolo11n_20260101_000000_640px_1epochs_b1"
    release_dir = tmp_path / "models" / "ds" / "run1"
    models = release_dir / "models"
    models.mkdir(parents=True)
    (release_dir / "train-ultralytics").mkdir(parents=True)
    (release_dir / "tests").mkdir(parents=True)
    (tmp_path / "runs" / "ds" / "src_run").mkdir(parents=True)
    pt = models / f"{stem}.pt"
    pt.write_bytes(b"pt")
    (release_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    foreign_root = "/data/NextCloud/PROJECT/OtherHost/UralkSmarTrain"
    sidecar = {
        "comment": "",
        "source": {
            "source_run": f"{foreign_root}/runs/ds/src_run",
            "source_run_relative": "runs\\ds\\src_run",
            "source_weights": f"{stem}.pt",
            "source_sha256": "abc",
            "released_at": "2026-01-15T12:00:00+00:00",
        },
        "training": {},
        "artifacts": {
            "model_path": f"{foreign_root}/models/ds/run1/models/{stem}.pt",
            "json_path": f"{foreign_root}/models/ds/run1/models/{stem}.json",
            "release_dir": f"{foreign_root}/models/ds/run1",
            "train_copy_dir": f"{foreign_root}/models/ds/run1/train",
            "test_copy_dir": f"{foreign_root}/models/ds/run1/test",
        },
    }
    (models / f"{stem}.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    upsert_entry(layout, entry_key=entry_key_for_pt(pt), model_path=pt, comment="")
    # Force Windows-style separators into manifest (simulates pre-fix upsert)
    manifest_path = tmp_path / "models" / "releases_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = next(iter(payload["entries"]))
    payload["entries"][key]["model_path"] = str(Path("models") / "ds" / "run1" / "models" / f"{stem}.pt")
    assert "\\" in payload["entries"][key]["model_path"] or payload["entries"][key]["model_path"].startswith("models")
    # Ensure backslash form on Windows
    payload["entries"][key]["model_path"] = f"models\\ds\\run1\\models\\{stem}.pt"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    plan = scan_workspace(layout)
    assert any(s.action == "rewrite_sidecar_paths" for s in plan.steps)
    assert any(s.action == "relativize_manifest_path" for s in plan.steps)

    with pytest.raises(SystemExit) as exc:
        uc.main(["--workspace", str(tmp_path), "--yes", "--apply-all", "--only", "manifest"])
    assert exc.value.code in (0, None) or exc.value.code == 0

    rewritten = json.loads((models / f"{stem}.json").read_text(encoding="utf-8"))
    art = rewritten["artifacts"]
    assert art["model_path"] == f"models/ds/run1/models/{stem}.pt"
    assert art["json_path"] == f"models/ds/run1/models/{stem}.json"
    assert art["release_dir"] == "models/ds/run1"
    assert art["train_copy_dir"] == "models/ds/run1/train-ultralytics"
    assert art["test_copy_dir"] == "models/ds/run1/tests"
    assert rewritten["source"]["source_run"] == "runs/ds/src_run"
    assert "\\" not in art["model_path"]
    assert not art["model_path"].startswith("/")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mp = manifest["entries"][key]["model_path"]
    assert mp == f"models/ds/run1/models/{stem}.pt"
    assert "\\" not in mp

    with pytest.raises(SystemExit) as exc2:
        uc.main(["--workspace", str(tmp_path), "--check", "--only", "manifest"])
    assert exc2.value.code in (0, None) or exc2.value.code == 0


def test_update_normalizes_runtime_yaml_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    run = tmp_path / "runs" / "ds" / "run1"
    (run / "tmp").mkdir(parents=True)
    (run / "training_metadata.json").write_text("{}", encoding="utf-8")
    (run / "tmp" / "_runtime_data_train.yaml").write_text(
        "path: /data/NextCloud/PROJECT/Other/UralkSmarTrain/datasets/ds\n"
        "train: train/images\n"
        "val: valid/images\n"
        "names: [a]\n",
        encoding="utf-8",
    )
    (tmp_path / "datasets" / "ds" / "train" / "images").mkdir(parents=True)
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    with pytest.raises(SystemExit) as exc:
        uc.main(["--workspace", str(tmp_path), "--yes", "--only", "yaml"])
    assert exc.value.code in (0, None) or exc.value.code == 0
    text = (run / "tmp" / "_runtime_data_train.yaml").read_text(encoding="utf-8")
    assert "path:" not in text
    assert "\\" not in text
