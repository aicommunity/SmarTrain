"""Tests for dataset rename (workspace catalog)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from smartrain.core.runtime.workspace_paths import WorkspaceLayout, deploy_workspace
from smartrain.services.datasets.dataset_rename_service import (
    DatasetRenameError,
    apply_dataset_rename,
    build_rename_plan,
    format_plan_report,
)
from smartrain.workflows.datasets import dataset_rename_cli as drc


def _write_catalog(workspace: Path, name: str, *, structure: str = "flat") -> None:
    datasets_dir = workspace / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = datasets_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "data.yaml").write_text("names: [a]\n", encoding="utf-8")
    info_path = datasets_dir / "datasets_info.json"
    catalog = {}
    if info_path.is_file():
        catalog = json.loads(info_path.read_text(encoding="utf-8"))
    catalog[name] = {
        "classes": {"a": 0},
        "structure": structure,
        "elements_count": 1,
        "data_path": f"datasets/{name}",
    }
    info_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=4), encoding="utf-8")


def test_rename_dataset_updates_catalog_and_directory(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _write_catalog(tmp_path, "old_ds")
    layout = WorkspaceLayout(str(tmp_path))

    plan = build_rename_plan(layout, "old_ds", "new_ds")
    result = apply_dataset_rename(plan)

    assert not result.skipped
    assert (tmp_path / "datasets" / "new_ds").is_dir()
    assert not (tmp_path / "datasets" / "old_ds").exists()
    catalog = json.loads((tmp_path / "datasets" / "datasets_info.json").read_text(encoding="utf-8"))
    assert "new_ds" in catalog
    assert "old_ds" not in catalog
    assert catalog["new_ds"]["data_path"] == "datasets/new_ds"


def test_rename_conflict_raises(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _write_catalog(tmp_path, "ds_a")
    _write_catalog(tmp_path, "ds_b")
    layout = WorkspaceLayout(str(tmp_path))

    with pytest.raises(DatasetRenameError, match="already exists"):
        build_rename_plan(layout, "ds_a", "ds_b")


def test_rename_noop_same_name(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _write_catalog(tmp_path, "same_ds")
    layout = WorkspaceLayout(str(tmp_path))

    plan = build_rename_plan(layout, "same_ds", "same_ds")
    result = apply_dataset_rename(plan)
    assert result.skipped


def test_rename_moves_runs_and_models(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _write_catalog(tmp_path, "old_ds")
    (tmp_path / "runs" / "old_ds" / "2026-01-01_00-00-00").mkdir(parents=True)
    (tmp_path / "models" / "old_ds").mkdir(parents=True)
    layout = WorkspaceLayout(str(tmp_path))

    plan = build_rename_plan(layout, "old_ds", "new_ds")
    apply_dataset_rename(plan)

    assert (tmp_path / "runs" / "new_ds").is_dir()
    assert not (tmp_path / "runs" / "old_ds").exists()
    assert (tmp_path / "models" / "new_ds").is_dir()
    assert not (tmp_path / "models" / "old_ds").exists()


def test_rename_updates_passport_and_queue(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _write_catalog(tmp_path, "old_ds")
    _write_catalog(tmp_path, "child_ds")
    child_dir = tmp_path / "datasets" / "child_ds"
    passport = {
        "command": "augment",
        "source_dataset": [{"name": "old_ds", "path": "datasets/old_ds"}],
        "created_dataset": {"name": "child_ds", "path": "datasets/child_ds"},
        "parameters": {"dataset": "old_ds"},
    }
    (child_dir / "dataset_passport.json").write_text(json.dumps(passport), encoding="utf-8")
    (tmp_path / "queue.txt").write_text(
        'smartrain train --data old_ds -y\nsmartrain augment --dataset old_ds --output-name child_ds\n',
        encoding="utf-8",
    )
    layout = WorkspaceLayout(str(tmp_path))

    plan = build_rename_plan(layout, "old_ds", "new_ds")
    apply_dataset_rename(plan)

    updated_passport = json.loads((child_dir / "dataset_passport.json").read_text(encoding="utf-8"))
    assert updated_passport["source_dataset"][0]["name"] == "new_ds"
    assert updated_passport["parameters"]["dataset"] == "new_ds"
    queue_text = (tmp_path / "queue.txt").read_text(encoding="utf-8")
    assert "--data new_ds" in queue_text
    assert "--dataset new_ds" in queue_text
    assert "old_ds" not in queue_text


def test_rename_updates_run_metadata(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _write_catalog(tmp_path, "old_ds")
    run_dir = tmp_path / "runs" / "old_ds" / "2026-01-01_00-00-00"
    run_dir.mkdir(parents=True)
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "old_ds"}}}),
        encoding="utf-8",
    )
    (run_dir / "args.yaml").write_text(yaml.safe_dump({"data": "datasets/old_ds/data.yaml"}), encoding="utf-8")
    layout = WorkspaceLayout(str(tmp_path))

    plan = build_rename_plan(layout, "old_ds", "new_ds")
    apply_dataset_rename(plan)

    new_run_dir = tmp_path / "runs" / "new_ds" / "2026-01-01_00-00-00"
    meta = json.loads((new_run_dir / "training_metadata.json").read_text(encoding="utf-8"))
    assert meta["training_info"]["dataset"]["name"] == "new_ds"
    args = yaml.safe_load((new_run_dir / "args.yaml").read_text(encoding="utf-8"))
    assert args["data"] == "datasets/new_ds/data.yaml"


def test_dry_run_does_not_modify_filesystem(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _write_catalog(tmp_path, "old_ds")
    layout = WorkspaceLayout(str(tmp_path))

    plan = build_rename_plan(layout, "old_ds", "new_ds")
    apply_dataset_rename(plan, dry_run=True)

    assert (tmp_path / "datasets" / "old_ds").is_dir()
    catalog = json.loads((tmp_path / "datasets" / "datasets_info.json").read_text(encoding="utf-8"))
    assert "old_ds" in catalog


def test_custom_data_path_requires_move_flag(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    custom_dir = tmp_path / "external" / "old_ds"
    custom_dir.mkdir(parents=True)
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    catalog = {
        "old_ds": {
            "classes": {"a": 0},
            "structure": "flat",
            "elements_count": 1,
            "data_path": "external/old_ds",
        }
    }
    (datasets_dir / "datasets_info.json").write_text(json.dumps(catalog), encoding="utf-8")
    layout = WorkspaceLayout(str(tmp_path))

    with pytest.raises(DatasetRenameError, match="custom data_path"):
        build_rename_plan(layout, "old_ds", "new_ds")


def test_format_plan_report_lists_operations(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _write_catalog(tmp_path, "old_ds")
    (tmp_path / "runs" / "old_ds").mkdir(parents=True)
    layout = WorkspaceLayout(str(tmp_path))

    plan = build_rename_plan(layout, "old_ds", "new_ds")
    report = format_plan_report(plan)
    assert "old_ds -> new_ds" in report
    assert "datasets/old_ds" in report


def test_dataset_rename_cli_non_interactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    _write_catalog(tmp_path, "old_ds")
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    drc.main(["--workspace", str(tmp_path), "--dataset", "old_ds", "--new-name", "new_ds"])

    assert (tmp_path / "datasets" / "new_ds").is_dir()


def test_dataset_rename_cli_same_name_exits_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    deploy_workspace(str(tmp_path))
    _write_catalog(tmp_path, "same_ds")
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")

    with pytest.raises(SystemExit) as exc:
        drc.main(["--workspace", str(tmp_path), "--dataset", "same_ds", "--new-name", "same_ds"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Nothing to do" in out
