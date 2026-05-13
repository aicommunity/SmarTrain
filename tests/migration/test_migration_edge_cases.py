from __future__ import annotations

import json
from pathlib import Path

from smartrain.core.runtime.workspace_paths import deploy_workspace
from smartrain.workflows.migration.cli_migration import run_migration


def test_apply_run_without_training_metadata(tmp_path: Path) -> None:
    """Run layout discoverable via train/weights only (no training_metadata.json)."""
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_nometa" / "run_nometa"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "last.pt").write_bytes(b"x")
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "models" / f"{run_dir.name}.pt").write_bytes(b"x")
    report = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="apply",
        runs_root=None,
        models_root=None,
        continue_on_error=False,
    )
    assert report["stats"]["migrated"] == 1
    assert (run_dir / ".smartrain" / "canonical" / "snapshot.json").is_file()


def test_apply_partial_training_info(tmp_path: Path) -> None:
    """Minimal training_metadata still migrates."""
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_partial" / "run_partial"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text(json.dumps({"training_info": {}}), encoding="utf-8")
    report = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="apply",
        runs_root=None,
        models_root=None,
        continue_on_error=False,
    )
    assert report["stats"]["migrated"] == 1
    assert (run_dir / ".smartrain" / "canonical" / "manifest.json").is_file()


def test_apply_nested_weights_layout(tmp_path: Path) -> None:
    """Weights only under train/weights (non-canonical models/ path)."""
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_nested" / "run_nested"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds_nested"}, "task_type": "detection"}}),
        encoding="utf-8",
    )
    report = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="apply",
        runs_root=None,
        models_root=None,
        continue_on_error=False,
    )
    assert report["stats"]["migrated"] == 1


def test_apply_second_time_skipped_up_to_date(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_idem" / "run_idem"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds_idem"}}}),
        encoding="utf-8",
    )
    r1 = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="apply",
        runs_root=None,
        models_root=None,
        continue_on_error=False,
    )
    assert r1["stats"]["migrated"] == 1
    r2 = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="apply",
        runs_root=None,
        models_root=None,
        continue_on_error=False,
    )
    assert r2["stats"]["migrated"] == 0
    assert r2["stats"]["skipped"] == 1
