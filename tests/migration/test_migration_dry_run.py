from __future__ import annotations

import json
from pathlib import Path

from smartrain.workflows.migration.cli_migration import run_migration
from smartrain.core.runtime.workspace_paths import deploy_workspace


def test_dry_run_does_not_write_unified_snapshot(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_b"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds_a"}}}),
        encoding="utf-8",
    )
    report = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="dry-run",
        runs_root=None,
        models_root=None,
        continue_on_error=False,
    )
    assert report["stats"]["planned"] == 1
    assert not (run_dir / ".smartrain" / "unified" / "snapshot.json").exists()


def test_report_only_marks_planned_and_sets_rollback_hint(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_report_only"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds_a"}}}),
        encoding="utf-8",
    )

    report = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="report-only",
        runs_root=None,
        models_root=None,
        continue_on_error=False,
    )
    assert report["stats"]["planned"] == 1
    item = report["items"][0]
    assert item["status"] == "planned"
    assert item.get("rollback_hint")
    assert not (run_dir / ".smartrain" / "unified" / "snapshot.json").exists()


def test_dry_run_respects_runs_root_scope(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_in_scope = tmp_path / "custom_runs" / "ds_scope" / "run_scope"
    (run_in_scope / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_in_scope / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_in_scope / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds_scope"}}}),
        encoding="utf-8",
    )
    run_out_scope = tmp_path / "runs" / "ds_out" / "run_out"
    (run_out_scope / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_out_scope / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_out_scope / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds_out"}}}),
        encoding="utf-8",
    )

    report = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="dry-run",
        runs_root=str(tmp_path / "custom_runs"),
        models_root=None,
        continue_on_error=False,
    )
    assert report["stats"]["planned"] == 1
    assert Path(report["items"][0]["ref"]).as_posix().endswith("custom_runs/ds_scope/run_scope")

