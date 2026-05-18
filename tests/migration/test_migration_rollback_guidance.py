from __future__ import annotations

import json
from pathlib import Path

from smartrain.workflows.migration.cli_migration import run_migration
from smartrain.core.runtime.workspace_paths import deploy_workspace


def test_apply_is_idempotent_after_first_migration(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_c"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds_a"}}}),
        encoding="utf-8",
    )

    first = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="apply",
        runs_root=None,
        models_root=None,
        continue_on_error=False,
    )
    second = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="apply",
        runs_root=None,
        models_root=None,
        continue_on_error=False,
    )
    assert first["stats"]["migrated"] == 1
    assert second["stats"]["skipped"] == 1


def test_report_contains_operator_guidance_and_rollback_hints(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_fail"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds_a"}}}),
        encoding="utf-8",
    )

    def _raise(*args, **kwargs):
        raise RuntimeError("forced failure for guidance")

    monkeypatch.setattr("smartrain.workflows.migration.cli_migration.read_legacy_target", _raise)
    report = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="apply",
        runs_root=None,
        models_root=None,
        continue_on_error=True,
    )
    assert report["stats"]["failed"] == 1
    assert "operator_guidance" in report
    failed = [x for x in report["items"] if x.get("status") == "failed"]
    assert failed and failed[0].get("rollback_hint")


def test_apply_recovers_from_corrupted_existing_snapshot(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_corrupt_snapshot"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds_a"}}}),
        encoding="utf-8",
    )
    snap_dir = run_dir / ".smartrain" / "unified"
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "snapshot.json").write_text("{bad-json", encoding="utf-8")

    report = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="apply",
        runs_root=None,
        models_root=None,
        continue_on_error=False,
    )
    assert report["stats"]["migrated"] == 1
    payload = json.loads((snap_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


def test_apply_continue_on_error_false_stops_on_first_failure(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    run_fail = tmp_path / "runs" / "ds_a" / "run_fail_first"
    (run_fail / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_fail / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_fail / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds_a"}}}),
        encoding="utf-8",
    )
    run_second = tmp_path / "runs" / "ds_b" / "run_second"
    (run_second / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_second / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_second / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds_b"}}}),
        encoding="utf-8",
    )

    def _raise(*args, **kwargs):
        raise RuntimeError("forced stop")

    monkeypatch.setattr("smartrain.workflows.migration.cli_migration.read_legacy_target", _raise)
    report = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="apply",
        runs_root=None,
        models_root=None,
        continue_on_error=False,
    )
    assert report["stats"]["failed"] == 1
    assert report["stats"]["total"] == 1
    assert not (run_second / ".smartrain" / "unified" / "snapshot.json").exists()

