from __future__ import annotations

import json
from pathlib import Path

from smartrain.workflows.migration.cli_migration import run_migration
from smartrain.core.runtime.workspace_paths import deploy_workspace


def _mk_run(root: Path, dataset: str, run_name: str) -> Path:
    rd = root / "runs" / dataset / run_name
    (rd / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (rd / "train" / "weights" / "best.pt").write_bytes(b"x")
    (rd / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": dataset}, "task_type": "detection"}}),
        encoding="utf-8",
    )
    return rd


def test_apply_migrates_legacy_run_to_canonical_snapshot(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = _mk_run(tmp_path, "ds_a", "run_a")
    report = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="apply",
        runs_root=None,
        models_root=None,
        continue_on_error=False,
    )
    assert report["stats"]["migrated"] == 1
    snap = run_dir / ".smartrain" / "canonical" / "snapshot.json"
    man = run_dir / ".smartrain" / "canonical" / "manifest.json"
    assert snap.is_file()
    assert man.is_file()


def test_apply_continue_on_error_migrates_other_targets(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    run_ok = _mk_run(tmp_path, "ds_a", "run_ok")
    run_fail = _mk_run(tmp_path, "ds_b", "run_fail")

    from smartrain.workflows.migration import cli_migration as mig

    _orig = mig.read_legacy_target

    def _patched(ref: str, source_kind: str):
        if str(Path(ref).resolve()) == str(run_fail.resolve()):
            raise RuntimeError("forced read failure")
        return _orig(ref, source_kind=source_kind)

    monkeypatch.setattr(mig, "read_legacy_target", _patched)
    report = run_migration(
        workspace=str(tmp_path),
        source_kind="run",
        mode="apply",
        runs_root=None,
        models_root=None,
        continue_on_error=True,
    )
    assert report["stats"]["migrated"] == 1
    assert report["stats"]["failed"] == 1
    assert (run_ok / ".smartrain" / "canonical" / "snapshot.json").is_file()
    assert not (run_fail / ".smartrain" / "canonical" / "snapshot.json").is_file()


def test_apply_model_source_kind_discovers_model_directories(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    model_dir = tmp_path / "models" / "export_a"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best.onnx").write_bytes(b"model")
    (model_dir / "model_manifest.json").write_text(
        '{"task_type": "detection", "backend_type": "onnxruntime"}',
        encoding="utf-8",
    )

    report = run_migration(
        workspace=str(tmp_path),
        source_kind="model",
        mode="apply",
        runs_root=None,
        models_root=None,
        continue_on_error=False,
    )
    assert report["stats"]["migrated"] == 1
    assert (model_dir / ".smartrain" / "canonical" / "snapshot.json").is_file()

