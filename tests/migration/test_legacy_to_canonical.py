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

