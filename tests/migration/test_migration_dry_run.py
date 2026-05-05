from __future__ import annotations

import json
from pathlib import Path

from smartrain.cli_migration import run_migration
from smartrain.workspace_paths import deploy_workspace


def test_dry_run_does_not_write_canonical_snapshot(tmp_path: Path) -> None:
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
    assert not (run_dir / ".smartrain" / "canonical" / "snapshot.json").exists()

