from __future__ import annotations

import json
from pathlib import Path

from smartrain.cli_migration import run_migration
from smartrain.workspace_paths import deploy_workspace


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

