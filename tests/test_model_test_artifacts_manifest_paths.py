from __future__ import annotations

import json
from pathlib import Path

from smartrain.services.testing.model_test_service import (
    sync_test_artifacts_manifest,
    update_test_artifacts_manifest,
)


def test_update_test_artifacts_manifest_paths_are_posix_relative(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "ds" / "r1"
    root.mkdir(parents=True)
    models = root / "models"
    models.mkdir()
    target = models / "a.pt"
    target.write_bytes(b"pt")
    yaml_path = root / "tmp" / "data.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("path: datasets/ds\n", encoding="utf-8")
    # Create test dir so status records relative test_dir
    (root / "tests" / "test-ultralytics").mkdir(parents=True)

    update_test_artifacts_manifest(
        str(root),
        format_name="pt",
        target_path=str(target.resolve()),
        dataset_yaml=str(yaml_path.resolve()),
        backend="ultralytics",
    )
    sync_test_artifacts_manifest(str(root))
    man_path = root / "tests" / "test_artifacts_manifest.json"
    assert man_path.is_file()
    payload = json.loads(man_path.read_text(encoding="utf-8"))
    fmt = payload["formats"]["pt"]
    for key in ("target_path", "dataset_yaml", "test_dir", "metrics_csv"):
        val = fmt.get(key)
        if val is None:
            continue
        assert "\\" not in val, f"{key}={val!r}"
        assert not Path(val).is_absolute(), f"{key}={val!r}"
