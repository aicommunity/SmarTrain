from __future__ import annotations

import json
from pathlib import Path

from smartrain.datasets_json_former import main as datasets_json_main
from smartrain.workspace_paths import (
    CLASS_NAMES_FILE,
    DATASETS_INFO_FILE,
    DATASETS_SCAN_SUMMARY_FILE,
    deploy_workspace,
)


def _flat_dataset(root: Path, name: str) -> None:
    ds = root / name
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "images" / "a.jpg").write_bytes(b"\x00")
    (ds / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (ds / "data.yaml").write_text("nc: 1\nnames: ['bee']\n", encoding="utf-8")


def test_datasets_json_writes_scan_summary_and_diff(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    sd = tmp_path / "source_datasets"
    _flat_dataset(sd, "ds_a")

    datasets_json_main(["--workspace", str(tmp_path)])
    sum1 = sd / DATASETS_SCAN_SUMMARY_FILE
    assert sum1.is_file()
    j1 = json.loads(sum1.read_text(encoding="utf-8"))
    assert j1["datasets"]["final"] == ["ds_a"]
    assert j1["datasets"]["added"] == ["ds_a"]
    assert j1["datasets"]["removed"] == []
    assert "bee" in j1["class_names"]["final"]

    _flat_dataset(sd, "ds_b")
    (sd / "ds_b" / "data.yaml").write_text("nc: 1\nnames: ['ant']\n", encoding="utf-8")

    datasets_json_main(["--workspace", str(tmp_path)])
    j2 = json.loads(sum1.read_text(encoding="utf-8"))
    assert set(j2["datasets"]["final"]) == {"ds_a", "ds_b"}
    assert j2["datasets"]["added"] == ["ds_b"]
    assert j2["datasets"]["removed"] == []
    assert "ant" in j2["class_names"]["added"]

    import shutil

    shutil.rmtree(sd / "ds_b")
    datasets_json_main(["--workspace", str(tmp_path)])
    j3 = json.loads(sum1.read_text(encoding="utf-8"))
    assert j3["datasets"]["final"] == ["ds_a"]
    assert j3["datasets"]["removed"] == ["ds_b"]
    assert "ant" in j3["class_names"]["removed"]
