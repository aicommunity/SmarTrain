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


def _cvat11_dataset(root: Path, name: str) -> None:
    ds = root / name
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "images" / "img001.jpg").write_bytes(b"\x00")
    (ds / "annotations.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><labels><label><name>bee</name></label></labels></task></meta>
  <image id="0" name="img001.jpg" width="100" height="80">
    <box label="bee" xtl="10" ytl="10" xbr="40" ybr="30"/>
  </image>
</annotations>
""",
        encoding="utf-8",
    )


def test_datasets_json_writes_scan_summary_and_diff(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    sd = tmp_path / "raw_data"
    _flat_dataset(sd, "ds_a")

    datasets_json_main(["--workspace", str(tmp_path)])
    sum1 = tmp_path / "datasets" / DATASETS_SCAN_SUMMARY_FILE
    assert sum1.is_file()
    j1 = json.loads(sum1.read_text(encoding="utf-8"))
    assert j1["datasets"]["final"] == ["ds_a"]
    assert j1["datasets"]["added"] == ["ds_a"]
    assert j1["datasets"]["removed"] == []
    assert "bee" in j1["class_names"]["final"]

    _flat_dataset(sd, "ds_b")
    (sd / "ds_b" / "data.yaml").write_text("nc: 1\nnames: ['antelope']\n", encoding="utf-8")

    datasets_json_main(["--workspace", str(tmp_path)])
    j2 = json.loads(sum1.read_text(encoding="utf-8"))
    assert set(j2["datasets"]["final"]) == {"ds_a", "ds_b"}
    assert j2["datasets"]["added"] == ["ds_b"]
    assert j2["datasets"]["removed"] == []
    assert "antelope" in j2["class_names"]["added"]

    import shutil

    shutil.rmtree(sd / "ds_b")
    datasets_json_main(["--workspace", str(tmp_path)])
    j3 = json.loads(sum1.read_text(encoding="utf-8"))
    # raw_data — только источник обновлений; удаление в raw_data НЕ удаляет датасеты из datasets.
    assert set(j3["datasets"]["final"]) == {"ds_a", "ds_b"}
    assert j3["datasets"]["removed"] == []


def test_scan_skips_duplicate_content_with_other_name(tmp_path: Path, capsys) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _flat_dataset(rd, "ds_a")
    _flat_dataset(rd, "ds_b")
    # Make ds_b fully identical to ds_a content.
    (rd / "ds_b" / "data.yaml").write_text("nc: 1\nnames: ['bee']\n", encoding="utf-8")

    datasets_json_main(["--workspace", str(tmp_path)])
    out = capsys.readouterr().out
    info = json.loads((tmp_path / "datasets" / DATASETS_INFO_FILE).read_text(encoding="utf-8"))
    assert len(info) == 1
    assert set(info.keys()) in ({"ds_a"}, {"ds_b"})
    assert "совпадают с datasets" in out


def test_scan_marks_modified_and_stops_sync_for_dataset(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _flat_dataset(rd, "ds_a")

    datasets_json_main(["--workspace", str(tmp_path)])
    info_path = tmp_path / "datasets" / DATASETS_INFO_FILE
    info1 = json.loads(info_path.read_text(encoding="utf-8"))
    initial_hash = info1["ds_a"]["dataset_hash"]
    assert info1["ds_a"]["modified"] is False

    # Manual change in datasets should mark modified=true.
    ds_label = tmp_path / "datasets" / "ds_a" / "labels" / "a.txt"
    ds_label.write_text("0 0.4 0.4 0.25 0.25\n", encoding="utf-8")
    datasets_json_main(["--workspace", str(tmp_path)])
    info2 = json.loads(info_path.read_text(encoding="utf-8"))
    assert info2["ds_a"]["modified"] is True
    assert info2["ds_a"]["dataset_hash"] != initial_hash

    # Change source in raw_data, but sync must be blocked by modified=true.
    raw_label = rd / "ds_a" / "labels" / "a.txt"
    raw_label.write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    datasets_json_main(["--workspace", str(tmp_path)])
    info3 = json.loads(info_path.read_text(encoding="utf-8"))
    assert info3["ds_a"]["modified"] is True
    assert info3["ds_a"]["dataset_hash"] == info2["ds_a"]["dataset_hash"]


def test_scan_converts_cvat11_to_training_ready_layout(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _cvat11_dataset(rd, "cvat_src")

    datasets_json_main(["--workspace", str(tmp_path)])
    out_root = tmp_path / "datasets" / "cvat_src"
    assert (out_root / "data.yaml").is_file()
    assert (out_root / "labels" / "img001.txt").is_file()
    yaml_text = (out_root / "data.yaml").read_text(encoding="utf-8")
    assert "train: ./images" in yaml_text
    assert "names:" in yaml_text
