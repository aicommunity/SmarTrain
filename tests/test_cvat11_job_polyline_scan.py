from __future__ import annotations

import json
from pathlib import Path

from smartrain.services.datasets.cvat11_converter import (
    generate_temp_yolo_labels_from_cvat11_extracted,
    load_cvat11_label_names_from_xml,
)
from smartrain.services.datasets.datasets_json_scan_core_service import (
    _load_cvat11_label_names,
    process_dataset,
)


def _job_polyline_cvat_dataset(root: Path, name: str) -> None:
    ds = root / name
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "images" / "img001.jpg").write_bytes(b"\xff\xd8\xff")
    (ds / "annotations.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <job>
      <id>2549</id>
      <labels>
        <label><name>belt_side</name></label>
      </labels>
    </job>
  </meta>
  <image id="0" name="img001.jpg" width="100" height="80">
    <polyline label="belt_side" points="10.0,10.0;40.0,10.0;40.0,30.0;10.0,30.0"/>
  </image>
</annotations>
""",
        encoding="utf-8",
    )


def test_load_cvat11_label_names_job_export() -> None:
    xml = """<?xml version="1.0"?><annotations><meta><job><labels>
    <label><name>belt_side</name></label></labels></job></meta></annotations>"""
    path = Path("/tmp/cvat_test_labels.xml")
    path.write_text(xml, encoding="utf-8")
    assert _load_cvat11_label_names(str(path)) == ["belt_side"]
    assert load_cvat11_label_names_from_xml(path) == ["belt_side"]


def test_process_dataset_cvat11_job_polyline(tmp_path: Path) -> None:
    _job_polyline_cvat_dataset(tmp_path, "cvat_job")
    info = process_dataset(str(tmp_path / "cvat_job"), "cvat_job")
    assert info is not None
    assert info["structure"] == "cvat11"
    assert "belt_side" in info["classes"]


def test_generate_yolo_labels_from_cvat_polyline(tmp_path: Path) -> None:
    _job_polyline_cvat_dataset(tmp_path, "cvat_job")
    ds = tmp_path / "cvat_job"
    labels_dir = ds / "labels"
    generate_temp_yolo_labels_from_cvat11_extracted(
        dataset_root=ds,
        labels_out_dir=labels_dir,
        class_name_to_id={"belt_side": 0},
    )
    lbl = (labels_dir / "img001.txt").read_text(encoding="utf-8").strip()
    parts = lbl.split()
    assert parts[0] == "0"
    assert len(parts) >= 7  # class + 3+ polygon points
