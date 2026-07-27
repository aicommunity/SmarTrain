from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from smartrain.services.datasets.datasets_json_scan_core_service import detect_structure, detect_structures


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), color=(10, 20, 30)).save(path, format="JPEG", quality=90)


def _write_yolo_split(root: Path) -> None:
    for subset in ("train", "val"):
        images = root / subset / "images"
        labels = root / subset / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        _write_jpg(images / f"{subset}_001.jpg")
        (labels / f"{subset}_001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "data.yaml").write_text("names: ['cat']\n", encoding="utf-8")


def _write_cvat11(root: Path) -> None:
    images = root / "images"
    images.mkdir(parents=True)
    _write_jpg(images / "img001.jpg")
    ann = root / "annotations.xml"
    ann.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><labels><label><name>cat</name></label></labels></task></meta>
  <image id="0" name="img001.jpg" width="80" height="60">
    <box label="cat" xtl="10" ytl="10" xbr="40" ybr="40" occluded="0" z_order="0"></box>
  </image>
</annotations>
""",
        encoding="utf-8",
    )


def test_detect_structures_cvat11_and_split(tmp_path: Path) -> None:
    root = tmp_path / "dual"
    root.mkdir()
    _write_yolo_split(root)
    _write_cvat11(root)

    structures = detect_structures(str(root))
    assert "split" in structures
    assert "cvat11" in structures
    assert detect_structure(str(root)) == "split"


def test_detect_structures_cvat11_only(tmp_path: Path) -> None:
    root = tmp_path / "cvat"
    root.mkdir()
    _write_cvat11(root)
    assert detect_structures(str(root)) == ["cvat11"]
