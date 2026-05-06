from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image

from smartrain.workflows.datasets.cvat11_converter import import_cvat11_zip_to_yolo, export_yolo_to_cvat11_zip


def _write_jpg(path: Path, *, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color=(10, 20, 30))
    im.save(path, format="JPEG", quality=90)


def _make_cvat11_zip(tmp_path: Path) -> Path:
    # CVAT expected zip structure: <task>/annotations.xml + <task>/images/<name>.jpg
    task = tmp_path / "task1"
    images = task / "images"
    images.mkdir(parents=True, exist_ok=True)
    _write_jpg(images / "img001.jpg", size=(100, 80))

    annotations = task / "annotations.xml"
    annotations.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <name>task1</name>
      <labels>
        <label><name>cat</name><type>bbox</type><attributes></attributes></label>
        <label><name>dog</name><type>bbox</type><attributes></attributes></label>
      </labels>
    </task>
  </meta>
  <image id="0" name="img001.jpg" width="100" height="80">
    <box label="cat" xtl="10" ytl="10" xbr="60" ybr="50" occluded="0" z_order="0"></box>
  </image>
</annotations>
""",
        encoding="utf-8",
    )

    zip_path = tmp_path / "cvat11.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(annotations, arcname="task1/annotations.xml")
        zf.write(images / "img001.jpg", arcname="task1/images/img001.jpg")
    return zip_path


def test_import_then_export_roundtrip(tmp_path: Path) -> None:
    cvat_zip = _make_cvat11_zip(tmp_path)
    out_dir = tmp_path / "yolo_out"

    info = import_cvat11_zip_to_yolo(
        cvat_zip_path=cvat_zip,
        output_dir=out_dir,
        task_name=None,
        force=False,
    )
    assert info["nc"] == 2
    assert (out_dir / "images" / "img001.jpg").exists()
    assert (out_dir / "labels" / "img001.txt").exists()
    assert (out_dir / "data.yaml").exists()

    zip_out = tmp_path / "out.zip"
    info2 = export_yolo_to_cvat11_zip(
        dataset_dir=out_dir,
        task_name="exported",
        output_zip_path=zip_out,
        names=info["names"],
        force=False,
    )
    assert Path(info2["zip_path"]).exists()

    with zipfile.ZipFile(zip_out, "r") as zf:
        names = set(zf.namelist())
        assert "exported/annotations.xml" in names
        assert "exported/images/img001.jpg" in names

