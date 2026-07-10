from __future__ import annotations

import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from smartrain.services.datasets.dataset_convert_cli import main as dataset_convert_main
from smartrain.services.datasets.dataset_convert_service import (
    TARGET_CVAT11,
    TARGET_CVAT11_ZIP,
    TARGET_YOLO,
    run_conversion,
    DatasetSource,
    ConvertOptions,
)


def _write_jpg(path: Path, *, size: tuple[int, int] = (200, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(30, 40, 50)).save(path, format="JPEG", quality=90)


def _write_cvsdcldet_sample(source: Path) -> None:
    _write_jpg(source / "img_a.jpg", size=(200, 100))
    (source / "img_a.json").write_text(
        json.dumps(
            {
                "detections": [
                    {"classId": 0, "class_name": "cat", "x": 10, "y": 20, "width": 50, "height": 30, "score": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )


def _make_cvat11_zip(tmp_path: Path) -> Path:
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


def _make_yolo_flat(tmp_path: Path) -> Path:
    root = tmp_path / "yolo_flat"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    _write_jpg(root / "images" / "img001.jpg", size=(100, 80))
    (root / "labels" / "img001.txt").write_text("0 0.35 0.375 0.5 0.5\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "train: images\nval: images\ntest: images\nnc: 1\nnames: ['cat']\n",
        encoding="utf-8",
    )
    return root


def test_cvat_zip_to_yolo_service(tmp_path: Path) -> None:
    cvat_zip = _make_cvat11_zip(tmp_path)
    out = tmp_path / "yolo_out"
    source = DatasetSource(path=cvat_zip, structure="cvat11_zip", name=cvat_zip.stem, source_zip=cvat_zip)
    result = run_conversion(source, TARGET_YOLO, out, opts=ConvertOptions())
    assert result.output_dir is not None
    assert (result.output_dir / "data.yaml").is_file()
    assert (result.output_dir / "labels" / "img001.txt").is_file()


def test_yolo_to_cvat11_zip_service(tmp_path: Path) -> None:
    yolo = _make_yolo_flat(tmp_path)
    zip_out = tmp_path / "out.cvat11.zip"
    source = DatasetSource(path=yolo, structure="flat", name=yolo.name)
    result = run_conversion(source, TARGET_CVAT11_ZIP, zip_out, opts=ConvertOptions())
    assert result.zip_path is not None
    assert result.zip_path.is_file()
    with zipfile.ZipFile(result.zip_path, "r") as zf:
        assert any(n.endswith("annotations.xml") for n in zf.namelist())


def test_cvsdcldet_to_cvat11_cli(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "src"
    source.mkdir()
    _write_cvsdcldet_sample(source)
    out = tmp_path / "cvat_out"
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")
    dataset_convert_main(
        [
            "--source-dir",
            str(source),
            "--to",
            TARGET_CVAT11,
            "--output-dir",
            str(out),
            "--rename-classes",
            "cat",
            "feline",
        ]
    )
    tree = ET.parse(out / "annotations.xml")
    labels = {b.get("label") for b in tree.findall(".//box")}
    assert labels == {"feline"}


def test_cvsdcldet_to_cvat11_zip_with_delete_folder(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    _write_cvsdcldet_sample(source)
    zip_out = tmp_path / "out.cvat11.zip"
    folder_out = tmp_path / "out.cvat11"
    source_ds = DatasetSource(path=source, structure="cvsdcldet", name=source.name)
    result = run_conversion(
        source_ds,
        TARGET_CVAT11_ZIP,
        zip_out,
        opts=ConvertOptions(delete_after_zip=True),
    )
    assert result.zip_path is not None
    assert result.zip_path.is_file()
    assert not folder_out.is_dir()


def test_cvat_zip_import_cli(tmp_path: Path, monkeypatch) -> None:
    cvat_zip = _make_cvat11_zip(tmp_path)
    out = tmp_path / "yolo_cli"
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")
    dataset_convert_main(
        [
            "--source-zip",
            str(cvat_zip),
            "--to",
            TARGET_YOLO,
            "--output-dir",
            str(out),
        ]
    )
    assert (out / "data.yaml").is_file()


def test_yolo_export_cli(tmp_path: Path, monkeypatch) -> None:
    yolo = _make_yolo_flat(tmp_path)
    zip_out = tmp_path / "exported.cvat11.zip"
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")
    dataset_convert_main(
        [
            "--source-dir",
            str(yolo),
            "--to",
            TARGET_CVAT11_ZIP,
            "--output-dir",
            str(zip_out),
        ]
    )
    assert zip_out.is_file()
