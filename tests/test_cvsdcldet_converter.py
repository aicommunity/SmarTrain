from __future__ import annotations

import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from smartrain.services.datasets.cvsdcldet_converter import (
    collect_cvsdcldet_class_names,
    convert_cvsdcldet_to_cvat11,
    is_cvsdcldet_dir,
    parse_rename_classes_args,
)
from smartrain.services.datasets.dataset_convert_cli import main as dataset_convert_main


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
                    {"classId": 1, "class_name": "dog", "x": 100, "y": 10, "width": 40, "height": 40, "score": 0.8},
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_jpg(source / "img_b.jpg", size=(120, 80))
    (source / "img_b.json").write_text(
        json.dumps(
            {
                "detections": [
                    {"classId": 1, "class_name": "dog", "x": 5, "y": 5, "width": 20, "height": 15, "score": 0.7},
                ]
            }
        ),
        encoding="utf-8",
    )


def test_is_cvsdcldet_dir(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    assert not is_cvsdcldet_dir(source)
    _write_cvsdcldet_sample(source)
    assert is_cvsdcldet_dir(source)


def test_collect_class_names(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    _write_cvsdcldet_sample(source)
    assert collect_cvsdcldet_class_names(source) == ["cat", "dog"]


def test_convert_to_cvat11_extracted(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    _write_cvsdcldet_sample(source)
    out = tmp_path / "out"
    info = convert_cvsdcldet_to_cvat11(source_dir=source, output_dir=out, force=False, create_zip=False)
    assert info["images_count"] == 2
    assert info["boxes_count"] == 3
    assert set(info["classes"]) == {"cat", "dog"}
    assert (out / "annotations.xml").is_file()
    assert (out / "images" / "img_a.jpg").is_file()
    assert (out / "images" / "img_b.jpg").is_file()

    tree = ET.parse(out / "annotations.xml")
    root = tree.getroot()
    assert root.findtext("./version") == "1.1"
    boxes = root.findall(".//box")
    labels = {b.get("label") for b in boxes}
    assert labels == {"cat", "dog"}
    cat_box = next(b for b in boxes if b.get("label") == "cat")
    assert cat_box.get("xtl") == "10"
    assert cat_box.get("ytl") == "20"
    assert cat_box.get("xbr") == "60"
    assert cat_box.get("ybr") == "50"


def test_convert_with_rename_and_zip(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    _write_cvsdcldet_sample(source)
    out = tmp_path / "out"
    info = convert_cvsdcldet_to_cvat11(
        source_dir=source,
        output_dir=out,
        class_rename={"dog": "puppy"},
        force=False,
        create_zip=True,
    )
    tree = ET.parse(out / "annotations.xml")
    labels = {b.get("label") for b in tree.findall(".//box")}
    assert labels == {"cat", "puppy"}
    zip_path = Path(info["zip_path"])
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert any(n.endswith("annotations.xml") for n in names)
        assert any("/images/img_a.jpg" in n for n in names)


def test_parse_rename_classes_args() -> None:
    assert parse_rename_classes_args([["a", "b"], ["c", "d"]]) == {"a": "b", "c": "d"}


def test_dataset_convert_cli_from_cvsdcldet_non_interactive(tmp_path: Path, monkeypatch) -> None:
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
            "cvat11",
            "--output-dir",
            str(out),
            "--rename-classes",
            "cat",
            "feline",
            "--zip",
            "--no-delete-after-zip",
        ]
    )
    assert (out / "annotations.xml").is_file()
    assert Path(str(out) + ".cvat11.zip").is_file()
    tree = ET.parse(out / "annotations.xml")
    labels = {b.get("label") for b in tree.findall(".//box")}
    assert "feline" in labels
    assert "cat" not in labels
