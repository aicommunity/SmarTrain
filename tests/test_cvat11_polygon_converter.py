from __future__ import annotations

from pathlib import Path

from smartrain.services.datasets.cvat11_converter import (
    _cvat_polygon_to_yolo_line,
    build_cvat11_annotations_xml,
    export_yolo_to_cvat11_zip,
    CvatPolygon,
)
from smartrain.services.datasets.yolo_labels import YoloSegment, read_yolo_labels


def test_cvat_polygon_to_yolo_line() -> None:
    poly = CvatPolygon(label="obj", points=((10.0, 20.0), (90.0, 20.0), (90.0, 80.0), (10.0, 80.0)))
    line = _cvat_polygon_to_yolo_line(poly, class_id=0, img_w=100, img_h=100)
    assert line is not None
    labels = read_yolo_labels_from_line(line)
    assert len(labels) == 1
    assert isinstance(labels[0], YoloSegment)


def read_yolo_labels_from_line(line: str) -> list:
    import tempfile

    from smartrain.services.datasets.yolo_labels import read_yolo_labels

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "line.txt"
        p.write_text(line + "\n", encoding="utf-8")
        return read_yolo_labels(str(p))


def test_build_cvat_xml_with_polygon() -> None:
    xml = build_cvat11_annotations_xml(
        task_name="t",
        images=[("a.jpg", 100, 100, [], [("obj", [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0)])])],
        labels=["obj"],
    )
    assert "<polygon" in xml


def test_export_import_polygon_roundtrip(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (64, 48), color=(1, 2, 3)).save(ds / "images" / "a.jpg")
    (ds / "labels" / "a.txt").write_text("0 0.10 0.10 0.90 0.10 0.90 0.90 0.10 0.90\n", encoding="utf-8")
    zip_path = tmp_path / "out.zip"
    export_yolo_to_cvat11_zip(
        dataset_dir=ds,
        task_name="seg",
        output_zip_path=zip_path,
        names=["obj"],
        force=True,
    )
    assert zip_path.is_file()
