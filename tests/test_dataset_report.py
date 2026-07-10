from __future__ import annotations

from pathlib import Path

from PIL import Image

from smartrain.services.datasets.dataset_report import _build_report_instance_image, _labels_to_instances
from smartrain.services.datasets.yolo_labels import YoloSegment, write_yolo_labels


def test_labels_to_instances_polygon(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    lbl_path = tmp_path / "img.txt"
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(img_path, format="JPEG")
    seg = YoloSegment(cls_id=0, points=((0.10, 0.10), (0.90, 0.10), (0.90, 0.90), (0.10, 0.90)))
    write_yolo_labels(str(lbl_path), [seg])
    from smartrain.services.datasets.yolo_labels import read_yolo_labels

    labels = read_yolo_labels(str(lbl_path))
    instances = _labels_to_instances(str(img_path), str(lbl_path), labels)
    assert len(instances) == 1
    assert instances[0].kind == "segment"
    assert instances[0].segment_points is not None
    assert len(instances[0].segment_points) == 4


def test_build_report_instance_image_polygon(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    lbl_path = tmp_path / "img.txt"
    Image.new("RGB", (64, 48), color=(50, 60, 70)).save(img_path, format="JPEG")
    seg = YoloSegment(cls_id=0, points=((0.20, 0.20), (0.80, 0.20), (0.80, 0.80), (0.20, 0.80)))
    write_yolo_labels(str(lbl_path), [seg])
    from smartrain.services.datasets.yolo_labels import read_yolo_labels

    labels = read_yolo_labels(str(lbl_path))
    instances = _labels_to_instances(str(img_path), str(lbl_path), labels)
    out = _build_report_instance_image(
        instances[0],
        class_name="obj",
        padding_frac=0.1,
        canvas_w=128,
        canvas_h=128,
        max_letterbox_scale=4.0,
    )
    assert out is not None
    assert out.size == (128, 128)
