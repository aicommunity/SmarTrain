from __future__ import annotations

from pathlib import Path

import pytest

from smartrain.services.datasets.yolo_labels import (
    YoloBBox,
    YoloSegment,
    read_yolo_labels,
    rotate_yolo_labels_90cw_k,
    serialize_yolo_labels,
    write_yolo_labels,
)


def test_read_yolo_labels_bbox(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("0 0.50 0.50 0.40 0.30\n", encoding="utf-8")
    labels = read_yolo_labels(str(p))
    assert len(labels) == 1
    assert isinstance(labels[0], YoloBBox)
    assert labels[0].cls_id == 0
    assert labels[0].cx == pytest.approx(0.5)


def test_read_yolo_labels_polygon(tmp_path: Path) -> None:
    p = tmp_path / "seg.txt"
    p.write_text("1 0.10 0.10 0.90 0.10 0.90 0.90 0.10 0.90\n", encoding="utf-8")
    labels = read_yolo_labels(str(p))
    assert len(labels) == 1
    assert isinstance(labels[0], YoloSegment)
    assert labels[0].cls_id == 1
    assert len(labels[0].points) == 4
    assert labels[0].points[0] == pytest.approx((0.10, 0.10))


def test_read_yolo_labels_mixed(tmp_path: Path) -> None:
    p = tmp_path / "mix.txt"
    p.write_text(
        "0 0.50 0.50 0.20 0.20\n"
        "1 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n"
        "# comment\n"
        "bad line\n",
        encoding="utf-8",
    )
    labels = read_yolo_labels(str(p))
    assert len(labels) == 2
    assert isinstance(labels[0], YoloBBox)
    assert isinstance(labels[1], YoloSegment)


def test_read_yolo_labels_invalid_polygon_odd_coords(tmp_path: Path) -> None:
    p = tmp_path / "bad.txt"
    p.write_text("0 0.1 0.2 0.3 0.4 0.5\n", encoding="utf-8")
    assert read_yolo_labels(str(p)) == []


def test_read_yolo_labels_missing_file() -> None:
    assert read_yolo_labels("/nonexistent/path/labels.txt") == []


def test_write_read_roundtrip_bbox(tmp_path: Path) -> None:
    p = tmp_path / "out.txt"
    src = [YoloBBox(cls_id=2, cx=0.25, cy=0.75, w=0.10, h=0.20)]
    write_yolo_labels(str(p), src)
    back = read_yolo_labels(str(p))
    assert len(back) == 1
    lb = back[0]
    assert isinstance(lb, YoloBBox)
    assert lb.cls_id == 2
    assert lb.cx == pytest.approx(0.25, abs=1e-5)


def test_write_read_roundtrip_polygon(tmp_path: Path) -> None:
    p = tmp_path / "out.txt"
    pts = ((0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4))
    src = [YoloSegment(cls_id=0, points=pts)]
    write_yolo_labels(str(p), src)
    back = read_yolo_labels(str(p))
    assert len(back) == 1
    lb = back[0]
    assert isinstance(lb, YoloSegment)
    assert len(lb.points) == 4
    for a, b in zip(lb.points, pts, strict=True):
        assert a[0] == pytest.approx(b[0], abs=1e-4)
        assert a[1] == pytest.approx(b[1], abs=1e-4)


def test_serialize_yolo_labels_empty() -> None:
    assert serialize_yolo_labels([]) == ""


def test_rotate_polygon_90cw(tmp_path: Path) -> None:
    seg = YoloSegment(cls_id=0, points=((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)))
    rotated, nw, nh = rotate_yolo_labels_90cw_k([seg], w=100, h=80, k=1)
    assert (nw, nh) == (80, 100)
    assert len(rotated) == 1
    assert isinstance(rotated[0], YoloSegment)
    for x, y in rotated[0].points:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_rotate_polygon_180(tmp_path: Path) -> None:
    seg = YoloSegment(cls_id=0, points=((0.20, 0.30), (0.80, 0.30), (0.80, 0.70), (0.20, 0.70)))
    rotated, nw, nh = rotate_yolo_labels_90cw_k([seg], w=200, h=100, k=2)
    assert (nw, nh) == (200, 100)
    assert len(rotated) == 1


def test_rotate_polygon_270(tmp_path: Path) -> None:
    seg = YoloSegment(cls_id=0, points=((0.10, 0.10), (0.50, 0.10), (0.50, 0.50), (0.10, 0.50)))
    rotated, nw, nh = rotate_yolo_labels_90cw_k([seg], w=64, h=48, k=3)
    assert (nw, nh) == (48, 64)
    assert len(rotated) == 1


def test_task_output_dict_to_yolo_label_bbox() -> None:
    from smartrain.services.datasets.yolo_labels import task_output_dict_to_yolo_label

    det = {
        "bbox_original_xyxy": [10.0, 20.0, 30.0, 40.0],
        "class_index": 1,
        "confidence": 0.9,
    }
    lb = task_output_dict_to_yolo_label(det, 100, 100)
    assert lb is not None
    assert lb.cls_id == 1
    assert lb.cx == pytest.approx(0.2)
    assert lb.cy == pytest.approx(0.3)


def test_rotate_polygon_k0_identity() -> None:
    seg = YoloSegment(cls_id=0, points=((0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)))
    rotated, nw, nh = rotate_yolo_labels_90cw_k([seg], w=100, h=80, k=0)
    assert (nw, nh) == (100, 80)
    assert rotated[0].points == seg.points
