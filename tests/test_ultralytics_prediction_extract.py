"""Unit tests for smartrain.core.inference.ultralytics_prediction_extract."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from smartrain.core.inference.ultralytics_prediction_extract import extract_task_outputs_from_ultralytics_preds


def test_empty_preds_detection() -> None:
    out = extract_task_outputs_from_ultralytics_preds(None, [], task_type="detection")
    assert out == {"detections": []}


def test_empty_preds_classification() -> None:
    out = extract_task_outputs_from_ultralytics_preds(None, [], task_type="classification")
    assert out == {"classification": {}}


def test_empty_preds_segmentation() -> None:
    out = extract_task_outputs_from_ultralytics_preds(None, [], task_type="segmentation")
    assert out == {"segments": []}


def test_detection_one_box_no_model_names() -> None:
    """Launchers pass model=None; class_name falls back to string index."""
    boxes = MagicMock()
    boxes.__len__.return_value = 1
    boxes.xyxy = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10.0, 20.0, 30.0, 40.0]])
    boxes.cls = MagicMock()
    boxes.cls.cpu.return_value.numpy.return_value = np.array([3])
    boxes.conf = MagicMock()
    boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])
    r0 = MagicMock()
    r0.boxes = boxes
    preds = [r0]

    out = extract_task_outputs_from_ultralytics_preds(None, preds, task_type="detection")
    det = out["detections"]
    assert len(det) == 1
    assert det[0]["class_index"] == 3
    assert det[0]["class_name"] == "3"
    assert det[0]["confidence"] == pytest.approx(0.9)
    assert det[0]["bbox_roi_xyxy"] == [10.0, 20.0, 30.0, 40.0]


def test_detection_class_names_override() -> None:
    boxes = MagicMock()
    boxes.__len__.return_value = 1
    boxes.xyxy = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[0.0, 0.0, 1.0, 1.0]])
    boxes.cls = MagicMock()
    boxes.cls.cpu.return_value.numpy.return_value = np.array([1])
    boxes.conf = MagicMock()
    boxes.conf.cpu.return_value.numpy.return_value = np.array([0.5])
    r0 = MagicMock()
    r0.boxes = boxes
    out = extract_task_outputs_from_ultralytics_preds(
        None,
        [r0],
        task_type="detection",
        class_names={1: "car"},
    )
    assert out["detections"][0]["class_name"] == "car"


def test_classification_top1() -> None:
    probs = MagicMock()
    probs.top1 = 2
    probs.top1conf = 0.88
    probs.top5 = [2, 0]
    probs.top5conf = [0.88, 0.05]
    r0 = MagicMock()
    r0.probs = probs
    model = MagicMock()
    model.names = {0: "a", 2: "c"}
    out = extract_task_outputs_from_ultralytics_preds(model, [r0], task_type="classification")
    cls_out = out["classification"]
    assert cls_out["top1"]["class_index"] == 2
    assert cls_out["top1"]["class_name"] == "c"
    assert cls_out["top1"]["confidence"] == pytest.approx(0.88)


def test_segmentation_one_mask() -> None:
    boxes = MagicMock()
    boxes.__len__.return_value = 1
    boxes.xyxy = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[1.0, 2.0, 3.0, 4.0]])
    boxes.cls = MagicMock()
    boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
    boxes.conf = MagicMock()
    boxes.conf.cpu.return_value.numpy.return_value = np.array([0.99])
    masks = MagicMock()
    masks.xy = [[[0.0, 0.0], [1.0, 0.0]]]
    r0 = MagicMock()
    r0.boxes = boxes
    r0.masks = masks
    out = extract_task_outputs_from_ultralytics_preds(None, [r0], task_type="segmentation")
    segs = out["segments"]
    assert len(segs) == 1
    assert segs[0]["class_index"] == 0
    assert segs[0]["polygon_roi_xy"] == [[0.0, 0.0], [1.0, 0.0]]
