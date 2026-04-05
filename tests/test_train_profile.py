"""Тесты merge профиля обучения YAML + CLI."""
from smartrain.train_profile import (
    extract_smartrain_options,
    merge_cli_into_ultralytics_cfg,
    task_to_metadata_task_type,
)


def test_extract_smartrain_options():
    u, opts = extract_smartrain_options(
        {
            "epochs": 10,
            "export_onnx": True,
            "weighted_sampling": True,
            "model": "yolov8n.pt",
        }
    )
    assert u == {"epochs": 10, "model": "yolov8n.pt"}
    assert opts["export_onnx"] is True
    assert opts["weighted_sampling"] is True


def test_merge_cli_yaml_precedence_partial_cli():
    u = {"epochs": 300, "patience": 50, "model": "yolo11x.pt"}
    merge_cli_into_ultralytics_cfg(
        u,
        model=None,
        epochs=None,
        batch=8,
        imgsz=None,
        task=None,
        defaults={"model": "yolov8n", "epochs": 50, "batch": 16, "imgsz": 640, "task": "detect"},
    )
    assert u["epochs"] == 300
    assert u["model"] == "yolo11x.pt"
    assert u["batch"] == 8
    assert u["imgsz"] == 640
    assert u["patience"] == 50


def test_merge_cli_overrides_all_defaults():
    u: dict = {}
    merge_cli_into_ultralytics_cfg(
        u,
        model="yolov8s.pt",
        epochs=5,
        batch=2,
        imgsz=320,
        task="segment",
        defaults={"model": "yolov8n", "epochs": 50, "batch": 16, "imgsz": 640, "task": "detect"},
    )
    assert u == {
        "model": "yolov8s.pt",
        "epochs": 5,
        "batch": 2,
        "imgsz": 320,
        "task": "segment",
    }


def test_task_to_metadata_task_type():
    assert task_to_metadata_task_type("detect") == "detection"
    assert task_to_metadata_task_type("segment") == "segmentation"
    assert task_to_metadata_task_type("classify") == "classification"
