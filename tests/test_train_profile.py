"""YAML + CLI training profile merge tests."""
import argparse

from smartrain import model_training_module as mtm
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
        device=None,
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
        device="0",
        defaults={"model": "yolov8n", "epochs": 50, "batch": 16, "imgsz": 640, "task": "detect"},
    )
    assert u == {
        "model": "yolov8s.pt",
        "epochs": 5,
        "batch": 2,
        "imgsz": 320,
        "task": "segment",
        "device": "0",
    }


def test_task_to_metadata_task_type():
    assert task_to_metadata_task_type("detect") == "detection"
    assert task_to_metadata_task_type("segment") == "segmentation"
    assert task_to_metadata_task_type("classify") == "classification"


def test_merge_sources_priority_cli_over_ultralytics_yaml_over_config():
    args = argparse.Namespace(model=None, epochs=10, batch=None, img_size=None, task=None)
    u_cfg, _sm_opts = mtm._merge_sources_with_priority(
        config_profile={"model": "cfg.pt", "epochs": 100, "batch": 32},
        ultralytics_profile={"model": "ultra.pt", "epochs": 20, "batch": 8, "imgsz": 960},
        args=args,
    )
    merge_cli_into_ultralytics_cfg(
        u_cfg,
        model=getattr(args, "model", None),
        epochs=getattr(args, "epochs", None),
        batch=getattr(args, "batch", None),
        imgsz=getattr(args, "img_size", None),
        task=getattr(args, "task", None),
        device=getattr(args, "device", None),
        defaults={"model": "yolov8n", "epochs": 50, "batch": 16, "imgsz": 640, "task": "detect"},
    )
    assert u_cfg["model"] == "ultra.pt"
    assert u_cfg["epochs"] == 10
    assert u_cfg["batch"] == 8
    assert u_cfg["imgsz"] == 960


def test_merge_sources_ignores_forced_keys_from_ultralytics_yaml(capsys):
    args = argparse.Namespace(model=None, epochs=None, batch=None, img_size=None, task=None)
    u_cfg, _sm_opts = mtm._merge_sources_with_priority(
        config_profile={},
        ultralytics_profile={"data": "/tmp/data.yaml", "project": "/tmp/runs", "name": "x", "exist_ok": True, "epochs": 25},
        args=args,
    )
    out = capsys.readouterr().out
    assert "keys ignored" in out.lower()
    assert "data" in out and "project" in out
    assert "epochs" in u_cfg
    assert "data" not in u_cfg


def test_merge_sources_ignores_cfg_key_from_ultralytics_yaml():
    args = argparse.Namespace(model=None, epochs=None, batch=None, img_size=None, task=None)
    u_cfg, _sm_opts = mtm._merge_sources_with_priority(
        config_profile={},
        ultralytics_profile={"cfg": "/missing/args_mars.yaml", "device": "0,1,2", "epochs": 25},
        args=args,
    )
    assert "cfg" not in u_cfg
    assert "device" not in u_cfg
    assert u_cfg["epochs"] == 25


def test_finalize_train_kwargs_forces_dataset_and_run_paths():
    out = mtm._finalize_train_kwargs(
        {"data": "/tmp/from_yaml.yaml", "project": "/tmp/proj", "name": "custom", "exist_ok": True, "epochs": 2},
        data_yaml="/dataset/data.yaml",
        model_dir="/runs/out",
    )
    assert out["data"] == "/dataset/data.yaml"
    assert out["project"] == "/runs/out"
    assert out["name"] == "train"
    assert out["exist_ok"] is False
    assert out["epochs"] == 2


def test_merge_cli_sets_default_device_when_missing():
    u: dict = {}
    merge_cli_into_ultralytics_cfg(
        u,
        model=None,
        epochs=None,
        batch=None,
        imgsz=None,
        task=None,
        device=None,
        defaults={"model": "yolov8n", "epochs": 50, "batch": 16, "imgsz": 640, "task": "detect", "device": "0"},
    )
    assert u["device"] == "0"
