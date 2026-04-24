from __future__ import annotations

from smartrain.external_providers.adapters import build_external_infer_spec, build_external_train_spec


def test_mfel_train_adapter_uses_launcher() -> None:
    spec = build_external_train_spec(
        "mfel-yolo",
        "/tmp/mfel",
        dataset_path="/tmp/ds",
        model="mfel.pt",
        epochs=10,
        batch=4,
        imgsz=640,
        device="0",
        target_dir="/tmp/runs",
        run_name="2026-04-23_12-00_yolov8n_1epochs_b2-abc123",
    )
    assert "mfel_train_launcher.py" in spec.script_path
    assert "--repo" in spec.args
    assert "--project" in spec.args
    project_idx = spec.args.index("--project") + 1
    assert spec.args[project_idx].endswith("/runs/ds")
    assert "--name" in spec.args


def test_dr_infer_adapter_uses_launcher() -> None:
    spec = build_external_infer_spec(
        "dr-yolo",
        "/tmp/dr",
        model_path="/tmp/best.pt",
        source_path="/tmp/images",
        conf=0.25,
        imgsz=640,
        device="cpu",
    )
    assert "mp_infer_launcher.py" in spec.script_path
    assert "--source" in spec.args


def test_ssdm_train_adapter_uses_launcher_args() -> None:
    spec = build_external_train_spec(
        "ssdm-yolo",
        "/tmp/ssdm",
        dataset_path="/tmp/ds",
        model="yolov8n.pt",
        epochs=20,
        batch=8,
        imgsz=640,
        device="0",
        target_dir="/tmp/runs",
    )
    assert "mp_train_launcher.py" in spec.script_path
    assert "--data" in spec.args
    assert "--imgsz" in spec.args
    assert "--batch" in spec.args
    assert "--project" in spec.args


def test_ssdm_infer_adapter_uses_launcher_conf() -> None:
    spec = build_external_infer_spec(
        "ssdm-yolo",
        "/tmp/ssdm",
        model_path="/tmp/best.pt",
        source_path="/tmp/images",
        conf=0.3,
        imgsz=640,
        device="cpu",
    )
    assert "mp_infer_launcher.py" in spec.script_path
    assert "--conf" in spec.args
    assert "--imgsz" in spec.args


def test_enhanced_train_adapter_uses_launcher_args() -> None:
    spec = build_external_train_spec(
        "enhanced-yolov8",
        "/tmp/enhanced",
        dataset_path="/tmp/ds",
        model="best.pt",
        epochs=20,
        batch=8,
        imgsz=640,
        device="0",
        target_dir="/tmp/runs",
    )
    assert "mp_train_launcher.py" in spec.script_path
    assert "--model" in spec.args
    assert "--data" in spec.args


def test_enhanced_infer_adapter_uses_launcher_conf() -> None:
    spec = build_external_infer_spec(
        "enhanced-yolov8",
        "/tmp/enhanced",
        model_path="/tmp/best.pt",
        source_path="/tmp/images",
        conf=0.3,
        imgsz=640,
        device="cpu",
    )
    assert "mp_infer_launcher.py" in spec.script_path
    assert "--model" in spec.args
    assert "--conf" in spec.args

