from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

from PIL import Image

from smartrain.model_test_backends import run_native_format_backend


class _FakeInput:
    name = "images"
    shape = [1, 3, 640, 640]


class _FakeSession:
    def __init__(self, _path: str, providers=None):
        self.providers = providers or []

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, _output_names, _feeds):
        return [
            [
                [
                    [128.0, 128.0, 128.0, 128.0, 0.95],
                ]
            ]
        ]


def _install_fake_onnxruntime(monkeypatch) -> None:
    fake_mod = ModuleType("onnxruntime")
    fake_mod.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    fake_mod.InferenceSession = _FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_mod)


def _make_dataset(tmp_path: Path, name: str, *, with_test: bool = True) -> Path:
    ds = tmp_path / "datasets" / name
    split_name = "test" if with_test else "val"
    (ds / split_name / "images").mkdir(parents=True, exist_ok=True)
    (ds / split_name / "labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(ds / split_name / "images" / "a.jpg")
    (ds / split_name / "labels" / "a.txt").write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")
    yaml_lines = ["train: train/images", "val: val/images"]
    if with_test:
        yaml_lines.append("test: test/images")
    else:
        yaml_lines[1] = "val: val/images"
    yaml_lines.append("names: ['obj']")
    (ds / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return ds


def test_run_native_onnx_backend_writes_test_artifacts(monkeypatch, tmp_path: Path) -> None:
    _install_fake_onnxruntime(monkeypatch)
    root_dir = tmp_path / "run_x"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.onnx"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_a", with_test=True)

    result = run_native_format_backend(
        root_dir=str(root_dir),
        weights_path=str(weights_path),
        dataset_yaml_path=str(ds / "data.yaml"),
        format_name="onnx",
        imgsz=640,
        val_conf=0.25,
        val_iou=0.5,
        val_batch=1,
    )

    assert result.success is True
    assert (root_dir / "test_metrics_onnx.csv").is_file()
    assert (root_dir / "test_onnx" / "pr.csv").is_file()
    assert (root_dir / "test_onnx" / "pr_per_class.csv").is_file()
    assert (root_dir / "test_onnx" / "BoxPR_curve.png").is_file()
    assert (root_dir / "test_onnx" / "BoxF1_curve.png").is_file()
    assert (root_dir / "test_onnx" / "BoxP_curve.png").is_file()
    assert (root_dir / "test_onnx" / "BoxR_curve.png").is_file()
    assert (root_dir / "test_onnx" / "confusion_matrix.png").is_file()
    assert (root_dir / "test_onnx" / "confusion_matrix_normalized.png").is_file()
    assert (root_dir / "confidence_recommendations_test_onnx.json").is_file()
    assert (root_dir / "confidence_recommendations_val_onnx.json").is_file()
    manifest = json.loads((root_dir / "test_artifacts_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formats"]["onnx"]["status"] == "ok"


def test_run_native_onnx_backend_prints_progress(monkeypatch, tmp_path: Path, capsys) -> None:
    _install_fake_onnxruntime(monkeypatch)
    root_dir = tmp_path / "run_progress"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.onnx"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_progress", with_test=True)

    result = run_native_format_backend(
        root_dir=str(root_dir),
        weights_path=str(weights_path),
        dataset_yaml_path=str(ds / "data.yaml"),
        format_name="onnx",
        imgsz=640,
        val_conf=0.25,
        val_iou=0.5,
        val_batch=1,
    )

    out = capsys.readouterr().out
    assert result.success is True
    assert "[INFO] onnx: running native test on 1 images with" in out
    assert "onnx:test" in out
    assert "100%" in out
    assert "img/s" in out or "img]" in out or "1/1" in out


def test_run_native_onnx_backend_without_test_split_marks_unavailable(monkeypatch, tmp_path: Path) -> None:
    _install_fake_onnxruntime(monkeypatch)
    root_dir = tmp_path / "run_missing_test"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.onnx"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_no_test", with_test=False)

    result = run_native_format_backend(
        root_dir=str(root_dir),
        weights_path=str(weights_path),
        dataset_yaml_path=str(ds / "data.yaml"),
        format_name="onnx",
        imgsz=640,
        val_conf=0.25,
        val_iou=0.5,
        val_batch=1,
    )

    assert result.success is False
    manifest = json.loads((root_dir / "test_artifacts_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formats"]["onnx"]["status"] == "unavailable"
    assert "data.yaml has no split='test'" in str(manifest["formats"]["onnx"]["error"])


def test_run_native_tensorrt_backend_writes_test_artifacts(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "run_trt"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.engine"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_trt", with_test=True)

    def _fake_trt_infer(_engine_path, image_path, _input_hw, _conf_thr, _iou_thr, _names):
        from smartrain.model_test_backends import _Pred

        return [_Pred(image_path=image_path, cls_id=0, conf=0.95, x1=10.0, y1=10.0, x2=30.0, y2=30.0)]

    monkeypatch.setattr("smartrain.model_test_backends._infer_with_trt_engine", _fake_trt_infer)

    result = run_native_format_backend(
        root_dir=str(root_dir),
        weights_path=str(weights_path),
        dataset_yaml_path=str(ds / "data.yaml"),
        format_name="engine",
        imgsz=640,
        val_conf=0.25,
        val_iou=0.5,
        val_batch=1,
    )

    assert result.success is True
    assert (root_dir / "test_metrics_engine.csv").is_file()
    assert (root_dir / "test_engine" / "pr.csv").is_file()
    assert (root_dir / "confidence_recommendations_test_engine.json").is_file()
    manifest = json.loads((root_dir / "test_artifacts_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formats"]["engine"]["status"] == "ok"
