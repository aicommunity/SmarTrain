from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

from PIL import Image

import smartrain.workflows.testing.model_test_backends as model_test_backends
from smartrain.workflows.testing.model_test_backends import run_native_format_backend
from smartrain.workflows.testing.model_test_service import (
    format_metrics_path,
    format_recommendation_path,
    format_test_dir,
    has_matching_test_artifacts,
    test_artifacts_manifest_path as manifest_path_fn,
)
from smartrain.workflows.testing.unified_metrics_adapter import collect_ultralytics_style_gt


def _stub_prepare_trt_runtime(_engine_path: str) -> dict:
    """Avoid deserializing real TensorRT in tests — fake engine bytes can SIGABRT the interpreter."""
    return {
        "trt": None,
        "cudart": None,
        "engine": object(),
        "context": object(),
        "init_ns": 42_000_000,
    }


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
    monkeypatch.setenv("SMARTTRAIN_ONNX_USE_SUBPROCESS", "0")


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
    assert Path(format_metrics_path(str(root_dir), "onnx")).is_file()
    onnx_test_dir = Path(format_test_dir(str(root_dir), "onnx"))
    assert (onnx_test_dir / "pr.csv").is_file()
    assert (onnx_test_dir / "pr_per_class.csv").is_file()
    assert (onnx_test_dir / "BoxPR_curve.png").is_file()
    assert (onnx_test_dir / "BoxF1_curve.png").is_file()
    assert (onnx_test_dir / "BoxP_curve.png").is_file()
    assert (onnx_test_dir / "BoxR_curve.png").is_file()
    assert (onnx_test_dir / "confusion_matrix.png").is_file()
    assert (onnx_test_dir / "confusion_matrix_normalized.png").is_file()
    assert Path(format_recommendation_path(str(root_dir), "test", "onnx")).is_file()
    assert Path(format_recommendation_path(str(root_dir), "val", "onnx")).is_file()
    manifest = json.loads(Path(manifest_path_fn(str(root_dir))).read_text(encoding="utf-8"))
    assert manifest["formats"]["onnx"]["status"] == "ok"
    artifacts = manifest["formats"]["onnx"].get("artifacts") or []
    assert isinstance(artifacts, list) and isinstance(artifacts[0].get("test_system_profile"), dict)
    runtime = artifacts[0]["test_system_profile"].get("runtime") or {}
    assert runtime.get("stage") == "test"
    assert runtime.get("format") == "onnx"


def test_run_native_onnx_backend_writes_to_new_tests_layout_even_with_legacy_dirs(monkeypatch, tmp_path: Path) -> None:
    _install_fake_onnxruntime(monkeypatch)
    root_dir = tmp_path / "run_x_legacy_layout"
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "test_onnx").mkdir(parents=True, exist_ok=True)
    (root_dir / "test_onnx" / "legacy.txt").write_text("legacy", encoding="utf-8")
    weights_path = root_dir / "model.onnx"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_a_new_layout", with_test=True)

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
    assert (root_dir / "tests" / "test_onnx" / "pr.csv").is_file()
    assert (root_dir / "tests" / "test_metrics_onnx.csv").is_file()
    assert (root_dir / "tests" / "test_artifacts_manifest.json").is_file()


def test_run_native_onnx_backend_collects_performance(monkeypatch, tmp_path: Path) -> None:
    _install_fake_onnxruntime(monkeypatch)
    root_dir = tmp_path / "run_perf_onnx"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.onnx"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_perf_onnx", with_test=True)

    result = run_native_format_backend(
        root_dir=str(root_dir),
        weights_path=str(weights_path),
        dataset_yaml_path=str(ds / "data.yaml"),
        format_name="onnx",
        imgsz=640,
        val_conf=0.25,
        val_iou=0.5,
        val_batch=1,
        collect_performance=True,
        perf_warmup_images=1,
    )
    assert result.success is True
    perf = result.inference.get("performance")
    assert isinstance(perf, dict)
    assert perf.get("warmup_images") == 1
    assert "latency_ms" in perf
    manifest = json.loads(Path(manifest_path_fn(str(root_dir))).read_text(encoding="utf-8"))
    artifacts = manifest["formats"]["onnx"].get("artifacts") or []
    assert isinstance(artifacts, list) and isinstance(artifacts[0].get("performance"), dict)

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
    manifest = json.loads(Path(manifest_path_fn(str(root_dir))).read_text(encoding="utf-8"))
    assert manifest["formats"]["onnx"]["status"] == "unavailable"
    assert "data.yaml has no split='test'" in str(manifest["formats"]["onnx"]["error"])


def test_run_native_onnx_backend_retries_after_cuda_oom(monkeypatch, tmp_path: Path, capsys) -> None:
    _install_fake_onnxruntime(monkeypatch)
    root_dir = tmp_path / "run_onnx_retry"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.onnx"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_onnx_retry", with_test=True)

    calls = {"n": 0}

    def _flaky_onnx_infer(*_args, **_kwargs):
        from smartrain.workflows.testing.model_test_backends import _Pred

        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory while running onnxruntime")
        return [_Pred(image_path=str(ds / "test" / "images" / "a.jpg"), cls_id=0, conf=0.9, x1=10.0, y1=10.0, x2=30.0, y2=30.0)]

    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners._infer_with_onnx_session",
        _flaky_onnx_infer,
    )

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
    assert "CUDA OOM during test on attempt 1/3" in out
    assert Path(format_metrics_path(str(root_dir), "onnx")).is_file()


def test_run_native_onnx_backend_subprocess_mode(monkeypatch, tmp_path: Path) -> None:
    import subprocess

    monkeypatch.setenv("SMARTTRAIN_ONNX_USE_SUBPROCESS", "1")
    root_dir = tmp_path / "run_onnx_subprocess"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.onnx"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_onnx_subprocess", with_test=True)
    test_image = str(ds / "test" / "images" / "a.jpg")

    def _fake_subprocess_run(*_args, **kwargs):
        req = json.loads(str(kwargs.get("input", "{}")))
        payload = {
            "ok": True,
            "preds": [
                {
                    "image_path": test_image,
                    "cls_id": 0,
                    "conf": 0.9,
                    "x1": 10.0,
                    "y1": 10.0,
                    "x2": 30.0,
                    "y2": 30.0,
                }
            ],
            "input_hw": [int(req.get("imgsz", 640)), int(req.get("imgsz", 640))],
            "provider": "CUDAExecutionProvider",
        }
        return subprocess.CompletedProcess(args=_args[0] if _args else [], returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("smartrain.services.testing.backends.format_runners.subprocess.run", _fake_subprocess_run)

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
    assert Path(format_metrics_path(str(root_dir), "onnx")).is_file()
    manifest = json.loads(Path(manifest_path_fn(str(root_dir))).read_text(encoding="utf-8"))
    runtime = ((manifest.get("formats") or {}).get("onnx") or {}).get("artifacts", [{}])[0].get("test_system_profile", {}).get("runtime", {})
    assert runtime.get("provider_actual") == "CUDAExecutionProvider"


def test_run_native_onnx_backend_subprocess_error_has_reason_code(monkeypatch, tmp_path: Path) -> None:
    import subprocess

    monkeypatch.setenv("SMARTTRAIN_ONNX_USE_SUBPROCESS", "1")
    root_dir = tmp_path / "run_onnx_subprocess_err"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.onnx"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_onnx_subprocess_err", with_test=True)

    def _fake_subprocess_run(*_args, **_kwargs):
        payload = {"ok": False, "error": "CUDA out of memory during worker run"}
        return subprocess.CompletedProcess(args=_args[0] if _args else [], returncode=1, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("smartrain.services.testing.backends.format_runners.subprocess.run", _fake_subprocess_run)

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
    assert isinstance(result.error, str)
    assert result.error.startswith("[oom_gpu]")


def test_run_native_onnx_gpu_strict_fails_when_cuda_provider_unavailable(monkeypatch, tmp_path: Path) -> None:
    fake_mod = ModuleType("onnxruntime")
    fake_mod.get_available_providers = lambda: ["CPUExecutionProvider"]
    fake_mod.InferenceSession = _FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_mod)
    monkeypatch.setenv("SMARTTRAIN_ONNX_USE_SUBPROCESS", "0")
    root_dir = tmp_path / "run_onnx_gpu_strict_no_cuda"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.onnx"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_onnx_gpu_strict_no_cuda", with_test=True)

    result = run_native_format_backend(
        root_dir=str(root_dir),
        weights_path=str(weights_path),
        dataset_yaml_path=str(ds / "data.yaml"),
        format_name="onnx",
        imgsz=640,
        onnx_provider_policy="gpu_strict",
    )
    assert result.success is False
    assert isinstance(result.error, str) and result.error.startswith("[provider_unavailable]")


def test_run_native_onnx_auto_aligns_shape_mismatch(monkeypatch, tmp_path: Path) -> None:
    _install_fake_onnxruntime(monkeypatch)
    root_dir = tmp_path / "run_onnx_auto_align"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.onnx"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_onnx_auto_align", with_test=True)

    result = run_native_format_backend(
        root_dir=str(root_dir),
        weights_path=str(weights_path),
        dataset_yaml_path=str(ds / "data.yaml"),
        format_name="onnx",
        imgsz=1280,
    )
    assert result.success is True
    args_yaml = Path(format_test_dir(str(root_dir), "onnx")) / "args.yaml"
    assert "imgsz: 640" in args_yaml.read_text(encoding="utf-8")


def test_run_native_tensorrt_backend_writes_test_artifacts(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "run_trt"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.engine"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_trt", with_test=True)

    def _fake_trt_infer(_engine_path, image_path, _input_hw, _conf_thr, _iou_thr, _names):
        from smartrain.workflows.testing.model_test_backends import _Pred

        return [_Pred(image_path=image_path, cls_id=0, conf=0.95, x1=10.0, y1=10.0, x2=30.0, y2=30.0)]

    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners._prepare_trt_runtime",
        _stub_prepare_trt_runtime,
    )
    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners._infer_with_trt_engine",
        _fake_trt_infer,
    )

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
    assert Path(format_metrics_path(str(root_dir), "engine")).is_file()
    engine_test_dir = Path(format_test_dir(str(root_dir), "engine"))
    assert (engine_test_dir / "pr.csv").is_file()
    assert Path(format_recommendation_path(str(root_dir), "test", "engine")).is_file()
    manifest = json.loads(Path(manifest_path_fn(str(root_dir))).read_text(encoding="utf-8"))
    assert manifest["formats"]["engine"]["status"] == "ok"
    artifacts = manifest["formats"]["engine"].get("artifacts") or []
    assert isinstance(artifacts, list) and isinstance(artifacts[0].get("test_system_profile"), dict)


def test_run_native_tensorrt_backend_marks_partial_ok_when_val_fails(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "run_trt_partial"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.engine"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_trt_partial", with_test=True)
    (ds / "val" / "images").mkdir(parents=True, exist_ok=True)
    (ds / "val" / "labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(ds / "val" / "images" / "a.jpg")
    (ds / "val" / "labels" / "a.txt").write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")

    def _fake_trt_infer(_engine_path, image_path, _input_hw, _conf_thr, _iou_thr, _names):
        from smartrain.workflows.testing.model_test_backends import _Pred

        return [_Pred(image_path=image_path, cls_id=0, conf=0.95, x1=10.0, y1=10.0, x2=30.0, y2=30.0)]

    orig_collect_gt = model_test_backends._collect_gt

    def _collect_gt_with_val_failure(data_yaml: str, split: str):
        if split == "val":
            raise RuntimeError("val split forced failure")
        return orig_collect_gt(data_yaml, split)

    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners._prepare_trt_runtime",
        _stub_prepare_trt_runtime,
    )
    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners._infer_with_trt_engine",
        _fake_trt_infer,
    )
    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners._collect_gt",
        _collect_gt_with_val_failure,
    )

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
    manifest = json.loads(Path(manifest_path_fn(str(root_dir))).read_text(encoding="utf-8"))
    fmt_payload = manifest["formats"]["engine"]
    assert fmt_payload["status"] == "partial_ok"
    assert "val split failed" in str(fmt_payload.get("error") or "")
    split_status = fmt_payload.get("split_status") or {}
    assert (split_status.get("val") or {}).get("status") == "failed"


def test_has_matching_test_artifacts_rejects_all_zero_native_metrics(tmp_path: Path) -> None:
    root_dir = tmp_path / "run_match_zero"
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "models").mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "models" / "model.engine"
    weights_path.write_bytes(b"fake")
    data_yaml = tmp_path / "datasets" / "ds_match_zero" / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text("train: train/images\nval: val/images\ntest: test/images\nnames: ['obj']\n", encoding="utf-8")
    (root_dir / "tests" / "test_engine").mkdir(parents=True, exist_ok=True)
    (root_dir / "tests" / "test_engine" / "args.yaml").write_text(
        "\n".join(
            [
                "backend: tensorrt",
                "format: engine",
                f"weights: {weights_path}",
                f"data: {data_yaml}",
                "imgsz: 640",
                "conf: 0.25",
                "iou: 0.5",
                "batch: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for split_name in ("test", "val"):
        metrics_name = "test_metrics_engine.csv" if split_name == "test" else "val_metrics_engine.csv"
        (root_dir / "tests" / metrics_name).write_text(
            "mAP50-95,mAP50,Box-F1,Box-P,Box-R\n0.0,0.0,0.0,0.0,0.0\n",
            encoding="utf-8",
        )
    (root_dir / "tests" / "test_artifacts_manifest.json").write_text(
        json.dumps(
            {
                "formats": {
                    "engine": {
                        "format": "engine",
                        "target_path": "models/model.engine",
                        "dataset_yaml": str(data_yaml),
                        "backend": "tensorrt",
                        "status": "ok",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert (
        has_matching_test_artifacts(
            str(root_dir),
            format_name="engine",
            target_path=str(weights_path),
            dataset_yaml=str(data_yaml),
            imgsz=640,
            conf=0.25,
            iou=0.5,
        )
        is False
    )


def test_run_native_tensorrt_uses_artifact_imgsz_when_requested_differs(monkeypatch, tmp_path: Path, capsys) -> None:
    root_dir = tmp_path / "run_trt_imgsz_override"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.engine"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_trt_imgsz_override", with_test=True)
    (ds / "val" / "images").mkdir(parents=True, exist_ok=True)
    (ds / "val" / "labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(ds / "val" / "images" / "a.jpg")
    (ds / "val" / "labels" / "a.txt").write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")

    def _fake_trt_infer(_engine_path, image_path, input_hw, _conf_thr, _iou_thr, _names):
        from smartrain.workflows.testing.model_test_backends import _Pred

        assert input_hw == (640, 640)
        return [_Pred(image_path=image_path, cls_id=0, conf=0.95, x1=10.0, y1=10.0, x2=30.0, y2=30.0)]

    monkeypatch.setattr(
        "smartrain.services.testing.backends.native_eval.read_model_sidecar_metadata",
        lambda _path: {"params": {"imgsz": 640}},
    )
    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners._prepare_trt_runtime",
        _stub_prepare_trt_runtime,
    )
    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners._infer_with_trt_engine",
        _fake_trt_infer,
    )

    result = run_native_format_backend(
        root_dir=str(root_dir),
        weights_path=str(weights_path),
        dataset_yaml_path=str(ds / "data.yaml"),
        format_name="engine",
        imgsz=1280,
        val_conf=0.25,
        val_iou=0.5,
        val_batch=1,
    )
    out = capsys.readouterr().out
    assert result.success is True
    assert "native eval imgsz mismatch" in out
    args_yaml = Path(format_test_dir(str(root_dir), "engine")) / "args.yaml"
    payload = json.loads(json.dumps(__import__("yaml").safe_load(args_yaml.read_text(encoding="utf-8")) or {}))
    assert int(payload.get("imgsz")) == 640


def test_run_native_onnx_backend_writes_deep_diagnostics_artifacts(monkeypatch, tmp_path: Path) -> None:
    _install_fake_onnxruntime(monkeypatch)
    root_dir = tmp_path / "run_onnx_deep"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.onnx"
    weights_path.write_bytes(b"fake")

    ds = _make_dataset(tmp_path, "ds_onnx_deep", with_test=True)

    result = run_native_format_backend(
        root_dir=str(root_dir),
        weights_path=str(weights_path),
        dataset_yaml_path=str(ds / "data.yaml"),
        format_name="onnx",
        imgsz=640,
        val_conf=0.25,
        val_iou=0.5,
        val_batch=1,
        deep_diagnostics=True,
    )

    assert result.success is True

    deep_dir = Path(format_test_dir(str(root_dir), "onnx")) / "deep_diagnostics"
    assert deep_dir.is_dir()
    assert (deep_dir / "debug_params.json").is_file()
    assert (deep_dir / "debug_test.jsonl").is_file()
    assert (deep_dir / "debug_test_summary.json").is_file()

    summary = json.loads((deep_dir / "debug_test_summary.json").read_text(encoding="utf-8"))
    assert "tp_counts_by_iou" in summary
    assert "fp_counts_by_iou" in summary
    assert isinstance(summary.get("tp_counts_by_iou"), list)
    assert isinstance(summary.get("fp_counts_by_iou"), list)
    assert len(summary["tp_counts_by_iou"]) == 10
    assert len(summary["fp_counts_by_iou"]) == 10

    # Verify JSONL schema + aggregation invariants.
    tp_sum = [0 for _ in range(10)]
    line_count = 0
    with open(deep_dir / "debug_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            line_count += 1
            assert "image_path" in payload
            assert "gts" in payload
            assert "preds" in payload
            assert "matching" in payload
            matching = payload["matching"]
            assert isinstance(matching.get("tp_counts_by_iou"), list)
            assert len(matching["tp_counts_by_iou"]) == 10
            assert isinstance(matching.get("fp_counts_by_iou"), list)
            assert len(matching["fp_counts_by_iou"]) == 10

            # best_iou hist dimensions: 10 IoU thresholds x 10 bins.
            best_tp = matching.get("best_iou_tp_hist_by_iou")
            assert isinstance(best_tp, list)
            assert len(best_tp) == 10
            assert all(isinstance(row, list) and len(row) == 10 for row in best_tp)

            tp = matching["tp_counts_by_iou"]
            tp_sum = [tp_sum[i] + int(tp[i]) for i in range(10)]

    assert line_count > 0
    assert tp_sum == [int(x) for x in summary["tp_counts_by_iou"]]


def test_ultralytics_style_metrics_payload_is_stable() -> None:
    from smartrain.workflows.testing.model_test_backends import _Gt, _Pred, _compute_ultralytics_style_payload

    names = ["obj"]
    preds = [
        _Pred("im1.jpg", 0, 0.95, 10.0, 10.0, 30.0, 30.0),
        _Pred("im2.jpg", 0, 0.90, 10.0, 10.0, 30.0, 30.0),
    ]
    gts = [
        _Gt("im1.jpg", 0, 10.0, 10.0, 30.0, 30.0),
        _Gt("im2.jpg", 0, 10.0, 10.0, 30.0, 30.0),
    ]

    payload = _compute_ultralytics_style_payload(preds, gts, names)
    assert float(payload["map50"]) > 0.99
    assert float(payload["map5095"]) > 0.99
    assert float(payload["box_p"]) > 0.99
    assert float(payload["box_r"]) > 0.99
    assert float(payload["box_f1"]) > 0.99


def test_collect_ultralytics_style_gt_filters_out_of_bounds_labels(tmp_path: Path) -> None:
    ds = tmp_path / "datasets" / "ds_invalid"
    (ds / "test" / "images").mkdir(parents=True, exist_ok=True)
    (ds / "test" / "labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(ds / "test" / "images" / "a.jpg")
    (ds / "test" / "labels" / "a.txt").write_text("0 1.10 0.5 0.2 0.2\n", encoding="utf-8")
    (ds / "data.yaml").write_text(
        "train: test/images\nval: test/images\ntest: test/images\nnames: ['obj']\n",
        encoding="utf-8",
    )

    gt_rows, image_paths, issues = collect_ultralytics_style_gt(str(ds / "data.yaml"), "test", ["obj"])
    assert len(image_paths) == 1
    assert gt_rows == []
    assert any("non-normalized or out of bounds coordinates" in str(msg) for msg in issues)


def test_pt_uni_uses_validator_style_backend(monkeypatch, tmp_path: Path) -> None:
    from smartrain.workflows.testing.model_test_backends import BackendRunResult

    root_dir = tmp_path / "run_pt_uni"
    root_dir.mkdir(parents=True, exist_ok=True)
    weights_path = root_dir / "model.pt"
    weights_path.write_bytes(b"fake")
    ds = _make_dataset(tmp_path, "ds_pt_uni", with_test=True)

    called = {"n": 0}

    def _fake_run_ultralytics_backend(**kwargs):
        called["n"] += 1
        return BackendRunResult(
            format="pt_uni",
            backend="ultralytics",
            success=True,
            test_start_time=None,
            test_end_time=None,
            inference={"imgsz": kwargs.get("imgsz"), "conf": kwargs.get("val_conf"), "iou": kwargs.get("val_iou")},
            target_path=str(weights_path),
        )

    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners.run_ultralytics_backend",
        _fake_run_ultralytics_backend,
    )

    result = run_native_format_backend(
        root_dir=str(root_dir),
        weights_path=str(weights_path),
        dataset_yaml_path=str(ds / "data.yaml"),
        format_name="pt_uni",
        imgsz=640,
        val_conf=0.001,
        val_iou=0.7,
        val_batch=1,
    )

    assert result.success is True
    assert called["n"] == 1
    assert result.backend == "unified_pt"
    assert result.inference["inference_source"] == "ultralytics_model_val"
