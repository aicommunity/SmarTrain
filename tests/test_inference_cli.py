from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from smartrain.workflows.inference.inference_backends import BackendPrediction
from smartrain.workflows.inference.inference_cli import main as inference_main
from smartrain.workflows.inference.inference_cli import _resolve_model
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace


class _FakeTensor:
    def __init__(self, data):
        self._data = data

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class _FakeBoxes:
    def __init__(self):
        self.xyxy = _FakeTensor([[10.0, 12.0, 40.0, 48.0]])
        self.cls = _FakeTensor([0.0])
        self.conf = _FakeTensor([0.91])

    def __len__(self):
        return 1


class _FakeResult:
    def __init__(self):
        self.boxes = _FakeBoxes()


class _FakeYOLO:
    last_predict_kwargs = None

    def __init__(self, _weights: str):
        self.names = {0: "obj"}

    def predict(self, **_kwargs):
        _FakeYOLO.last_predict_kwargs = dict(_kwargs)
        return [_FakeResult()]


def _install_fake_ultralytics(monkeypatch) -> None:
    fake_mod = ModuleType("ultralytics")
    fake_mod.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(128, 128, 128)).save(path)


def _latest_report_path(ws: Path) -> Path:
    root = ws / "inference"
    all_json = sorted(root.rglob("inference_results.json"))
    assert all_json, "inference_results.json not found"
    return all_json[-1]


def test_inference_folder_model_name(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)

    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "demo_model.pt", "task_type": "detection"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")
    _write_image(src / "b.jpg")

    inference_main(
        [
            "--workspace",
            str(tmp_path),
            "--model-name",
            "demo_model",
            "--data-mode",
            "folder",
            "--source-dir",
            str(src),
        ]
    )

    report = json.loads(_latest_report_path(tmp_path).read_text(encoding="utf-8"))
    assert report["summary"]["images_processed"] == 2
    assert report["model"]["name"] == "demo_model"
    assert report["source"]["mode"] == "folder"
    assert report["images"][0]["detections"][0]["bbox_roi_xyxy"] == [10.0, 12.0, 40.0, 48.0]
    assert isinstance(report.get("performance"), dict)
    assert "end_to_end" in report["performance"]
    assert "infer_only" in report["performance"]
    env_path = Path(report["artifacts"]["environment_profile"]["path_absolute"])
    assert env_path.is_file()


def test_inference_uses_gpu0_default_device_when_available(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)
    monkeypatch.setattr("smartrain.workflows.inference.inference_cli.default_device_value", lambda: "0")
    monkeypatch.setattr("smartrain.workflows.inference.inference_cli._ensure_device_available_or_exit", lambda _d: None)

    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "demo_model.pt", "task_type": "detection"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")

    inference_main(
        [
            "--workspace",
            str(tmp_path),
            "--model-name",
            "demo_model",
            "--data-mode",
            "folder",
            "--source-dir",
            str(src),
        ]
    )
    assert _FakeYOLO.last_predict_kwargs is not None
    assert _FakeYOLO.last_predict_kwargs.get("device") == "0"


def test_inference_dataset_split(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)

    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "demo_model.pt", "task_type": "detection"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ds_root = tmp_path / "datasets" / "ds_a"
    _write_image(ds_root / "test" / "images" / "x.jpg")
    (ds_root / "test" / "labels").mkdir(parents=True, exist_ok=True)
    (ds_root / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames: ['obj']\n",
        encoding="utf-8",
    )
    (tmp_path / "datasets" / "datasets_info.json").write_text(
        json.dumps({"ds_a": {"structure": "split", "classes": {"obj": 0}}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    inference_main(
        [
            "--workspace",
            str(tmp_path),
            "--model-name",
            "demo_model",
            "--data-mode",
            "dataset-split",
            "--dataset",
            "ds_a",
            "--split",
            "test",
        ]
    )

    report = json.loads(_latest_report_path(tmp_path).read_text(encoding="utf-8"))
    assert report["source"]["mode"] == "dataset-split"
    assert report["source"]["dataset"] == "ds_a"
    assert report["source"]["split"] == "test"
    assert report["summary"]["images_processed"] == 1


def test_inference_supports_engine_weights(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)

    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")
    engine_path = tmp_path / "models" / "demo_model" / "demo_model.engine"
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(b"fake-engine")

    inference_main(
        [
            "--workspace",
            str(tmp_path),
            "--weights",
            str(engine_path),
            "--data-mode",
            "folder",
            "--source-dir",
            str(src),
        ]
    )
    report = json.loads(_latest_report_path(tmp_path).read_text(encoding="utf-8"))
    assert report["model"]["weights_absolute"].endswith(".engine")
    assert report["summary"]["images_processed"] == 1


def test_inference_passes_task_hint_to_capability_resolution(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)

    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "demo_model.pt", "task_type": "detection"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")

    captured: dict[str, str] = {}

    def _fake_resolve_infer_backend(*, task_type: str, model_format: str):
        captured["task_type"] = task_type
        captured["model_format"] = model_format

        class _Caps:
            backend = "ultralytics"

        return _Caps()

    monkeypatch.setattr("smartrain.services.inference_service.resolve_infer_backend", _fake_resolve_infer_backend)
    inference_main(
        [
            "--workspace",
            str(tmp_path),
            "--model-name",
            "demo_model",
            "--data-mode",
            "folder",
            "--source-dir",
            str(src),
            "--task",
            "segment",
        ]
    )
    assert captured["task_type"] == "segmentation"
    assert captured["model_format"] == "pt"
    report = json.loads(_latest_report_path(tmp_path).read_text(encoding="utf-8"))
    assert report["task_type"] == "segmentation"
    assert report["v2"]["metrics"]["namespace"] == "segmentation"


def test_inference_passes_task_hint_to_runtime_backend_predict(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)

    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "demo_model.pt", "task_type": "detection"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")

    class _FakeBackend:
        name = "ultralytics:pt"

        def predict(self, _image_source, **kwargs):
            captured["task_type"] = str(kwargs.get("task_type"))
            return BackendPrediction(
                task_type=captured["task_type"],
                infer_only_ns=0,
                stage_ns={},
                outputs={"detections": []},
            )

    captured: dict[str, str] = {}
    monkeypatch.setattr(
        "smartrain.backends.ultralytics_adapter.UltralyticsAdapter.create_inference_backend",
        lambda self, **_kwargs: _FakeBackend(),
    )

    inference_main(
        [
            "--workspace",
            str(tmp_path),
            "--model-name",
            "demo_model",
            "--data-mode",
            "folder",
            "--source-dir",
            str(src),
            "--task",
            "classify",
        ]
    )

    assert captured["task_type"] == "classification"


def test_inference_writes_classification_task_outputs(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)

    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "demo_model.pt", "task_type": "detection"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")

    class _FakeBackend:
        name = "ultralytics:pt"

        def predict(self, _image_source, **_kwargs):
            return BackendPrediction(
                task_type="classification",
                infer_only_ns=0,
                stage_ns={},
                outputs={
                    "classification": {
                        "top1": {"class_index": 2, "class_name": "cat", "confidence": 0.91},
                        "top_k": [{"class_index": 2, "class_name": "cat", "confidence": 0.91}],
                    }
                },
            )

    monkeypatch.setattr(
        "smartrain.backends.ultralytics_adapter.UltralyticsAdapter.create_inference_backend",
        lambda self, **_kwargs: _FakeBackend(),
    )

    inference_main(
        [
            "--workspace",
            str(tmp_path),
            "--model-name",
            "demo_model",
            "--data-mode",
            "folder",
            "--source-dir",
            str(src),
            "--task",
            "classify",
        ]
    )

    report = json.loads(_latest_report_path(tmp_path).read_text(encoding="utf-8"))
    assert report["task_type"] == "classification"
    assert report["summary"]["detections_total"] == 0
    assert report["summary"]["task_outputs_total"] == 1
    assert report["images"][0]["detections"] == []
    assert report["images"][0]["task_outputs"]["classification"]["top1"]["class_name"] == "cat"


def test_inference_writes_segmentation_task_outputs(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)

    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "demo_model.pt", "task_type": "detection"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")

    class _FakeBackend:
        name = "ultralytics:pt"

        def predict(self, _image_source, **_kwargs):
            return BackendPrediction(
                task_type="segmentation",
                infer_only_ns=0,
                stage_ns={},
                outputs={
                    "segments": [
                        {
                            "bbox_roi_xyxy": [1.0, 2.0, 10.0, 12.0],
                            "class_index": 0,
                            "class_name": "obj",
                            "confidence": 0.88,
                            "polygon_roi_xy": [[1.0, 2.0], [5.0, 6.0], [10.0, 12.0]],
                        }
                    ]
                },
            )

    monkeypatch.setattr(
        "smartrain.backends.ultralytics_adapter.UltralyticsAdapter.create_inference_backend",
        lambda self, **_kwargs: _FakeBackend(),
    )

    inference_main(
        [
            "--workspace",
            str(tmp_path),
            "--model-name",
            "demo_model",
            "--data-mode",
            "folder",
            "--source-dir",
            str(src),
            "--task",
            "segment",
        ]
    )

    report = json.loads(_latest_report_path(tmp_path).read_text(encoding="utf-8"))
    assert report["task_type"] == "segmentation"
    assert report["summary"]["task_outputs_total"] == 1
    segments = report["images"][0]["task_outputs"]["segments"]
    assert len(segments) == 1
    assert segments[0]["class_name"] == "obj"
    assert segments[0]["polygon_original_xy"][0] == [1.0, 2.0]


def test_inference_fails_on_backend_capability_mismatch(tmp_path: Path, monkeypatch, capsys) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)

    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "demo_model.pt", "task_type": "detection"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")

    class _Caps:
        backend = "ultralytics"

    class _BadBackend:
        name = "foreign-runtime:pt"

        def predict(self, *_args, **_kwargs):
            raise AssertionError("predict should not be called on capability mismatch")

    monkeypatch.setattr("smartrain.services.inference_service.resolve_infer_backend", lambda **_kwargs: _Caps())
    monkeypatch.setattr(
        "smartrain.backends.ultralytics_adapter.UltralyticsAdapter.create_inference_backend",
        lambda self, **_kwargs: _BadBackend(),
    )

    with pytest.raises(SystemExit) as ex:
        inference_main(
            [
                "--workspace",
                str(tmp_path),
                "--model-name",
                "demo_model",
                "--data-mode",
                "folder",
                "--source-dir",
                str(src),
            ]
        )
    assert int(ex.value.code or 0) == 1
    err = capsys.readouterr().err
    assert "Inference backend mismatch" in err
    assert "Aborting to keep capability routing deterministic" in err


def test_inference_interactive_replay(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)

    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "demo_model.pt", "task_type": "detection"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    src = tmp_path / "images"
    _write_image(src / "a.jpg")

    monkeypatch.setattr("smartrain.workflows.inference.inference_cli.print_replay_command", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("smartrain.workflows.inference.inference_cli.sys.stdin.isatty", lambda: True)

    def _fake_interactive_fill(args, _layout):
        args.model_name = "demo_model"
        args.run = None
        args.weights = None
        args.data_mode = "folder"
        args.source_dir = str(src)
        args.limit = 0
        args.conf = 0.25
        args.device = "cpu"
        args.half = False
        args.roi_pre_detect = False
        args.non_interactive = True
        return True

    monkeypatch.setattr("smartrain.workflows.inference.inference_cli._interactive_fill", _fake_interactive_fill)
    inference_main([])
    report = json.loads(_latest_report_path(tmp_path).read_text(encoding="utf-8"))
    assert report["summary"]["images_processed"] == 1
    assert report["source"]["mode"] == "folder"


def test_inference_external_provider_parsed_from_prefixed_weights(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)
    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")

    captured: dict[str, object] = {}

    def _fake_run_external_infer(provider_id: str, repo_path: str, venv_path: str, **kwargs) -> int:
        captured["provider_id"] = provider_id
        captured["repo_path"] = repo_path
        captured["venv_path"] = venv_path
        captured["model_path"] = kwargs.get("model_path")
        return 0

    monkeypatch.setattr("smartrain.backends.implementations.ultralytics.inference.run_external_infer", _fake_run_external_infer)
    with pytest.raises(SystemExit) as ex:
        inference_main(
            [
                "--workspace",
                str(tmp_path),
                "--weights",
                "dr-yolo:yolov8n",
                "--external-repo",
                str(tmp_path / "dr-repo"),
                "--data-mode",
                "folder",
                "--source-dir",
                str(src),
            ]
        )
    assert int(ex.value.code or 0) == 0
    assert captured["provider_id"] == "dr-yolo"
    assert captured["model_path"] == "yolov8n"
    report = json.loads(_latest_report_path(tmp_path).read_text(encoding="utf-8"))
    assert report["model"]["provider"]["type"] == "external"
    assert report["model"]["provider"]["id"] == "dr-yolo"
    assert report["external_execution"]["provider_id"] == "dr-yolo"
    assert report["summary"]["task_outputs_total"] == 0
    assert report["summary"]["detections_total"] == 0
    assert report["images"] == []
    assert Path(report["artifacts"]["environment_profile"]["path_absolute"]).is_file()


def test_inference_external_provider_accepts_task_outputs_payload(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)
    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")

    def _fake_run_external_infer(provider_id: str, repo_path: str, venv_path: str, **kwargs):
        assert provider_id == "dr-yolo"
        assert kwargs.get("model_path") == "yolov8n"
        return {
            "return_code": 0,
            "images": [
                {
                    "image_path_absolute": str(src / "a.jpg"),
                    "image_path_relative": "raw_images/a.jpg",
                    "task_outputs": {
                        "classification": {
                            "top1": {"class_index": 1, "class_name": "dog", "confidence": 0.77},
                            "top_k": [{"class_index": 1, "class_name": "dog", "confidence": 0.77}],
                        }
                    },
                    "detections": [],
                }
            ],
        }

    monkeypatch.setattr("smartrain.backends.implementations.ultralytics.inference.run_external_infer", _fake_run_external_infer)
    with pytest.raises(SystemExit) as ex:
        inference_main(
            [
                "--workspace",
                str(tmp_path),
                "--weights",
                "dr-yolo:yolov8n",
                "--external-repo",
                str(tmp_path / "dr-repo"),
                "--data-mode",
                "folder",
                "--source-dir",
                str(src),
                "--task",
                "classify",
            ]
        )
    assert int(ex.value.code or 0) == 0
    report = json.loads(_latest_report_path(tmp_path).read_text(encoding="utf-8"))
    assert report["task_type"] == "classification"
    assert report["summary"]["images_processed"] == 1
    assert report["summary"]["task_outputs_total"] == 1
    assert report["images"][0]["task_outputs"]["classification"]["top1"]["class_name"] == "dog"


def test_inference_external_provider_classification_empty_payload_normalized(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)
    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")

    def _fake_run_external_infer(_provider_id: str, _repo_path: str, _venv_path: str, **_kwargs):
        return {
            "return_code": 0,
            "images": [
                {
                    "image_path_absolute": str(src / "a.jpg"),
                    "image_path_relative": "raw_images/a.jpg",
                    "task_outputs": {},
                    "detections": [],
                }
            ],
        }

    monkeypatch.setattr("smartrain.backends.implementations.ultralytics.inference.run_external_infer", _fake_run_external_infer)
    with pytest.raises(SystemExit) as ex:
        inference_main(
            [
                "--workspace",
                str(tmp_path),
                "--weights",
                "dr-yolo:yolov8n",
                "--external-repo",
                str(tmp_path / "dr-repo"),
                "--data-mode",
                "folder",
                "--source-dir",
                str(src),
                "--task",
                "classify",
            ]
        )
    assert int(ex.value.code or 0) == 0
    report = json.loads(_latest_report_path(tmp_path).read_text(encoding="utf-8"))
    assert report["task_type"] == "classification"
    assert report["summary"]["task_outputs_total"] == 0
    assert report["summary"]["capability_gap_images"] == 1
    assert report["summary"]["capability_gap_reasons"] == ["missing_task_outputs.classification"]
    assert report["images"][0]["task_outputs"] == {"classification": {}}
    assert report["images"][0]["capability_gap"] is True
    assert report["images"][0]["capability_gap_reason"] == "missing_task_outputs.classification"


def test_inference_external_provider_segmentation_empty_payload_normalized(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)
    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")

    def _fake_run_external_infer(_provider_id: str, _repo_path: str, _venv_path: str, **_kwargs):
        return {
            "return_code": 0,
            "images": [
                {
                    "image_path_absolute": str(src / "a.jpg"),
                    "image_path_relative": "raw_images/a.jpg",
                    "task_outputs": {},
                    "detections": [],
                }
            ],
        }

    monkeypatch.setattr("smartrain.backends.implementations.ultralytics.inference.run_external_infer", _fake_run_external_infer)
    with pytest.raises(SystemExit) as ex:
        inference_main(
            [
                "--workspace",
                str(tmp_path),
                "--weights",
                "dr-yolo:yolov8n",
                "--external-repo",
                str(tmp_path / "dr-repo"),
                "--data-mode",
                "folder",
                "--source-dir",
                str(src),
                "--task",
                "segment",
            ]
        )
    assert int(ex.value.code or 0) == 0
    report = json.loads(_latest_report_path(tmp_path).read_text(encoding="utf-8"))
    assert report["task_type"] == "segmentation"
    assert report["summary"]["task_outputs_total"] == 0
    assert report["summary"]["capability_gap_images"] == 1
    assert report["summary"]["capability_gap_reasons"] == ["missing_task_outputs.segments"]
    assert report["images"][0]["task_outputs"] == {"segments": []}
    assert report["images"][0]["capability_gap"] is True
    assert report["images"][0]["capability_gap_reason"] == "missing_task_outputs.segments"


def test_inference_unknown_provider_in_weights_returns_error(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)
    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")
    with pytest.raises(SystemExit) as ex:
        inference_main(
            [
                "--workspace",
                str(tmp_path),
                "--weights",
                "unknown-provider:yolov8n",
                "--data-mode",
                "folder",
                "--source-dir",
                str(src),
            ]
        )
    assert int(ex.value.code or 0) == 2
    err = capsys.readouterr().err
    assert "Unknown external provider in model ref" in err


def test_inference_rejects_unsupported_model_for_external_provider(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)
    src = tmp_path / "raw_images"
    _write_image(src / "a.jpg")
    with pytest.raises(SystemExit) as ex:
        inference_main(
            [
                "--workspace",
                str(tmp_path),
                "--weights",
                "dr-yolo:yolov7",
                "--external-repo",
                str(tmp_path / "dr-repo"),
                "--data-mode",
                "folder",
                "--source-dir",
                str(src),
            ]
        )
    assert int(ex.value.code or 0) == 2
    err = capsys.readouterr().err
    assert "is not supported by external provider" in err


def test_resolve_model_run_ignores_internal_trtprep(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))

    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")

    internal = models_dir / "run_a_imgsz1280x1280_b1_static_op17_fp32_simplify1_nms0_trtprep.onnx"
    public = models_dir / "run_a.onnx"
    internal.write_bytes(b"internal")
    public.write_bytes(b"public")

    from smartrain.core.runtime.workspace_paths import WorkspaceLayout
    import argparse

    args = argparse.Namespace(model_name=None, run=str(run_dir), weights=None)
    model_path, _name, source = _resolve_model(args, WorkspaceLayout(str(tmp_path)))
    assert source == "runs"
    assert model_path == public.resolve()


def test_resolve_model_run_prefers_newest_profile_onnx_variant(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))

    run_dir = tmp_path / "runs" / "ds_a" / "run_b"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")

    old_variant = models_dir / "run_b_imgsz640x640_b1_static_op17_fp32_simplify1_nms0.onnx"
    new_variant = models_dir / "run_b_imgsz1280x1280_b1_static_op17_fp32_simplify1_nms0.onnx"
    old_variant.write_bytes(b"old")
    new_variant.write_bytes(b"new")
    os.utime(old_variant, (1_700_000_000, 1_700_000_000))
    os.utime(new_variant, (1_800_000_000, 1_800_000_000))

    import argparse
    from smartrain.core.runtime.workspace_paths import WorkspaceLayout

    args = argparse.Namespace(model_name=None, run=str(run_dir), weights=None)
    model_path, _name, source = _resolve_model(args, WorkspaceLayout(str(tmp_path)))
    assert source == "runs"
    assert model_path == new_variant.resolve()


def test_resolve_model_uses_canonical_gateway_for_run_when_enabled(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    monkeypatch.setenv("SMARTTRAIN_CANONICAL_READ", "1")

    run_dir = tmp_path / "runs" / "ds_a" / "run_c"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "from_gateway.onnx"
    target.write_bytes(b"gateway")

    class _M:
        weights_path = str(target)
        model_id = "mid"

    class _R:
        run_id = "rid"

    class _P:
        models = [_M()]
        runs = [_R()]

    class _C:
        run_id = "rid"

    monkeypatch.setattr("smartrain.orchestrators.canonical_gateway.load_target", lambda *_a, **_k: _P())
    monkeypatch.setattr("smartrain.orchestrators.canonical_gateway.resolve_task_context", lambda *_a, **_k: _C())
    import argparse
    from smartrain.core.runtime.workspace_paths import WorkspaceLayout

    args = argparse.Namespace(model_name=None, run=str(run_dir), weights=None)
    model_path, name, source = _resolve_model(args, WorkspaceLayout(str(tmp_path)))
    assert source == "runs"
    assert name == "rid"
    assert model_path == target.resolve()


def test_resolve_model_falls_back_when_canonical_gateway_fails(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    monkeypatch.setenv("SMARTTRAIN_CANONICAL_READ", "1")

    run_dir = tmp_path / "runs" / "ds_a" / "run_d"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    fallback = models_dir / "run_d.onnx"
    fallback.write_bytes(b"fallback")
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "smartrain.orchestrators.canonical_gateway.load_target",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "smartrain.orchestrators.canonical_gateway.resolve_task_context",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    import argparse
    from smartrain.core.runtime.workspace_paths import WorkspaceLayout

    args = argparse.Namespace(model_name=None, run=str(run_dir), weights=None)
    with pytest.raises(RuntimeError):
        _resolve_model(args, WorkspaceLayout(str(tmp_path)))

