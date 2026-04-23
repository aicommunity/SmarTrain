from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from smartrain.inference_cli import main as inference_main
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace


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
    def __init__(self, _weights: str):
        self.names = {0: "obj"}

    def predict(self, **_kwargs):
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
        json.dumps({"weights_file": "demo_model.pt"}, ensure_ascii=False, indent=2),
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


def test_inference_dataset_split(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)

    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "demo_model.pt"}, ensure_ascii=False, indent=2),
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


def test_inference_interactive_replay(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    _install_fake_ultralytics(monkeypatch)

    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "demo_model.pt"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    src = tmp_path / "images"
    _write_image(src / "a.jpg")

    monkeypatch.setattr("smartrain.inference_cli.print_replay_command", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("smartrain.inference_cli.sys.stdin.isatty", lambda: True)

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

    monkeypatch.setattr("smartrain.inference_cli._interactive_fill", _fake_interactive_fill)
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

    monkeypatch.setattr("smartrain.inference_cli.run_external_infer", _fake_run_external_infer)
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
