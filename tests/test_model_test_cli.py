from __future__ import annotations

import json
import subprocess
import sys
import pytest
from pathlib import Path
from types import ModuleType

from PIL import Image

from smartrain.model_test_cli import main as smartrain_test_main
from smartrain.workspace_paths import deploy_workspace


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


def _answers(monkeypatch, values: list[str]) -> None:
    it = iter(values)
    monkeypatch.setattr("smartrain.model_test_cli.prompt_choice", lambda *args, **kwargs: next(it))
    monkeypatch.setattr("smartrain.model_test_cli.prompt_text", lambda *args, **kwargs: next(it))


def test_model_test_cli_run_uses_existing_resume_logic(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (tmp_path / "datasets" / "ds_a").mkdir(parents=True, exist_ok=True)
    ((tmp_path / "datasets" / "ds_a") / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    called: dict[str, str] = {}

    def _fake_complete(run_dir_arg: str, **kwargs) -> bool:
        called["run_dir"] = run_dir_arg
        called["workspace_root"] = kwargs["workspace_root"]
        return True

    monkeypatch.setattr("smartrain.model_test_cli.complete_missing_test_artifacts", _fake_complete)
    monkeypatch.setattr("smartrain.model_test_cli.has_complete_test_artifacts", lambda *_args, **_kwargs: False)

    smartrain_test_main(["--workspace", str(tmp_path), "--run", str(run_dir), "--formats", "pt", "-y"])
    assert called["run_dir"] == str(run_dir)
    assert called["workspace_root"] == str(tmp_path)


def test_model_test_cli_prints_selected_model_and_dataset(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_dir / "train" / "weights" / "best.onnx").write_bytes(b"fake-onnx")
    (tmp_path / "datasets" / "ds_a").mkdir(parents=True, exist_ok=True)
    ((tmp_path / "datasets" / "ds_a") / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    class _FakeResult:
        success = True
        error = None

    monkeypatch.setattr("smartrain.model_test_cli.run_native_format_backend", lambda **_kwargs: _FakeResult())
    monkeypatch.setattr("smartrain.model_test_cli.has_complete_test_artifacts", lambda *_args, **_kwargs: False)

    smartrain_test_main(["--workspace", str(tmp_path), "--run", str(run_dir), "--formats", "onnx", "-y"])
    out = capsys.readouterr().out
    assert "[INFO] Test plan:" in out
    assert "  split:   test" in out
    assert str((tmp_path / "datasets" / "ds_a" / "data.yaml")) in out
    assert "  model[onnx]:" in out
    assert str(run_dir / "train" / "weights" / "best.onnx") in out


def test_model_test_cli_interactive_replay_command_is_complete(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_dir / "train" / "weights" / "best.onnx").write_bytes(b"fake-onnx")
    dataset_yaml = tmp_path / "datasets" / "ds_a" / "data.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text("train: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    class _FakeResult:
        success = True
        error = None

    monkeypatch.setattr("smartrain.model_test_cli.is_interactive_allowed", lambda _flag: True)
    monkeypatch.setattr("smartrain.model_test_cli._pick_interactive_target", lambda _layout: (str(run_dir), str(run_dir / "train" / "weights" / "best.pt"), "runs", run_dir.name))
    monkeypatch.setattr("smartrain.model_test_cli._prompt_formats_interactive", lambda default="pt,onnx,engine,trt": ["onnx"])
    _answers(monkeypatch, [str(dataset_yaml)])
    monkeypatch.setattr("smartrain.model_test_cli.run_native_format_backend", lambda **_kwargs: _FakeResult())
    monkeypatch.setattr("smartrain.model_test_cli.has_complete_test_artifacts", lambda *_args, **_kwargs: False)

    smartrain_test_main(["--workspace", str(tmp_path)])
    out = capsys.readouterr().out
    assert "smartrain test" in out
    assert f"--run {run_dir}" in out
    assert f"--data {dataset_yaml}" in out
    assert "--formats onnx" in out


def test_model_test_cli_skips_matching_existing_test_non_interactive(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_skip"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    onnx_path = run_dir / "train" / "weights" / "best.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    dataset_yaml = tmp_path / "datasets" / "ds_a" / "data.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text("train: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    called = {"native": 0}

    class _FakeResult:
        success = True
        error = None

    def _fake_native(**_kwargs):
        called["native"] += 1
        return _FakeResult()

    monkeypatch.setattr("smartrain.model_test_cli.run_native_format_backend", _fake_native)
    monkeypatch.setattr("smartrain.model_test_cli.has_matching_test_artifacts", lambda *_args, **_kwargs: True)

    smartrain_test_main(["--workspace", str(tmp_path), "--run", str(run_dir), "--formats", "onnx", "-y"])
    out = capsys.readouterr().out
    assert called["native"] == 0
    assert "matching test artifacts already exist for this model and dataset, skipping." in out


def test_model_test_cli_force_reruns_matching_existing_test_non_interactive(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_force"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    onnx_path = run_dir / "train" / "weights" / "best.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    dataset_yaml = tmp_path / "datasets" / "ds_a" / "data.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text("train: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    called = {"native": 0}

    class _FakeResult:
        success = True
        error = None

    def _fake_native(**_kwargs):
        called["native"] += 1
        return _FakeResult()

    monkeypatch.setattr("smartrain.model_test_cli.run_native_format_backend", _fake_native)
    monkeypatch.setattr("smartrain.model_test_cli.has_matching_test_artifacts", lambda *_args, **_kwargs: True)

    smartrain_test_main(["--workspace", str(tmp_path), "--run", str(run_dir), "--formats", "onnx", "--force", "-y"])
    out = capsys.readouterr().out
    assert called["native"] == 1
    assert "matching test artifacts already exist for this model and dataset, skipping." not in out


def test_model_test_cli_reports_missing_data_yaml_without_traceback(tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_bad_data"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")

    with pytest.raises(SystemExit):
        smartrain_test_main(
            [
                "--workspace",
                str(tmp_path),
                "--run",
                str(run_dir),
                "--formats",
                "onnx",
                "--data",
                str(tmp_path / "datasets" / "..." / "data.yaml"),
                "-y",
            ]
        )
    err = capsys.readouterr().err
    assert "data.yaml not found for dataset" in err
    assert "replace with full real path" in err


def test_model_test_cli_reports_missing_run_path_without_traceback(tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    missing_run = tmp_path / "runs" / "..." / "run_x"
    with pytest.raises(SystemExit):
        smartrain_test_main(
            [
                "--workspace",
                str(tmp_path),
                "--run",
                str(missing_run),
                "--formats",
                "onnx",
                "--data",
                str(tmp_path / "datasets" / "ds_a" / "data.yaml"),
                "-y",
            ]
        )
    err = capsys.readouterr().err
    assert "Run directory not found" in err
    assert "replace with full real path" in err


def test_model_test_cli_skips_matching_existing_test_from_args_yaml_fallback(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_skip_args_yaml"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    onnx_path = run_dir / "train" / "weights" / "best.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    dataset_yaml = tmp_path / "datasets" / "ds_a" / "data.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text("train: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "test_metrics_onnx.csv").write_text("metric,value\nmAP50,0.9\n", encoding="utf-8")
    (run_dir / "test_onnx").mkdir(parents=True, exist_ok=True)
    (run_dir / "test_onnx" / "args.yaml").write_text(f"data: {dataset_yaml}\n", encoding="utf-8")
    for name in ("pr.csv", "pr_per_class.csv"):
        (run_dir / "test_onnx" / name).write_text("x", encoding="utf-8")
    (run_dir / "confidence_recommendations_test_onnx.json").write_text(
        json.dumps({"objectives": {"A": {"global": {"threshold": 0.1}}, "B": {"global": {"threshold": 0.1}}, "C": {"global": {"threshold": 0.1}}}}),
        encoding="utf-8",
    )
    (run_dir / "confidence_recommendations_val_onnx.json").write_text(
        json.dumps({"objectives": {"A": {"global": {"threshold": 0.1}}, "B": {"global": {"threshold": 0.1}}, "C": {"global": {"threshold": 0.1}}}}),
        encoding="utf-8",
    )
    (run_dir / "test_artifacts_manifest.json").write_text(
        json.dumps(
            {
                "formats": {
                    "onnx": {
                        "format": "onnx",
                        "target_path": "train/weights/best.onnx",
                        "status": "ok",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    called = {"native": 0}

    class _FakeResult:
        success = True
        error = None

    def _fake_native(**_kwargs):
        called["native"] += 1
        return _FakeResult()

    monkeypatch.setattr("smartrain.model_test_cli.run_native_format_backend", _fake_native)

    smartrain_test_main(["--workspace", str(tmp_path), "--run", str(run_dir), "--formats", "onnx", "-y"])
    out = capsys.readouterr().out
    assert called["native"] == 0
    assert "matching test artifacts already exist for this model and dataset, skipping." in out


def test_model_test_cli_run_finds_engine_in_nested_dir(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_engine_nested"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake-pt")
    engine_path = run_dir / "exports" / "demo.engine"
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(b"fake-engine")
    dataset_dir = tmp_path / "datasets" / "ds_a"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data.yaml").write_text("train: train/images\nval: val/images\ntest: test/images\nnames: ['obj']\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    called: dict[str, str] = {}

    def _fake_isolated(**kwargs):
        called["weights_path"] = kwargs["weights_path"]
        called["format_name"] = kwargs["format_name"]
        return True, None

    monkeypatch.setattr("smartrain.model_test_cli._run_native_backend_isolated", _fake_isolated)
    monkeypatch.setattr("smartrain.model_test_cli.has_complete_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.model_test_cli.has_matching_test_artifacts", lambda *_args, **_kwargs: False)

    smartrain_test_main(["--workspace", str(tmp_path), "--run", str(run_dir), "--formats", "engine", "-y"])
    assert called["format_name"] == "engine"
    assert called["weights_path"] == str(engine_path)


def test_model_test_cli_prompts_before_rerun_matching_existing_test_interactive(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_prompt"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    onnx_path = run_dir / "train" / "weights" / "best.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    dataset_yaml = tmp_path / "datasets" / "ds_a" / "data.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text("train: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    called = {"native": 0, "prompt": 0}

    class _FakeResult:
        success = True
        error = None

    monkeypatch.setattr("smartrain.model_test_cli.is_interactive_allowed", lambda _flag: True)
    monkeypatch.setattr("smartrain.model_test_cli._pick_interactive_target", lambda _layout: (str(run_dir), str(run_dir / "train" / "weights" / "best.pt"), "runs", run_dir.name))
    monkeypatch.setattr("smartrain.model_test_cli._prompt_formats_interactive", lambda default="pt,onnx,engine,trt": ["onnx"])
    _answers(monkeypatch, [str(dataset_yaml)])

    def _fake_prompt_yes_no(label: str, default: bool = False) -> bool:
        called["prompt"] += 1
        assert "matching test artifacts already exist" in label
        assert default is False
        return False

    def _fake_native(**_kwargs):
        called["native"] += 1
        return _FakeResult()

    monkeypatch.setattr("smartrain.model_test_cli.prompt_yes_no", _fake_prompt_yes_no)
    monkeypatch.setattr("smartrain.model_test_cli.run_native_format_backend", _fake_native)
    monkeypatch.setattr("smartrain.model_test_cli.has_matching_test_artifacts", lambda *_args, **_kwargs: True)

    smartrain_test_main(["--workspace", str(tmp_path)])
    _out = capsys.readouterr().out
    assert called["prompt"] == 1
    assert called["native"] == 0


def test_model_test_cli_model_builds_pt_for_promoted_model(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"weights_file": "demo_model.pt", "source_run": ""}, ensure_ascii=False),
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "datasets" / "ds_a"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )

    called: dict[str, str] = {}

    class _FakeResult:
        success = True
        error = None

    def _fake_backend(**kwargs):
        called["root_dir"] = kwargs["root_dir"]
        called["weights_path"] = kwargs["weights_path"]
        called["dataset_yaml_path"] = kwargs["dataset_yaml_path"]
        called["format_name"] = kwargs["format_name"]
        return _FakeResult()

    monkeypatch.setattr("smartrain.model_test_cli.run_ultralytics_backend", _fake_backend)
    monkeypatch.setattr("smartrain.model_test_cli.has_complete_test_artifacts", lambda *_args, **_kwargs: False)

    smartrain_test_main(
        [
            "--workspace",
            str(tmp_path),
            "--model-name",
            "demo_model",
            "--formats",
            "pt",
            "--data",
            str(dataset_dir),
            "-y",
        ]
    )
    assert called["root_dir"] == str(model_dir)
    assert called["weights_path"].endswith("demo_model.pt")
    assert called["dataset_yaml_path"].endswith("data.yaml")
    assert called["format_name"] == "pt"


def test_model_test_cli_continues_when_tensorrt_export_fails(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_dir / "train" / "weights" / "best.onnx").write_bytes(b"fake-onnx")
    (tmp_path / "datasets" / "ds_a").mkdir(parents=True, exist_ok=True)
    ((tmp_path / "datasets" / "ds_a") / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\nnames: ['obj']\n",
        encoding="utf-8",
    )
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    class _FakeResult:
        def __init__(self, success: bool, error: str | None = None) -> None:
            self.success = success
            self.error = error

    monkeypatch.setattr(
        "smartrain.model_test_cli.run_native_format_backend",
        lambda **kwargs: _FakeResult(success=(kwargs["format_name"] == "onnx"), error=None if kwargs["format_name"] == "onnx" else "x"),
    )
    monkeypatch.setattr("smartrain.model_test_cli.has_complete_test_artifacts", lambda *_args, **_kwargs: False)

    smartrain_test_main(
        [
            "--workspace",
            str(tmp_path),
            "--run",
            str(run_dir),
            "--formats",
            "onnx,engine,trt",
            "-y",
        ]
    )

    manifest = json.loads((run_dir / "test_artifacts_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formats"]["engine"]["status"] == "failed"
    assert manifest["formats"]["trt"]["status"] == "failed"


def test_model_test_cli_engine_crash_isolated_and_recorded(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_engine_crash"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_dir / "engine.engine").write_bytes(b"fake")
    (tmp_path / "datasets" / "ds_a").mkdir(parents=True, exist_ok=True)
    ((tmp_path / "datasets" / "ds_a") / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\nnames: ['obj']\n",
        encoding="utf-8",
    )
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr("smartrain.model_test_cli.has_complete_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.model_test_cli.has_matching_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        "smartrain.model_test_cli.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=-6, stdout="", stderr="Aborted (core dumped)"),
    )

    smartrain_test_main(
        [
            "--workspace",
            str(tmp_path),
            "--run",
            str(run_dir),
            "--formats",
            "engine",
            "-y",
        ]
    )

    manifest = json.loads((run_dir / "test_artifacts_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formats"]["engine"]["status"] == "failed"
    assert "signal 6" in str(manifest["formats"]["engine"]["error"]).lower()


def test_model_test_cli_run_builds_onnx_artifacts_end_to_end(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _install_fake_onnxruntime(monkeypatch)

    run_dir = tmp_path / "runs" / "ds_a" / "run_onnx"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake-pt")
    (run_dir / "train" / "weights" / "best.onnx").write_bytes(b"fake-onnx")
    dataset_dir = tmp_path / "datasets" / "ds_a"
    (dataset_dir / "test" / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "test" / "labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(dataset_dir / "test" / "images" / "a.jpg")
    (dataset_dir / "test" / "labels" / "a.txt").write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")
    (dataset_dir / "data.yaml").write_text(
        "train: train/images\nval: test/images\ntest: test/images\nnames: ['obj']\n",
        encoding="utf-8",
    )
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr("smartrain.model_test_cli.has_complete_test_artifacts", lambda *_args, **_kwargs: False)

    smartrain_test_main(
        [
            "--workspace",
            str(tmp_path),
            "--run",
            str(run_dir),
            "--formats",
            "onnx",
            "-y",
        ]
    )

    assert (run_dir / "test_metrics_onnx.csv").is_file()
    assert (run_dir / "test_onnx" / "pr.csv").is_file()
    assert (run_dir / "test_onnx" / "pr_per_class.csv").is_file()
    assert (run_dir / "test_onnx" / "BoxPR_curve.png").is_file()
    assert (run_dir / "test_onnx" / "BoxF1_curve.png").is_file()
    assert (run_dir / "test_onnx" / "BoxP_curve.png").is_file()
    assert (run_dir / "test_onnx" / "BoxR_curve.png").is_file()
    assert (run_dir / "test_onnx" / "confusion_matrix.png").is_file()
    assert (run_dir / "test_onnx" / "confusion_matrix_normalized.png").is_file()
    assert (run_dir / "confidence_recommendations_test_onnx.json").is_file()
    assert (run_dir / "confidence_recommendations_val_onnx.json").is_file()
    manifest = json.loads((run_dir / "test_artifacts_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formats"]["onnx"]["status"] == "ok"
