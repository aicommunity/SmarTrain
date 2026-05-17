from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import pytest
from pathlib import Path
from types import ModuleType

from PIL import Image

from smartrain.workflows.testing.model_test_cli import (
    _check_onnx_format_preflight,
    _discover_run_artifact_candidates,
    _infer_task_from_training_metadata,
    _prompt_export_backends_interactive,
    _prompt_artifact_selection_interactive,
    _resolve_existing_artifact,
    main as smartrain_test_main,
)
from smartrain.workflows.testing.model_test_service import (
    artifacts_manifest_path_for_write,
    format_metrics_path,
    format_recommendation_path,
    format_test_dir,
    test_artifacts_manifest_path as manifest_path_fn,
)
from smartrain.core.runtime.workspace_paths import deploy_workspace


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
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.prompt_choice", lambda *args, **kwargs: next(it))
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.prompt_text", lambda *args, **kwargs: next(it))


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

    called: dict[str, Any] = {}

    from datetime import datetime

    from smartrain.workflows.testing.model_test_backends import BackendRunResult

    def _fake_run_ultralytics_backend(**kwargs):
        called["root_dir"] = kwargs.get("root_dir")
        return BackendRunResult(
            format="pt",
            backend="ultralytics",
            success=True,
            test_start_time=datetime.now(),
            test_end_time=datetime.now(),
            inference={},
            target_path=kwargs.get("weights_path"),
        )

    monkeypatch.setattr(
        "smartrain.core.workflow_adapters.testing_runtime_api.run_ultralytics_backend",
        _fake_run_ultralytics_backend,
    )
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_args, **_kwargs: False)
    class _FakeResult:
        success = True
        error = None
    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", lambda **_kwargs: _FakeResult())

    smartrain_test_main(
        ["--workspace", str(tmp_path), "--run", str(run_dir), "--formats", "pt", "--no-perf", "-y"]
    )
    assert called["root_dir"] == str(run_dir)


def test_model_test_cli_rejects_public_pt_uni_format(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (tmp_path / "datasets" / "ds_a").mkdir(parents=True, exist_ok=True)
    ((tmp_path / "datasets" / "ds_a") / "data.yaml").write_text("train: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        smartrain_test_main(["--workspace", str(tmp_path), "--run", str(run_dir), "--formats", "pt_uni", "-y"])


def test_infer_task_from_metadata_uses_canonical_gateway_when_enabled(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))

    class _C:
        task_type = "segmentation"

    monkeypatch.setattr("smartrain.orchestrators.unified_gateway.resolve_task_context", lambda *_a, **_k: _C())
    assert _infer_task_from_training_metadata(str(tmp_path)) == "segment"


def test_infer_task_from_metadata_falls_back_to_legacy_when_gateway_fails(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setattr(
        "smartrain.orchestrators.unified_gateway.resolve_task_context",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError):
        _infer_task_from_training_metadata(str(tmp_path))


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

    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", lambda **_kwargs: _FakeResult())
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface._check_onnx_format_preflight", lambda _policy: (True, None))

    smartrain_test_main(["--workspace", str(tmp_path), "--run", str(run_dir), "--formats", "onnx", "-y"])
    out = capsys.readouterr().out
    assert "[INFO] Test plan:" in out
    assert "  split:   test" in out
    assert str((tmp_path / "datasets" / "ds_a" / "data.yaml")) in out
    assert "  model[onnx]:" in out
    assert str(run_dir / "models" / "best.onnx") in out or str(run_dir / "train-ultralytics" / "weights" / "best.onnx") in out


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

    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.is_interactive_allowed", lambda _flag: True)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface._pick_interactive_target", lambda _layout: (str(run_dir), str(run_dir / "train" / "weights" / "best.pt"), "runs", run_dir.name))
    monkeypatch.setattr(
        "smartrain.services.testing.model_test_cli_surface._prompt_export_backends_interactive",
        lambda _root, _c: ["onnx"],
    )
    monkeypatch.setattr(
        "smartrain.services.testing.model_test_cli_surface._prompt_artifact_selection_interactive",
        lambda _candidates: [("onnx", str(run_dir / "train" / "weights" / "best.onnx"))],
    )
    _answers(monkeypatch, [str(dataset_yaml)])
    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", lambda **_kwargs: _FakeResult())
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_args, **_kwargs: False)

    smartrain_test_main(["--workspace", str(tmp_path)])
    out = capsys.readouterr().out
    assert "smartrain test" in out
    assert f"--run {run_dir}" in out
    assert f"--data {dataset_yaml}" in out
    assert "--formats onnx" in out


def test_model_test_cli_interactive_pt_not_queued_for_native_backend(monkeypatch, tmp_path: Path, capsys) -> None:
    """Selecting .pt in interactive mode must not call run_native_format_backend(format_name='pt')."""
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    pt_path = run_dir / "train" / "weights" / "best.pt"
    pt_path.write_bytes(b"fake")
    onnx_path = run_dir / "train" / "weights" / "best.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    dataset_yaml = tmp_path / "datasets" / "ds_a" / "data.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text("train: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    native_formats: list[str] = []

    class _FakeResult:
        success = True
        error = None

    def _fake_native(**kwargs):
        native_formats.append(str(kwargs.get("format_name", "")))
        return _FakeResult()

    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.is_interactive_allowed", lambda _flag: True)
    monkeypatch.setattr(
        "smartrain.services.testing.model_test_cli_surface._pick_interactive_target",
        lambda _layout: (str(run_dir), str(pt_path), "runs", run_dir.name),
    )
    monkeypatch.setattr(
        "smartrain.services.testing.model_test_cli_surface._prompt_export_backends_interactive",
        lambda _root, _c: ["pt", "onnx"],
    )
    monkeypatch.setattr(
        "smartrain.services.testing.model_test_cli_surface._prompt_artifact_selection_interactive",
        lambda _candidates: [("pt", str(pt_path)), ("onnx", str(onnx_path))],
    )
    _answers(monkeypatch, [str(dataset_yaml)])
    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_ultralytics_backend", lambda **_kwargs: _FakeResult())
    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", _fake_native)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface._check_onnx_format_preflight", lambda _policy: (True, None))
    _install_fake_onnxruntime(monkeypatch)

    smartrain_test_main(["--workspace", str(tmp_path)])
    out = capsys.readouterr().out
    assert "pt" not in native_formats


def test_model_test_cli_classification_runs_internal_pt_uni_compare(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_cls"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
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

    seen_formats: list[str] = []
    pt_uni_tasks: list[str | None] = []

    def _fake_native(**kwargs):
        seen_formats.append(str(kwargs.get("format_name", "")))
        pt_uni_tasks.append(kwargs.get("task_type"))
        return _FakeResult()

    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_ultralytics_backend", lambda **_kwargs: _FakeResult())
    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", _fake_native)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface._check_onnx_format_preflight", lambda _policy: (True, None))

    smartrain_test_main(
        ["--workspace", str(tmp_path), "--run", str(run_dir), "--formats", "pt", "--task", "classify", "-y"]
    )
    out = capsys.readouterr().out
    assert "Generating internal PT-vs-PT-uni comparison artifacts." in out
    assert "pt_uni" in seen_formats
    assert "classification" in pt_uni_tasks


def test_prompt_export_backends_lists_all_formats_and_skips_missing(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "run1"
    root.mkdir(parents=True, exist_ok=True)
    pt = root / "models" / "m.pt"
    pt.parent.mkdir(parents=True, exist_ok=True)
    pt.write_bytes(b"x")
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.preferred_run_model_path", lambda _r, _ext=".pt": str(pt))
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.print_numbered_options", lambda *a, **k: None)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.prompt_text", lambda _p, default="": "1,2,3,4")
    out = _prompt_export_backends_interactive(
        str(root),
        {"pt": [str(pt)], "onnx": [], "engine": [], "trt": []},
    )
    assert out == ["pt"]


def test_prompt_artifact_selection_interactive_always_prompts_even_for_single_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.print_numbered_options", lambda *a, **k: None)
    prompts: list[str] = []

    def _pt(prompt: str, default: str = "") -> str:
        prompts.append(prompt)
        return default

    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.prompt_text", _pt)
    one_pt = str(tmp_path / "w.pt")
    out = _prompt_artifact_selection_interactive({"pt": [one_pt]})
    assert out == [("pt", one_pt)]
    assert prompts and "Select models for test" in prompts[0]


def test_interactive_run_without_formats_defaults_to_all_export_formats(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    dataset_yaml = tmp_path / "datasets" / "ds_a" / "data.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text("train: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.is_interactive_allowed", lambda _f: True)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.prompt_yes_no", lambda *a, **k: False)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_matching_test_artifacts", lambda *_a, **_k: True)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "smartrain.services.testing.model_test_cli_surface._prompt_export_backends_interactive",
        lambda _root, _c: ["pt"],
    )
    monkeypatch.setattr(
        "smartrain.services.testing.model_test_cli_surface._prompt_artifact_selection_interactive",
        lambda _c: [],
    )

    class _FakeResult:
        success = True
        error = None

    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", lambda **_k: _FakeResult())
    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_ultralytics_backend", lambda **_k: _FakeResult())

    smartrain_test_main(["--workspace", str(tmp_path), "--run", str(run_dir)])
    out = capsys.readouterr().out
    assert "formats: pt" in out


def test_model_test_cli_pt_uni_matching_uses_imgsz_after_metadata_defaults(monkeypatch, tmp_path: Path) -> None:
    """pt_uni skip/match must compare effective imgsz (args after defaults), not pre-default None."""
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    ds = tmp_path / "datasets" / "ds_a" / "data.yaml"
    ds.parent.mkdir(parents=True)
    ds.write_text("train: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}},
                "inference": {"imgsz": 1280, "conf": 0.001, "iou": 0.7},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pt_uni_kw: list[dict[str, Any]] = []

    def _fake_matching(_root_dir: str, **kwargs: Any) -> bool:
        if kwargs.get("format_name") == "pt_uni":
            pt_uni_kw.append(
                {"imgsz": kwargs.get("imgsz"), "conf": kwargs.get("conf"), "iou": kwargs.get("iou")}
            )
            return False
        return True

    class _FakeResult:
        success = True
        error = None

    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_matching_test_artifacts", _fake_matching)
    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", lambda **_k: _FakeResult())

    smartrain_test_main(
        [
            "--workspace",
            str(tmp_path),
            "--run",
            str(run_dir),
            "--data",
            str(ds),
            "--formats",
            "pt",
            "-y",
        ]
    )
    assert pt_uni_kw
    assert pt_uni_kw[0]["imgsz"] == 1280
    assert pt_uni_kw[0]["conf"] == 0.001


def test_model_test_cli_replay_contains_perf_flags(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_perf" / "run_perf"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_dir / "train" / "weights" / "best.onnx").write_bytes(b"fake-onnx")
    dataset_yaml = tmp_path / "datasets" / "ds_perf" / "data.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text("train: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_perf"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    class _FakeResult:
        success = True
        error = None

    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", lambda **_kwargs: _FakeResult())
    smartrain_test_main(
        [
            "--workspace",
            str(tmp_path),
            "--run",
            str(run_dir),
            "--data",
            str(dataset_yaml),
            "--formats",
            "onnx",
            "--perf",
            "--perf-warmup-images",
            "7",
            "-y",
        ]
    )
    out = capsys.readouterr().out
    assert "--perf" in out
    assert "--perf-warmup-images 7" in out


def test_check_onnx_preflight_respects_gpu_strict(monkeypatch) -> None:
    fake_mod = ModuleType("onnxruntime")
    fake_mod.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_mod)
    ok, reason = _check_onnx_format_preflight("gpu_strict")
    assert ok is False
    assert "CUDAExecutionProvider is unavailable" in str(reason)


def test_model_test_cli_cpu_device_forces_cpu_only_onnx_policy(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_cpu"
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

    captured: dict[str, str] = {}

    def _fake_native(**kwargs):
        captured["policy"] = str(kwargs.get("onnx_provider_policy"))
        captured["device"] = str(kwargs.get("runtime_device"))
        return _FakeResult()

    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", _fake_native)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_matching_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface._check_onnx_format_preflight", lambda _policy: (True, None))

    smartrain_test_main(
        [
            "--workspace",
            str(tmp_path),
            "--run",
            str(run_dir),
            "--data",
            str(dataset_yaml),
            "--formats",
            "onnx",
            "--device",
            "cpu",
            "-y",
        ]
    )
    assert captured.get("policy") == "cpu_only"
    assert captured.get("device") == "cpu"

def test_model_test_cli_skips_matching_existing_test_non_interactive(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_skip"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    onnx_path = run_dir / "train" / "weights" / "best.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    (run_dir / "models" / "best.onnx").write_bytes(b"fake-onnx")
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

    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", _fake_native)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_matching_test_artifacts", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface._check_onnx_format_preflight", lambda _policy: (True, None))

    smartrain_test_main(["--workspace", str(tmp_path), "--run", str(run_dir), "--formats", "onnx", "-y"])
    out = capsys.readouterr().out
    assert called["native"] == 0
    assert "matching test artifacts already exist for this model and dataset, skipping." in out


def test_model_test_cli_force_reruns_matching_existing_test_non_interactive(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_force"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    onnx_path = run_dir / "train" / "weights" / "best.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    (run_dir / "models" / "best.onnx").write_bytes(b"fake-onnx")
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

    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", _fake_native)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_matching_test_artifacts", lambda *_args, **_kwargs: True)

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
    (run_dir / "train-ultralytics" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    onnx_path = run_dir / "train-ultralytics" / "weights" / "best.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    (run_dir / "models" / "best.onnx").write_bytes(b"fake-onnx")
    dataset_yaml = tmp_path / "datasets" / "ds_a" / "data.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text("train: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics_path = Path(format_metrics_path(str(run_dir), "onnx"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("metric,value\nmAP50,0.9\n", encoding="utf-8")
    test_onnx_dir = Path(format_test_dir(str(run_dir), "onnx"))
    test_onnx_dir.mkdir(parents=True, exist_ok=True)
    (test_onnx_dir / "args.yaml").write_text(f"data: {dataset_yaml}\n", encoding="utf-8")
    from smartrain.workflows.testing.ultralytics_test_contract import native_format_rich_files_required

    for name in native_format_rich_files_required():
        dest = test_onnx_dir / name
        if dest.is_file():
            continue
        if name.endswith(".png"):
            dest.write_bytes(b"\x89PNG\r\n\x1a\n")
        else:
            dest.write_text("x", encoding="utf-8")
    Path(format_recommendation_path(str(run_dir), "test", "onnx")).write_text(
        json.dumps({"objectives": {"A": {"global": {"threshold": 0.1}}, "B": {"global": {"threshold": 0.1}}, "C": {"global": {"threshold": 0.1}}}}),
        encoding="utf-8",
    )
    Path(format_recommendation_path(str(run_dir), "val", "onnx")).write_text(
        json.dumps({"objectives": {"A": {"global": {"threshold": 0.1}}, "B": {"global": {"threshold": 0.1}}, "C": {"global": {"threshold": 0.1}}}}),
        encoding="utf-8",
    )
    Path(manifest_path_fn(str(run_dir))).write_text(
        json.dumps(
            {
                "formats": {
                    "onnx": {
                        "format": "onnx",
                            "target_path": "models/best.onnx",
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

    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", _fake_native)

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

    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_backend_isolated", _fake_isolated)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_matching_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.check_native_format_preflight", lambda _fmt: (True, None))

    smartrain_test_main(["--workspace", str(tmp_path), "--run", str(run_dir), "--formats", "engine", "-y"])
    assert called["format_name"] == "engine"
    assert called["weights_path"] == str(engine_path)


def test_model_test_cli_run_materializes_legacy_tests_manifest_to_new_layout(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_manifest_migration"
    (run_dir / "test_onnx").mkdir(parents=True, exist_ok=True)
    (run_dir / "test_onnx" / "pr.csv").write_text("legacy", encoding="utf-8")
    (run_dir / "test_artifacts_manifest.json").write_text('{"formats":{}}', encoding="utf-8")
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    onnx_path = run_dir / "models" / "model.onnx"
    onnx_path.write_bytes(b"fake")
    dataset_dir = tmp_path / "datasets" / "ds_a"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data.yaml").write_text("train: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"path_under_workspace": "datasets/ds_a"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    class _FakeResult:
        success = True
        error = None

    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_matching_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface._resolve_existing_artifact", lambda **_kwargs: str(onnx_path))
    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", lambda **_kwargs: _FakeResult())

    smartrain_test_main(["--workspace", str(tmp_path), "--run", str(run_dir), "--formats", "onnx", "-y"])
    assert Path(artifacts_manifest_path_for_write(str(run_dir))).is_file()


def test_model_test_cli_prompts_before_rerun_matching_existing_test_interactive(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_prompt"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake-pt")
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

    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.is_interactive_allowed", lambda _flag: True)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface._pick_interactive_target", lambda _layout: (str(run_dir), str(run_dir / "train" / "weights" / "best.pt"), "runs", run_dir.name))
    monkeypatch.setattr(
        "smartrain.services.testing.model_test_cli_surface._prompt_export_backends_interactive",
        lambda _root, _c: ["onnx"],
    )
    monkeypatch.setattr(
        "smartrain.services.testing.model_test_cli_surface._prompt_artifact_selection_interactive",
        lambda _candidates: [("onnx", str(run_dir / "train" / "weights" / "best.onnx"))],
    )
    _answers(monkeypatch, [str(dataset_yaml)])

    def _fake_prompt_yes_no(label: str, default: bool = False) -> bool:
        called["prompt"] += 1
        assert "matching test artifacts already exist" in label
        assert default is False
        return False

    def _fake_native(**_kwargs):
        called["native"] += 1
        return _FakeResult()

    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.prompt_yes_no", _fake_prompt_yes_no)
    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend", _fake_native)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_matching_test_artifacts", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface._check_onnx_format_preflight", lambda _policy: (True, None))

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
        json.dumps(
            {"weights_file": "demo_model.pt", "source_run": "", "task_type": "detection"},
            ensure_ascii=False,
        ),
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

    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_ultralytics_backend", _fake_backend)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_args, **_kwargs: False)

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
        "smartrain.core.workflow_adapters.testing_runtime_api.run_native_format_backend",
        lambda **kwargs: _FakeResult(success=(kwargs["format_name"] == "onnx"), error=None if kwargs["format_name"] == "onnx" else "x"),
    )
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_args, **_kwargs: False)

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

    manifest = json.loads(Path(manifest_path_fn(str(run_dir))).read_text(encoding="utf-8"))
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

    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_matching_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface._check_native_format_preflight", lambda _fmt: (True, None))
    monkeypatch.setattr(
        "smartrain.services.testing.model_test_cli_surface.subprocess.run",
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

    manifest = json.loads(Path(manifest_path_fn(str(run_dir))).read_text(encoding="utf-8"))
    assert manifest["formats"]["engine"]["status"] == "failed"
    assert "signal 6" in str(manifest["formats"]["engine"]["error"]).lower()


def test_model_test_cli_engine_preflight_fail_skips_isolated(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_engine_preflight"
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

    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_matching_test_artifacts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.check_native_format_preflight", lambda _fmt: (False, "python CUDA runtime is unavailable"))
    calls = {"isolated": 0}

    def _fake_isolated(**_kwargs):
        calls["isolated"] += 1
        return True, None

    monkeypatch.setattr("smartrain.core.workflow_adapters.testing_runtime_api.run_native_backend_isolated", _fake_isolated)

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

    assert calls["isolated"] == 0
    manifest = json.loads(Path(manifest_path_fn(str(run_dir))).read_text(encoding="utf-8"))
    assert manifest["formats"]["engine"]["status"] == "failed"
    assert "python cuda runtime is unavailable" in str(manifest["formats"]["engine"]["error"]).lower()


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
    monkeypatch.setattr("smartrain.services.testing.model_test_cli_surface.has_complete_test_artifacts", lambda *_args, **_kwargs: False)

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

    assert Path(format_metrics_path(str(run_dir), "onnx")).is_file()
    onnx_test_dir = Path(format_test_dir(str(run_dir), "onnx"))
    assert (onnx_test_dir / "pr.csv").is_file()
    assert (onnx_test_dir / "pr_per_class.csv").is_file()
    assert (onnx_test_dir / "BoxPR_curve.png").is_file()
    assert (onnx_test_dir / "BoxF1_curve.png").is_file()
    assert (onnx_test_dir / "BoxP_curve.png").is_file()
    assert (onnx_test_dir / "BoxR_curve.png").is_file()
    assert (onnx_test_dir / "confusion_matrix.png").is_file()
    assert (onnx_test_dir / "confusion_matrix_normalized.png").is_file()
    assert Path(format_recommendation_path(str(run_dir), "test", "onnx")).is_file()
    assert Path(format_recommendation_path(str(run_dir), "val", "onnx")).is_file()
    manifest = json.loads(Path(manifest_path_fn(str(run_dir))).read_text(encoding="utf-8"))
    assert manifest["formats"]["onnx"]["status"] == "ok"


def test_discover_run_artifact_candidates_hides_internal_trtprep(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_hidden_trtprep"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    (models_dir / "run_hidden_trtprep.pt").write_bytes(b"pt")
    public_onnx = models_dir / "run_hidden_trtprep.onnx"
    public_onnx.write_bytes(b"onnx")
    internal_onnx = models_dir / "run_hidden_trtprep_imgsz1280x1280_b1_static_op17_fp32_simplify1_nms0_trtprep.onnx"
    internal_onnx.write_bytes(b"onnx")

    candidates = _discover_run_artifact_candidates(str(run_dir), ["onnx"])
    onnx_list = candidates.get("onnx", [])
    assert str(public_onnx) in onnx_list
    assert str(internal_onnx) not in onnx_list


def test_resolve_existing_artifact_prefers_newest_profile_onnx(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_a" / "run_pref_onnx"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")

    old_variant = models_dir / "run_pref_onnx_imgsz640x640_b1_static_op17_fp32_simplify1_nms0.onnx"
    new_variant = models_dir / "run_pref_onnx_imgsz1280x1280_b1_static_op17_fp32_simplify1_nms0.onnx"
    internal = models_dir / "run_pref_onnx_imgsz1280x1280_b1_static_op17_fp32_simplify1_nms0_trtprep.onnx"
    old_variant.write_bytes(b"old")
    new_variant.write_bytes(b"new")
    internal.write_bytes(b"cache")
    os.utime(old_variant, (1_700_000_000, 1_700_000_000))
    os.utime(new_variant, (1_800_000_000, 1_800_000_000))

    selected = _resolve_existing_artifact(
        root_dir=str(run_dir),
        primary_path=str(models_dir / "run_pref_onnx.pt"),
        format_name="onnx",
        target_kind="runs",
    )
    assert selected == str(new_variant.resolve())
