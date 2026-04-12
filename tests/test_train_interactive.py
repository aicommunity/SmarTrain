from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from smartrain import cli_prompts
from smartrain import model_training_module as mtm
from smartrain.workspace_paths import DATASETS_INFO_FILE, deploy_workspace


def _patch_train_prompts(monkeypatch: pytest.MonkeyPatch, answers: Iterator[str]) -> None:
    """Dataset choice uses ``cli_prompts.prompt``; other fields use ``mtm._prompt_input``."""

    def _one(*_a, **_k):
        return next(answers)

    monkeypatch.setattr(cli_prompts, "prompt", _one)
    monkeypatch.setattr(mtm, "_prompt_input", _one)


def _base_args(workspace: Path) -> argparse.Namespace:
    return argparse.Namespace(
        workspace=str(workspace),
        config=None,
        ultralytics_yaml=None,
        data=None,
        task="detect",
        model=mtm.MODEL_VERSION,
        epochs=mtm.EPOCHS,
        batch=mtm.BATCH,
        img_size=mtm.IMG_SIZE,
        target_path=None,
        model_dir=None,
        test_only=False,
        non_interactive=False,
        val_imgsz=None,
        val_conf=None,
        val_iou=None,
        weighted_sampling=False,
        export_onnx=False,
        export_onnx_fp32=False,
        clearml=False,
        clearml_project=None,
    )


def test_train_interactive_defaults_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    (tmp_path / "datasets" / DATASETS_INFO_FILE).write_text(
        json.dumps({"ds_a": {"classes": {"cat": 0}}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args = _base_args(tmp_path)

    answers = iter(
        [
            "ds_a",  # dataset
            "",  # ultralytics_yaml
            "",  # task
            "",  # model
            "",  # epochs
            "",  # batch
            "",  # img_size
            "",  # target_path
            "",  # test_only
            "",  # val_imgsz
            "",  # val_conf
            "",  # val_iou
            "",  # weighted_sampling
            "",  # export_onnx
            "",  # export_onnx_fp32
            "",  # clearml
            "",  # non_interactive
        ]
    )
    _patch_train_prompts(monkeypatch, answers)

    assert mtm._run_interactive_train_setup(args) is True
    assert args.data == "ds_a"
    assert args.model == mtm.MODEL_VERSION
    assert args.epochs == mtm.EPOCHS
    assert args.batch == mtm.BATCH
    assert args.img_size == mtm.IMG_SIZE
    assert args.target_path == str(tmp_path / "runs")
    assert args.test_only is False


def test_train_interactive_prints_available_datasets_like_fusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    deploy_workspace(str(tmp_path))
    (tmp_path / "datasets" / DATASETS_INFO_FILE).write_text(
        json.dumps({"ds_b": {"classes": {"dog": 0}}, "ds_a": {"classes": {"cat": 0}}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args = _base_args(tmp_path)

    answers = iter(
        [
            "ds_a",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ]
    )
    _patch_train_prompts(monkeypatch, answers)
    assert mtm._run_interactive_train_setup(args) is True

    out = capsys.readouterr().out
    assert "[INFO] Options for Dataset:" in out
    assert "  1. ds_a" in out
    assert "  2. ds_b" in out


def test_train_interactive_test_only_requires_model_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deploy_workspace(str(tmp_path))
    (tmp_path / "datasets" / DATASETS_INFO_FILE).write_text(
        json.dumps({"ds_a": {"classes": {"cat": 0}}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args = _base_args(tmp_path)

    answers = iter(
        [
            "ds_a",  # dataset
            "",  # ultralytics_yaml
            "",  # task
            "",  # model
            "",  # epochs
            "",  # batch
            "",  # img_size
            "",  # target_path
            "y",  # test_only = True
            "",  # model_dir empty -> retry
            "/tmp/existing_model",  # model_dir
            "",  # val_imgsz
            "",  # val_conf
            "",  # val_iou
            "",  # weighted_sampling
            "",  # export_onnx
            "",  # export_onnx_fp32
            "",  # clearml
            "",  # non_interactive
        ]
    )
    _patch_train_prompts(monkeypatch, answers)

    assert mtm._run_interactive_train_setup(args) is True
    assert args.test_only is True
    assert args.model_dir == "/tmp/existing_model"


def test_train_main_no_args_enters_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"interactive": 0}
    args = argparse.Namespace(config=None)

    monkeypatch.setattr(mtm, "parse_args", lambda argv: args)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def _fake_setup(ns):
        called["interactive"] += 1
        ns.data = "dummy"
        return True

    monkeypatch.setattr(mtm, "_run_interactive_train_setup", _fake_setup)
    monkeypatch.setattr(mtm, "_resolve_cli_paths_with_profile", lambda *a, **k: (_ for _ in ()).throw(ValueError("stop")))

    mtm.main([])
    assert called["interactive"] == 1


def test_train_interactive_skips_prompts_for_values_from_ultralytics_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deploy_workspace(str(tmp_path))
    (tmp_path / "datasets" / DATASETS_INFO_FILE).write_text(
        json.dumps({"ds_a": {"classes": {"cat": 0}}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args = _base_args(tmp_path)

    answers = iter(
        [
            "ds_a",  # dataset
            "/tmp/ultra.yaml",  # ultralytics_yaml
            "",  # target_path
            "",  # test_only
            "",  # val_imgsz
            "",  # val_conf
            "",  # val_iou
            "",  # non_interactive
        ]
    )
    _patch_train_prompts(monkeypatch, answers)
    monkeypatch.setattr(
        mtm,
        "_load_ultralytics_yaml",
        lambda _path: {
            "model": "yolo11s.pt",
            "epochs": 7,
            "batch": 4,
            "imgsz": 512,
            "task": "segment",
            "weighted_sampling": True,
            "export_onnx": True,
            "export_onnx_half": False,
            "clearml": True,
            "clearml_project": "ProjA",
        },
    )

    assert mtm._run_interactive_train_setup(args) is True
    assert args.task == "segment"
    assert args.model == "yolo11s.pt"
    assert args.epochs == 7
    assert args.batch == 4
    assert args.img_size == 512
    assert args.weighted_sampling is True
    assert args.export_onnx is True
    assert args.export_onnx_fp32 is True
    assert args.clearml is True
    assert args.clearml_project == "ProjA"


def test_collect_available_base_runs_prioritizes_selected_dataset(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    (tmp_path / "runs" / "ds_a" / "run1" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs" / "ds_a" / "run1" / "train" / "args.yaml").write_text("epochs: 5\n", encoding="utf-8")
    (tmp_path / "runs" / "ds_b" / "run2" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs" / "ds_b" / "run2" / "train" / "args.yaml").write_text("epochs: 7\n", encoding="utf-8")
    runs = mtm._collect_available_base_runs(mtm.WorkspaceLayout(str(tmp_path)), "ds_b")
    assert len(runs) == 2
    assert runs[0]["dataset"] == "ds_b"
    assert runs[1]["dataset"] == "ds_a"


def test_train_interactive_uses_selected_base_run_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deploy_workspace(str(tmp_path))
    (tmp_path / "datasets" / DATASETS_INFO_FILE).write_text(
        json.dumps({"ds_a": {"classes": {"cat": 0}}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    base_run_dir = tmp_path / "runs" / "ds_a" / "run_base" / "train"
    base_run_dir.mkdir(parents=True, exist_ok=True)
    (base_run_dir / "args.yaml").write_text(
        "task: detect\nmodel: yolo11m.pt\nepochs: 42\nbatch: 6\nimgsz: 960\n",
        encoding="utf-8",
    )

    args = _base_args(tmp_path)
    args.task = None
    args.model = None
    args.epochs = None
    args.batch = None
    args.img_size = None
    answers = iter(
        [
            "ds_a",  # dataset
            "1",  # select base run
            "",  # ultralytics_yaml
            "",  # task default from base
            "",  # model default from base
            "",  # epochs default from base
            "",  # batch default from base
            "",  # img_size default from base
            "",  # target_path
            "",  # test_only
            "",  # val_imgsz
            "",  # val_conf
            "",  # val_iou
            "",  # weighted_sampling
            "",  # export_onnx
            "",  # export_onnx_fp32
            "",  # clearml
            "",  # non_interactive
        ]
    )
    _patch_train_prompts(monkeypatch, answers)

    assert mtm._run_interactive_train_setup(args) is True
    assert args.model == "yolo11m.pt"
    assert args.epochs == 42
    assert args.batch == 6
    assert args.img_size == 960

