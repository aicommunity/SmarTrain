from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from smartrain import model_training_module as mtm
from smartrain.workspace_paths import DATASETS_INFO_FILE, deploy_workspace


def _base_args(workspace: Path) -> argparse.Namespace:
    return argparse.Namespace(
        workspace=str(workspace),
        config=None,
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
    monkeypatch.setattr(mtm, "_prompt_input", lambda *a, **k: next(answers))

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
        ]
    )
    monkeypatch.setattr(mtm, "_prompt_input", lambda *a, **k: next(answers))
    assert mtm._run_interactive_train_setup(args) is True

    out = capsys.readouterr().out
    assert "[INFO] Доступные датасеты:" in out
    assert "  - ds_a" in out
    assert "  - ds_b" in out


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
    monkeypatch.setattr(mtm, "_prompt_input", lambda *a, **k: next(answers))

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

