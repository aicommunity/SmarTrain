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
    assert args.model == f"{mtm.MODEL_VERSION}.pt"
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


def test_train_interactive_model_manual_entry(
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
            "<manual>",  # choose manual model input
            "fork-yolo11s.pt",  # manual model
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
    assert args.model == "fork-yolo11s.pt"


def test_train_interactive_model_options_filtered_by_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
            "segment",  # task
            "",  # model (default from segment-filtered list)
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
    assert args.model.endswith("-seg.pt")
    out = capsys.readouterr().out
    assert "[INFO] Model options:" in out
    assert "-seg" in out


def test_train_interactive_selects_external_provider_from_prefixed_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deploy_workspace(str(tmp_path))
    (tmp_path / "datasets" / DATASETS_INFO_FILE).write_text(
        json.dumps({"ds_a": {"classes": {"cat": 0}}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args = _base_args(tmp_path)

    monkeypatch.setattr(
        mtm,
        "list_provider_records",
        lambda: [
            {
                "provider_id": "dr-yolo",
                "install_state": "installed",
                "repo_path": "/tmp/dr-yolo",
            }
        ],
    )
    answers = iter(
        [
            "ds_a",  # dataset
            "",  # ultralytics_yaml
            "",  # task
            "dr-yolo:yolov8n",  # model via external provider alias
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
    assert args.external_provider == "dr-yolo"
    assert args.model == "yolov8n.pt"


def test_train_main_parses_provider_prefixed_model_and_writes_external_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deploy_workspace(str(tmp_path))
    dataset_dir = tmp_path / "datasets" / "ds_a"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data.yaml").write_text("train: train/images\nval: val/images\n", encoding="utf-8")
    (tmp_path / "target").mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    def _fake_run_external_train(provider_id: str, repo_path: str, venv_path: str, **kwargs):
        captured["provider_id"] = provider_id
        captured["repo_path"] = repo_path
        captured["venv_path"] = venv_path
        captured["model"] = kwargs.get("model")
        return 0

    monkeypatch.setattr(mtm, "run_external_train", _fake_run_external_train)
    rc = mtm.main(
        [
            "--workspace",
            str(tmp_path),
            "--data",
            str(dataset_dir),
            "--target-path",
            str(tmp_path / "target"),
            "--external-repo",
            str(tmp_path / "ext-repo"),
            "--model",
            "dr-yolo:yolov8n",
        ]
    )
    assert rc == 0
    assert captured["provider_id"] == "dr-yolo"
    assert captured["model"] == "yolov8n.pt"
    marker = tmp_path / "target" / "_external_train_last.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["provider"]["id"] == "dr-yolo"
    assert payload["model"] == "yolov8n.pt"


def test_train_main_unknown_provider_in_model_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deploy_workspace(str(tmp_path))
    dataset_dir = tmp_path / "datasets" / "ds_a"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data.yaml").write_text("train: train/images\nval: val/images\n", encoding="utf-8")
    rc = mtm.main(
        [
            "--workspace",
            str(tmp_path),
            "--data",
            str(dataset_dir),
            "--target-path",
            str(tmp_path / "target"),
            "--model",
            "unknown-provider:yolov8n",
        ]
    )
    assert rc == 2
    out = capsys.readouterr().out
    assert "Unknown external provider in model ref" in out


def test_train_main_external_layout_normalized_to_train_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deploy_workspace(str(tmp_path))
    dataset_dir = tmp_path / "datasets" / "ds_a"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data.yaml").write_text("train: train/images\nval: val/images\n", encoding="utf-8")
    target_root = tmp_path / "target"
    target_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mtm, "calculate_dataset_hash", lambda _p: "abc12345")
    monkeypatch.setattr(mtm, "_build_run_name", lambda *a, **k: "run-fixed")

    def _fake_run_external_train(provider_id: str, repo_path: str, venv_path: str, **kwargs):
        run_dir = target_root / "ds_a" / "run-fixed"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "args.yaml").write_text("epochs: 1\n", encoding="utf-8")
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"fake")
        return 0

    monkeypatch.setattr(mtm, "run_external_train", _fake_run_external_train)
    rc = mtm.main(
        [
            "--workspace",
            str(tmp_path),
            "--data",
            str(dataset_dir),
            "--target-path",
            str(target_root),
            "--external-provider",
            "dr-yolo",
            "--external-repo",
            str(tmp_path / "ext-repo"),
            "--epochs",
            "1",
            "--batch",
            "2",
            "--img-size",
            "640",
        ]
    )
    assert rc == 0
    run_dir = target_root / "ds_a" / "run-fixed"
    assert (run_dir / "train" / "args.yaml").is_file()
    assert (run_dir / "train" / "weights" / "best.pt").is_file()
    assert (run_dir / "training_metadata.json").is_file()

