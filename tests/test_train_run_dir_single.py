from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartrain.services.train_runtime_helpers import (
    build_run_name,
    recover_builtin_run_dir_after_train_error,
    run_dir_has_train_artifacts,
)
from smartrain.services.train_service import _run_builtin_train_and_eval_flow
from smartrain.services.training.train_runtime_ops import TrainRuntimeOps, build_train_runtime_ops
from smartrain.services.training.train_yolo_execution_service import train_yolo
from smartrain.services.training.train_yolo_hooks import build_train_yolo_hooks


def _make_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "datasets" / "ds_a"
    ds.mkdir(parents=True)
    (ds / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        (ds / split / "images").mkdir(parents=True)
    return ds


def _seed_train_artifacts(run_dir: Path) -> None:
    train_backend = run_dir / "train-ultralytics"
    (train_backend / "weights").mkdir(parents=True)
    (train_backend / "weights" / "best.pt").write_bytes(b"fake-weights")
    (train_backend / "args.yaml").write_text("epochs: 1\n", encoding="utf-8")


def test_train_yolo_returns_training_ok_without_name_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ds = _make_dataset(tmp_path)
    target_dir = tmp_path / "runs"
    target_dir.mkdir(parents=True)
    fixed_start = datetime(2026, 7, 11, 14, 30)

    monkeypatch.setattr(
        "smartrain.services.training.train_yolo_execution_service.calculate_dataset_hash",
        lambda *_args, **_kwargs: "abc12345",
    )
    monkeypatch.setattr(
        "smartrain.services.training.train_yolo_execution_service.build_run_name",
        lambda *_args, **_kwargs: "run-fixed",
    )
    monkeypatch.setattr(
        "smartrain.services.training.train_yolo_execution_service.datetime",
        SimpleNamespace(now=lambda: fixed_start),
    )

    class _FakeModel:
        def train(self, **_kwargs):
            run_dir = target_dir / ds.name / "run-fixed"
            _seed_train_artifacts(run_dir)

    monkeypatch.setattr(
        "smartrain.services.training.train_yolo_execution_service.YOLO",
        lambda *_args, **_kwargs: _FakeModel(),
    )

    model_dir, _start, _end, _hash, _ws, meta = train_yolo(
        str(ds),
        str(target_dir),
        non_interactive=True,
        ultralytics_cfg={"model": "yolo11n.pt", "epochs": 1, "batch": 2, "imgsz": 640},
        hooks=build_train_yolo_hooks(),
    )

    expected_run = target_dir / ds.name / "run-fixed"
    assert Path(model_dir) == expected_run
    assert meta["training_ok"] is True
    assert run_dir_has_train_artifacts(str(expected_run))


def test_recover_builtin_run_dir_finds_existing_train_artifacts(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    target_dir = tmp_path / "runs"
    fixed_start = datetime(2026, 7, 11, 14, 30)
    run_name = build_run_name("ultralytics", "yolo11n.pt", 1, 2, "abc12345", timestamp=fixed_start)
    run_dir = target_dir / ds.name / run_name
    _seed_train_artifacts(run_dir)

    recovered = recover_builtin_run_dir_after_train_error(
        target_dir=str(target_dir),
        dataset_path=str(ds),
        model_version="yolo11n.pt",
        epochs=1,
        batch=2,
        dataset_hash="abc12345",
        training_start_time=None,
    )
    assert recovered == str(run_dir)


def test_builtin_train_and_eval_uses_single_run_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ds = _make_dataset(tmp_path)
    target_dir = tmp_path / "runs"
    fixed_start = datetime(2026, 7, 11, 14, 30)
    run_name = build_run_name("ultralytics", "yolo11n.pt", 1, 2, "abc12345", timestamp=fixed_start)
    run_dir = target_dir / ds.name / run_name
    _seed_train_artifacts(run_dir)

    test_dirs: list[str] = []

    def _fake_train_yolo(**_kwargs):
        return (
            str(run_dir),
            fixed_start,
            fixed_start,
            "abc12345",
            str(tmp_path),
            {"training_ok": True, "task_type": "detection", "train_kw": {}, "mpl_runtime": {}},
        )

    def _fake_test_yolo(model_dir, *_args, **_kwargs):
        test_dirs.append(model_dir)
        tests_root = Path(model_dir) / "tests"
        tests_root.mkdir(parents=True, exist_ok=True)
        (tests_root / "test_metrics.csv").write_text("metric,value\nmap50,0.5\n", encoding="utf-8")
        return fixed_start, fixed_start, {"imgsz": 640}

    base_ops = build_train_runtime_ops()
    runtime_ops = TrainRuntimeOps(
        train_yolo=_fake_train_yolo,
        test_yolo=_fake_test_yolo,
        save_training_metadata=lambda **_kwargs: None,
        collect_system_profile=lambda *_args, **_kwargs: {},
        build_run_name=base_ops.build_run_name,
        resolve_external_eval_source=base_ops.resolve_external_eval_source,
        json_safe_train_summary=lambda *_args, **_kwargs: None,
        load_batch_from_training_metadata=base_ops.load_batch_from_training_metadata,
        run_external_train=base_ops.run_external_train,
        run_external_infer=base_ops.run_external_infer,
    )

    args = argparse.Namespace(
        non_interactive=True,
        val_batch=None,
        val_imgsz=None,
        val_conf=None,
        val_iou=None,
        conf_rec_disable=True,
        conf_rec_beta_recall=2.0,
        conf_rec_beta_precision=0.5,
        conf_rec_fallback=0.25,
    )

    _run_builtin_train_and_eval_flow(
        runtime_ops=runtime_ops,
        args=args,
        u_cfg={"task": "detect"},
        sm_opts={},
        workspace_root=str(tmp_path),
        data=str(ds),
        target_dir=str(target_dir),
        model_version="yolo11n.pt",
        epochs=1,
        batch=2,
        img_size=640,
        task_type="detection",
    )

    assert test_dirs == [str(run_dir)]
    assert len(list((target_dir / ds.name).iterdir())) == 1
    assert (run_dir / "tests" / "test_metrics.csv").is_file()


def test_builtin_train_error_recovery_does_not_create_duplicate_run_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ds = _make_dataset(tmp_path)
    target_dir = tmp_path / "runs"
    fixed_start = datetime(2026, 7, 11, 14, 30)
    run_name = build_run_name("ultralytics", "yolo11n.pt", 1, 2, "abc12345", timestamp=fixed_start)
    run_dir = target_dir / ds.name / run_name
    _seed_train_artifacts(run_dir)

    build_run_calls: list[dict] = []
    original_build_run_name = build_run_name

    def _counting_build_run_name(*args, **kwargs):
        build_run_calls.append({"args": args, "kwargs": kwargs})
        return original_build_run_name(*args, **kwargs)

    def _raise_train_yolo(**_kwargs):
        raise NameError("weights_dest")

    test_dirs: list[str] = []

    def _fake_test_yolo(model_dir, *_args, **_kwargs):
        test_dirs.append(model_dir)
        return fixed_start, fixed_start, {"imgsz": 640}

    base_ops = build_train_runtime_ops()
    runtime_ops = TrainRuntimeOps(
        train_yolo=_raise_train_yolo,
        test_yolo=_fake_test_yolo,
        save_training_metadata=lambda **_kwargs: None,
        collect_system_profile=lambda *_args, **_kwargs: {},
        build_run_name=_counting_build_run_name,
        resolve_external_eval_source=base_ops.resolve_external_eval_source,
        json_safe_train_summary=lambda *_args, **_kwargs: None,
        load_batch_from_training_metadata=base_ops.load_batch_from_training_metadata,
        run_external_train=base_ops.run_external_train,
        run_external_infer=base_ops.run_external_infer,
    )

    args = argparse.Namespace(
        non_interactive=True,
        val_batch=None,
        val_imgsz=None,
        val_conf=None,
        val_iou=None,
        conf_rec_disable=True,
        conf_rec_beta_recall=2.0,
        conf_rec_beta_precision=0.5,
        conf_rec_fallback=0.25,
    )

    _run_builtin_train_and_eval_flow(
        runtime_ops=runtime_ops,
        args=args,
        u_cfg={"task": "detect"},
        sm_opts={},
        workspace_root=str(tmp_path),
        data=str(ds),
        target_dir=str(target_dir),
        model_version="yolo11n.pt",
        epochs=1,
        batch=2,
        img_size=640,
        task_type="detection",
    )

    assert len(list((target_dir / ds.name).iterdir())) == 1
    assert test_dirs == [str(run_dir)]
    assert build_run_calls == []
