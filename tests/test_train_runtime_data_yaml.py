from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from smartrain.core.runtime.run_artifacts import ensure_run_layout, run_tmp_dir
from smartrain.services.training.train_runtime_data_yaml_service import (
    build_runtime_data_yaml,
    materialize_ultralytics_data_yaml,
)
from smartrain.services.training.train_yolo_execution_service import train_yolo
from smartrain.services.training.train_yolo_hooks import build_train_yolo_hooks


def _touch_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xd9")


def test_runtime_data_yaml_rebinds_paths_to_selected_dataset(tmp_path: Path) -> None:
    ds = tmp_path / "datasets" / "291124"
    _touch_jpg(ds / "train" / "images" / "a.jpg")
    _touch_jpg(ds / "val" / "images" / "b.jpg")
    (ds / "data.yaml").write_text(
        "\n".join(
            [
                "names: [edge, tear]",
                "nc: 2",
                "train: '/home/rvestnikov/Documents/mars/datasets/291124/train/images'",
                "val: '/home/rvestnikov/Documents/mars/datasets/291124/val/images'",
            ]
        ),
        encoding="utf-8",
    )

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = build_runtime_data_yaml(
        str(ds),
        str(run_dir),
        stage="train",
        ensure_run_layout_cb=ensure_run_layout,
        run_tmp_dir_cb=run_tmp_dir,
        workspace_root=str(tmp_path),
    )

    with open(out, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["path"] == "datasets/291124"
    assert "\\" not in cfg["path"]
    assert cfg["train"] == "train/images"
    assert cfg["val"] == "val/images"
    assert cfg["nc"] == 2
    assert cfg["names"] == ["edge", "tear"]


def test_runtime_data_yaml_external_dataset_keeps_abs(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside_ds"
    _touch_jpg(outside / "train" / "images" / "a.jpg")
    (outside / "data.yaml").write_text(
        "train: train/images\nval: train/images\nnc: 1\nnames: [a]\n",
        encoding="utf-8",
    )
    run_dir = ws / "runs" / "r1"
    run_dir.mkdir(parents=True)
    out = build_runtime_data_yaml(
        str(outside),
        str(run_dir),
        stage="train",
        ensure_run_layout_cb=ensure_run_layout,
        run_tmp_dir_cb=run_tmp_dir,
        workspace_root=str(ws),
    )
    cfg = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
    assert Path(cfg["path"]).resolve() == outside.resolve()


def test_materialize_ultralytics_data_yaml_resolves_abs(tmp_path: Path) -> None:
    ds = tmp_path / "datasets" / "d1"
    _touch_jpg(ds / "train" / "images" / "a.jpg")
    (ds / "data.yaml").write_text(
        "train: train/images\nval: train/images\nnc: 1\nnames: [a]\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    portable = build_runtime_data_yaml(
        str(ds),
        str(run_dir),
        stage="train",
        ensure_run_layout_cb=ensure_run_layout,
        run_tmp_dir_cb=run_tmp_dir,
        workspace_root=str(tmp_path),
    )
    portable_cfg = yaml.safe_load(Path(portable).read_text(encoding="utf-8"))
    assert portable_cfg["path"] == "datasets/d1"

    abs_yaml = materialize_ultralytics_data_yaml(portable, str(tmp_path))
    assert abs_yaml.endswith(".ultralytics.yaml")
    abs_cfg = yaml.safe_load(Path(abs_yaml).read_text(encoding="utf-8"))
    assert Path(abs_cfg["path"]).resolve() == ds.resolve()
    # portable unchanged
    assert yaml.safe_load(Path(portable).read_text(encoding="utf-8"))["path"] == "datasets/d1"


def test_runtime_data_yaml_cvat_style_shared_images_bucket(tmp_path: Path) -> None:
    """data.yaml train/val point at the same images/ tree (no train/images split dirs)."""
    ds = tmp_path / "cvat_ds"
    (ds / "images" / "sub").mkdir(parents=True, exist_ok=True)
    (ds / "labels" / "sub").mkdir(parents=True, exist_ok=True)
    _touch_jpg(ds / "images" / "sub" / "a.jpg")
    (ds / "labels" / "sub" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (ds / "data.yaml").write_text(
        "\n".join(
            [
                "train: images",
                "val: images",
                "test: images",
                "nc: 1",
                "names: [bee]",
            ]
        ),
        encoding="utf-8",
    )

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = build_runtime_data_yaml(
        str(ds),
        str(run_dir),
        stage="train",
        ensure_run_layout_cb=ensure_run_layout,
        run_tmp_dir_cb=run_tmp_dir,
    )

    with open(out, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["path"] == str(ds)
    assert cfg["train"] == "images"
    assert cfg["val"] == "images"
    assert cfg["test"] == "images"


def test_runtime_data_yaml_train_only_aug_dataset_falls_back_val_to_train(tmp_path: Path) -> None:
    """Augment output with only train/ must remain trainable (val -> train)."""
    ds = tmp_path / "datasets" / "ds_aug"
    _touch_jpg(ds / "train" / "images" / "a.jpg")
    (ds / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames: [belt_side]\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = build_runtime_data_yaml(
        str(ds),
        str(run_dir),
        stage="train",
        ensure_run_layout_cb=ensure_run_layout,
        run_tmp_dir_cb=run_tmp_dir,
        workspace_root=str(tmp_path),
    )

    with open(out, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["path"] == "datasets/ds_aug"
    assert cfg["train"] == "train/images"
    assert cfg["val"] == "train/images"
    assert cfg["test"] == "train/images"


def test_runtime_data_yaml_release_bundle_uses_tmp_without_tests(tmp_path: Path) -> None:
    from smartrain.core.runtime.run_artifacts import (
        ensure_runtime_layout_for_yaml,
        ensure_runtime_tmp_dir,
    )

    ds = tmp_path / "datasets" / "ds"
    _touch_jpg(ds / "train" / "images" / "a.jpg")
    _touch_jpg(ds / "test" / "images" / "b.jpg")
    (ds / "data.yaml").write_text(
        "train: train/images\nval: train/images\ntest: test/images\nnc: 1\nnames: [obj]\n",
        encoding="utf-8",
    )
    release_dir = tmp_path / "models" / "ds" / "2026-07-14_20-42_ultralytics_yolo11s_640px_400epochs_b16-b1ef93cc"
    release_dir.mkdir(parents=True, exist_ok=True)
    pt = release_dir / "detect_yolo11s_20260714_204230_640px_400epochs_b16.pt"
    pt.write_bytes(b"pt")
    (release_dir / f"{pt.stem}.json").write_text(
        '{"artifacts": {"release_dir": "x", "model_path": "y"}, "source": {"source_run": "z"}}',
        encoding="utf-8",
    )

    out = build_runtime_data_yaml(
        str(ds),
        str(release_dir),
        stage="test",
        ensure_run_layout_cb=ensure_runtime_layout_for_yaml,
        run_tmp_dir_cb=ensure_runtime_tmp_dir,
        workspace_root=str(tmp_path),
    )
    assert Path(out).is_file()
    assert Path(out).parent == release_dir / "tmp"
    assert not (release_dir / "tests").exists()


def test_resume_pt_test_runner_passes_runtime_yaml_callbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from smartrain.services.training import train_resume_pt_test_runner as mod

    ds = tmp_path / "datasets" / "ds"
    _touch_jpg(ds / "train" / "images" / "a.jpg")
    _touch_jpg(ds / "test" / "images" / "b.jpg")
    (ds / "data.yaml").write_text(
        "train: train/images\nval: train/images\ntest: test/images\nnc: 1\nnames: [obj]\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "models" / "r1.pt").write_bytes(b"pt")
    (run_dir / "training_metadata.json").write_text(
        '{"training_info": {"task_type": "detection", "provider": {"id": "ultralytics"}}}',
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_build(dataset_path, model_dir, *, stage, ensure_run_layout_cb, run_tmp_dir_cb, workspace_root=None):
        captured["stage"] = stage
        captured["has_layout_cb"] = callable(ensure_run_layout_cb)
        captured["has_tmp_cb"] = callable(run_tmp_dir_cb)
        captured["workspace_root"] = workspace_root
        ensure_run_layout_cb(model_dir)
        tmp = Path(run_tmp_dir_cb(model_dir))
        out = tmp / f"_runtime_data_{stage}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("path: datasets/ds\n", encoding="utf-8")
        return str(out)

    class _Res:
        success = True
        error = None
        inference = {}
        test_start_time = None
        test_end_time = None

    monkeypatch.setattr(mod, "build_runtime_data_yaml", _fake_build)
    monkeypatch.setattr(mod, "materialize_ultralytics_data_yaml", lambda p, _w: p)
    monkeypatch.setattr(mod, "run_ultralytics_backend", lambda **_k: _Res())
    monkeypatch.setattr(mod, "resolve_task_context", lambda *_a, **_k: type("C", (), {"task_type": "detection"})())
    monkeypatch.setattr(mod, "ensure_matplotlib_training_runtime", lambda **_k: type("R", (), {"as_dict": lambda self: {}})())

    mod.resume_ultralytics_pt_test_runner(str(run_dir), str(ds), non_interactive=True)
    assert captured["stage"] == "test"
    assert captured["has_layout_cb"] is True
    assert captured["has_tmp_cb"] is True


def test_train_yolo_builds_runtime_yaml_under_run_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ds = tmp_path / "datasets" / "ds"
    ds.mkdir(parents=True)
    (ds / "data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\n", encoding="utf-8")

    target_dir = tmp_path / "runs"
    target_dir.mkdir(parents=True)
    called: dict[str, str] = {}

    monkeypatch.setattr(
        "smartrain.services.training.train_model_resolution_service.normalize_model_spec",
        lambda *args, **kwargs: "yolo11n.pt",
    )
    monkeypatch.setattr(
        "smartrain.services.training.train_yolo_execution_service.calculate_dataset_hash",
        lambda *_args, **_kwargs: "hash",
    )
    monkeypatch.setattr(
        "smartrain.services.training.train_yolo_execution_service.build_run_name",
        lambda *args, **kwargs: "run-id",
    )

    def _fake_build_runtime_yaml(dataset_path: str, run_dir: str, *, stage: str, **_kwargs: object) -> str:
        called["dataset_path"] = dataset_path
        called["run_dir"] = run_dir
        called["stage"] = stage
        return "/tmp/runtime_data_train.yaml"

    materialized: dict[str, str] = {}

    def _fake_materialize(portable: str, workspace_root: str) -> str:
        materialized["portable"] = portable
        materialized["workspace_root"] = workspace_root
        return "/tmp/runtime_data_train.ultralytics.yaml"

    monkeypatch.setattr(
        "smartrain.services.training.train_yolo_execution_service.build_runtime_data_yaml",
        _fake_build_runtime_yaml,
    )
    monkeypatch.setattr(
        "smartrain.services.training.train_yolo_execution_service.materialize_ultralytics_data_yaml",
        _fake_materialize,
    )

    def _stop_after_finalize(*_args, **_kwargs):
        raise RuntimeError("stop-after-runtime-yaml")

    monkeypatch.setattr(
        "smartrain.services.training.train_yolo_execution_service.finalize_train_kwargs",
        _stop_after_finalize,
    )

    with pytest.raises(RuntimeError, match="stop-after-runtime-yaml"):
        train_yolo(
            str(ds),
            str(target_dir),
            non_interactive=True,
            workspace_root=str(tmp_path),
            hooks=build_train_yolo_hooks(),
        )

    expected_run_dir = target_dir / ds.name / "run-id"
    assert called["dataset_path"] == str(ds)
    assert called["run_dir"] == str(expected_run_dir)
    assert called["stage"] == "train"
    assert materialized.get("portable") == "/tmp/runtime_data_train.yaml"
    assert materialized.get("workspace_root") == str(tmp_path)


def test_runtime_data_yaml_accepts_data_yaml_file_path(tmp_path: Path) -> None:
    ds = tmp_path / "datasets" / "d1"
    _touch_jpg(ds / "train" / "images" / "a.jpg")
    _touch_jpg(ds / "val" / "images" / "b.jpg")
    yaml_path = ds / "data.yaml"
    yaml_path.write_text(
        "\n".join(["names: [a]", "nc: 1", "train: train/images", "val: val/images"]),
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)

    out = build_runtime_data_yaml(
        str(yaml_path),
        str(run_dir),
        stage="test",
        ensure_run_layout_cb=ensure_run_layout,
        run_tmp_dir_cb=run_tmp_dir,
        workspace_root=str(tmp_path),
    )
    with open(out, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["path"] == "datasets/d1"
    assert cfg["train"] == "train/images"
