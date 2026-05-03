from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from smartrain import model_training_module as mtm


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
    out = mtm._build_runtime_data_yaml(str(ds), str(run_dir), stage="train")

    with open(out, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["path"] == str(ds)
    assert cfg["train"] == "train/images"
    assert cfg["val"] == "val/images"
    assert cfg["nc"] == 2
    assert cfg["names"] == ["edge", "tear"]


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
    out = mtm._build_runtime_data_yaml(str(ds), str(run_dir), stage="train")

    with open(out, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["path"] == str(ds)
    assert cfg["train"] == "images"
    assert cfg["val"] == "images"
    assert cfg["test"] == "images"


def test_train_yolo_builds_runtime_yaml_under_run_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ds = tmp_path / "datasets" / "ds"
    ds.mkdir(parents=True)
    (ds / "data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\n", encoding="utf-8")

    target_dir = tmp_path / "runs"
    target_dir.mkdir(parents=True)
    called: dict[str, str] = {}

    monkeypatch.setattr(mtm, "_normalize_model_spec", lambda *args, **kwargs: "yolo11n.pt")
    monkeypatch.setattr(mtm, "calculate_dataset_hash", lambda *_args, **_kwargs: "hash")
    monkeypatch.setattr(mtm, "_build_run_name", lambda *args, **kwargs: "run-id")

    def _fake_build_runtime_yaml(dataset_path: str, run_dir: str, *, stage: str) -> str:
        called["dataset_path"] = dataset_path
        called["run_dir"] = run_dir
        called["stage"] = stage
        return "/tmp/runtime_data_train.yaml"

    monkeypatch.setattr(mtm, "_build_runtime_data_yaml", _fake_build_runtime_yaml)

    def _stop_after_finalize(*_args, **_kwargs):
        raise RuntimeError("stop-after-runtime-yaml")

    monkeypatch.setattr(mtm, "_finalize_train_kwargs", _stop_after_finalize)

    with pytest.raises(RuntimeError, match="stop-after-runtime-yaml"):
        mtm.train_yolo(str(ds), str(target_dir), non_interactive=True)

    expected_run_dir = target_dir / ds.name / "run-id"
    assert called["dataset_path"] == str(ds)
    assert called["run_dir"] == str(expected_run_dir)
    assert called["stage"] == "train"

