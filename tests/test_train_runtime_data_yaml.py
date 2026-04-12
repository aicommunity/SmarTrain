from __future__ import annotations

from pathlib import Path

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

