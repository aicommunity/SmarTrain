from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest
from PIL import Image

from smartrain.core.runtime.workspace_paths import deploy_workspace
from smartrain.services.datasets.dataset_rotate import (
    angle_to_k,
    default_output_name,
    rotate_dataset,
)
from smartrain.services.datasets.yolo_labels import read_yolo_labels
from smartrain.workflows.datasets.dataset_rotate import main as rotate_main
from smartrain.workflows.datasets.datasets_json_former import main as scan_main


def _write_flat_dataset(root: Path, name: str = "ds_r") -> Path:
    ds = root / "raw_data" / name
    images = ds / "images"
    labels = ds / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    img_path = images / "img001.jpg"
    Image.new("RGB", (100, 80), color=(30, 40, 50)).save(img_path, format="JPEG", quality=90)
    (labels / "img001.txt").write_text("0 0.50 0.50 0.40 0.30\n", encoding="utf-8")
    (ds / "data.yaml").write_text("nc: 1\nnames: ['obj']\n", encoding="utf-8")
    return ds


def test_angle_to_k_valid() -> None:
    assert angle_to_k(90) == 1
    assert angle_to_k(180) == 2
    assert angle_to_k(270) == 3


def test_angle_to_k_invalid() -> None:
    with pytest.raises(ValueError, match="Unsupported angle"):
        angle_to_k(45)


def test_default_output_name() -> None:
    assert default_output_name("StartupMarkerZeroData2", 90) == "StartupMarkerZeroData2_rot90"


def test_rotate_dataset_90_swaps_dimensions(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    src = _write_flat_dataset(tmp_path)
    scan_main(["--workspace", str(tmp_path)])
    src_root = tmp_path / "datasets" / "ds_r"
    from smartrain.services.datasets.dataset_access import iter_image_label_buckets

    buckets = list(
        iter_image_label_buckets(
            str(src_root),
            "flat",
            {"classes": {"obj": 0}, "structure": "flat"},
            dataset_name="ds_r",
            temp_root=str(tmp_path / "tmp"),
            exclude_test=False,
        )
    )
    out_dir = tmp_path / "datasets" / "ds_r_rot90"
    stats = rotate_dataset(src_root=str(src_root), out_dir=str(out_dir), k=1, buckets=buckets, no_legend=True)
    assert stats["processed"] == 1
    out_img = out_dir / "images" / "img001.jpg"
    assert out_img.is_file()
    bgr = cv2.imread(str(out_img))
    assert bgr is not None
    assert bgr.shape[0] == 100
    assert bgr.shape[1] == 80
    labels = read_yolo_labels(str(out_dir / "labels" / "img001.txt"))
    assert len(labels) == 1


def test_rotate_cli_creates_rot180_dataset(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    _write_flat_dataset(tmp_path)
    scan_main(["--workspace", str(tmp_path)])
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")
    rotate_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_r",
            "--angle",
            "180",
            "--no-legend",
        ]
    )
    out = tmp_path / "datasets" / "ds_r_rot180"
    assert out.is_dir()
    assert (out / "dataset_passport.json").is_file()
    passport = json.loads((out / "dataset_passport.json").read_text(encoding="utf-8"))
    assert passport["command"] == "rotate"
    assert passport["transformations"][0]["angle"] == 180
