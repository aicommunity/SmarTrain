from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from smartrain.workflows.datasets.dataset_augment import main as augment_main
from smartrain.datasets_json_former import main as scan_main
from smartrain.workspace_paths import deploy_workspace


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(path, format="JPEG", quality=85)


def _prepare_workspace(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_a"
    (raw / "train" / "images").mkdir(parents=True, exist_ok=True)
    (raw / "train" / "labels").mkdir(parents=True, exist_ok=True)
    _write_jpg(raw / "train" / "images" / "a.jpg")
    (raw / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 1\nnames: ['cat']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])


def _prepare_workspace_two_images(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_a"
    (raw / "train" / "images").mkdir(parents=True, exist_ok=True)
    (raw / "train" / "labels").mkdir(parents=True, exist_ok=True)
    _write_jpg(raw / "train" / "images" / "a.jpg")
    _write_jpg(raw / "train" / "images" / "b.jpg")
    (raw / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "train" / "labels" / "b.txt").write_text("0 0.4 0.4 0.2 0.2\n", encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 1\nnames: ['cat']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])


def _prepare_workspace_with_valid_split(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_v"
    (raw / "train" / "images").mkdir(parents=True, exist_ok=True)
    (raw / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (raw / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (raw / "valid" / "labels").mkdir(parents=True, exist_ok=True)
    _write_jpg(raw / "train" / "images" / "a.jpg")
    _write_jpg(raw / "valid" / "images" / "v.jpg")
    (raw / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "valid" / "labels" / "v.txt").write_text("0 0.4 0.4 0.2 0.2\n", encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 1\nnames: ['cat']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])


def test_augment_creates_new_dataset_and_passport(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    augment_main(["--workspace", str(tmp_path), "--dataset", "ds_a"])
    out = tmp_path / "datasets" / "ds_a_aug"
    assert out.is_dir()
    assert (out / "dataset_passport.json").is_file()
    p = json.loads((out / "dataset_passport.json").read_text(encoding="utf-8"))
    assert p["command"] == "augment"
    aug_labels = list((out / "train" / "labels").glob("*__a-*.txt"))
    assert aug_labels
    info = json.loads((tmp_path / "datasets" / "datasets_info.json").read_text(encoding="utf-8"))
    assert "ds_a_aug" in info
    assert info["ds_a_aug"]["data_path"] == "datasets/ds_a_aug"


def test_augment_default_name_is_incremented(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    augment_main(["--workspace", str(tmp_path), "--dataset", "ds_a"])
    augment_main(["--workspace", str(tmp_path), "--dataset", "ds_a"])
    assert (tmp_path / "datasets" / "ds_a_aug").is_dir()
    assert (tmp_path / "datasets" / "ds_a_aug_2").is_dir()


def test_augment_vertical_flip_changes_y_coordinate(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    augment_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_a",
            "--enable-flip",
            "--flip",
            "vertical",
            "--flip-prob",
            "1.0",
            "--disable-center-rotate",
        ]
    )
    out = tmp_path / "datasets" / "ds_a_aug" / "train" / "labels"
    aug_file = next(out.glob("*__a-*.txt"))
    line = aug_file.read_text(encoding="utf-8").strip()
    parts = line.split()
    assert len(parts) == 5
    assert abs(float(parts[1]) - 0.5) < 1e-6
    assert abs(float(parts[2]) - 0.5) < 1e-6


def test_augment_enable_conveyor_uses_conveyor_tag(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    augment_main(
        ["--workspace", str(tmp_path), "--dataset", "ds_a", "--enable-conveyor", "--disable-center-rotate"]
    )
    out = tmp_path / "datasets" / "ds_a_aug" / "train" / "labels"
    aug_file = next(out.glob("*__a-*.txt"))
    assert "__a-c" in aug_file.stem


def test_augment_flip_prob_zero_skips_flip_variant(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    augment_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_a",
            "--enable-flip",
            "--flip-prob",
            "0.0",
            "--disable-center-rotate",
        ]
    )
    out = tmp_path / "datasets" / "ds_a_aug" / "train" / "labels"
    assert not any("__a-f" in p.stem for p in out.glob("*__a-*.txt"))


def test_augment_rotate_copies_creates_multiple_variants(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    augment_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_a",
            "--enable-center-rotate",
            "--rotate-copies",
            "2",
            "--center-rotate-deg",
            "5",
                "--min-diversity-iou",
                "1.1",
        ]
    )
    out = tmp_path / "datasets" / "ds_a_aug" / "train" / "labels"
    rot = [p for p in out.glob("*__a-*.txt") if "__a-r" in p.stem]
    assert len(rot) >= 2


def test_augment_keeps_all_original_images_without_overwrite(tmp_path: Path) -> None:
    _prepare_workspace_two_images(tmp_path)
    augment_main(["--workspace", str(tmp_path), "--dataset", "ds_a"])
    out_images = tmp_path / "datasets" / "ds_a_aug" / "train" / "images"
    originals = {p.name for p in out_images.glob("*.jpg") if "__a-" not in p.stem}
    assert originals == {"a.jpg", "b.jpg"}


def test_augment_preserves_valid_split_and_data_yaml_points_to_valid(tmp_path: Path) -> None:
    _prepare_workspace_with_valid_split(tmp_path)
    augment_main(["--workspace", str(tmp_path), "--dataset", "ds_v"])
    out = tmp_path / "datasets" / "ds_v_aug"
    assert (out / "valid" / "images" / "v.jpg").is_file()
    data_yaml = (out / "data.yaml").read_text(encoding="utf-8")
    assert "val: valid/images" in data_yaml

