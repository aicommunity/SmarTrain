from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from PIL import Image

from smartrain.dataset_former import main as fusion_main, prune_output_empty_label_pairs
from smartrain.workspace_paths import DATASETS_INFO_FILE, CLASS_NAMES_FILE, deploy_workspace


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(path, format="JPEG", quality=85)


def _write_split_dataset(
    root: Path,
    name: str,
    class_name: str,
    class_idx: int,
    stem: str,
) -> None:
    base = root / "datasets" / name
    img = base / "train" / "images" / f"{stem}.jpg"
    lbl = base / "train" / "labels" / f"{stem}.txt"
    _write_jpg(img)
    lbl.parent.mkdir(parents=True, exist_ok=True)
    lbl.write_text(f"{class_idx} 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def test_fusion_without_partial_requires_all_classes_in_each_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    _write_split_dataset(tmp_path, "ds_a", "cat", 0, "a1")
    _write_split_dataset(tmp_path, "ds_b", "dog", 0, "b1")

    datasets_info = {
        "ds_a": {"classes": {"cat": 0}, "structure": "split"},
        "ds_b": {"classes": {"dog": 0}, "structure": "split"},
    }
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps(datasets_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (sd / CLASS_NAMES_FILE).write_text(
        json.dumps({"cat": "cat", "dog": "dog"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--output-name",
            "merged",
            "--classes",
            "cat,dog",
        ]
    )
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "Ни один датасет не содержит все выбранные классы" in out
    assert not (tmp_path / "datasets" / "merged" / "data.yaml").is_file()


def test_fusion_include_partial_merges_disjoint_class_datasets(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    _write_split_dataset(tmp_path, "ds_a", "cat", 0, "a1")
    _write_split_dataset(tmp_path, "ds_b", "dog", 0, "b1")

    datasets_info = {
        "ds_a": {"classes": {"cat": 0}, "structure": "split"},
        "ds_b": {"classes": {"dog": 0}, "structure": "split"},
    }
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps(datasets_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (sd / CLASS_NAMES_FILE).write_text(
        json.dumps({"cat": "cat", "dog": "dog"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--output-name",
            "merged",
            "--classes",
            "cat,dog",
            "--include-partial-datasets",
        ]
    )

    out_root = tmp_path / "datasets" / "merged"
    assert (out_root / "data.yaml").is_file()
    yaml_text = (out_root / "data.yaml").read_text(encoding="utf-8")
    assert "cat" in yaml_text and "dog" in yaml_text
    imgs = list(out_root.glob("*/images/*.jpg"))
    assert len(imgs) >= 2


def test_fusion_default_output_name_is_timestamp_merged(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    _write_split_dataset(tmp_path, "ds_a", "cat", 0, "a1")
    _write_split_dataset(tmp_path, "ds_b", "dog", 0, "b1")

    datasets_info = {
        "ds_a": {"classes": {"cat": 0}, "structure": "split"},
        "ds_b": {"classes": {"dog": 0}, "structure": "split"},
    }
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps(datasets_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (sd / CLASS_NAMES_FILE).write_text(
        json.dumps({"cat": "cat", "dog": "dog"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--classes",
            "cat,dog",
            "--include-partial-datasets",
        ]
    )

    wd = tmp_path / "datasets"
    subdirs = [p for p in wd.iterdir() if p.is_dir() and p.name not in {"ds_a", "ds_b"}]
    assert len(subdirs) == 1
    name = subdirs[0].name
    assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-merged$", name)
    assert (subdirs[0] / "data.yaml").is_file()


def test_prune_output_empty_label_pairs_removes_orphans(tmp_path: Path) -> None:
    split = tmp_path / "train"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir(parents=True)
    (split / "labels" / "x.txt").write_text("\n\n", encoding="utf-8")
    _write_jpg(split / "images" / "x.jpg")

    n = prune_output_empty_label_pairs(str(tmp_path))
    assert n == 1
    assert not (split / "labels" / "x.txt").exists()
    assert not (split / "images" / "x.jpg").exists()
