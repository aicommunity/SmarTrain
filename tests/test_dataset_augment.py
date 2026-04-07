from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from smartrain.dataset_augment import main as augment_main
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


def test_augment_creates_new_dataset_and_passport(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    augment_main(["--workspace", str(tmp_path), "--dataset", "ds_a", "--multiplier", "1"])
    out = tmp_path / "datasets" / "ds_a_aug"
    assert out.is_dir()
    assert (out / "dataset_passport.json").is_file()
    p = json.loads((out / "dataset_passport.json").read_text(encoding="utf-8"))
    assert p["command"] == "augment"


def test_augment_default_name_is_incremented(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    augment_main(["--workspace", str(tmp_path), "--dataset", "ds_a", "--multiplier", "1"])
    augment_main(["--workspace", str(tmp_path), "--dataset", "ds_a", "--multiplier", "1"])
    assert (tmp_path / "datasets" / "ds_a_aug").is_dir()
    assert (tmp_path / "datasets" / "ds_a_aug_2").is_dir()

