from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from smartrain.dataset_balance import main as balance_main
from smartrain.datasets_json_former import main as scan_main
from smartrain.workspace_paths import deploy_workspace


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(path, format="JPEG", quality=85)


def _prepare_workspace(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_b"
    (raw / "train" / "images").mkdir(parents=True, exist_ok=True)
    (raw / "train" / "labels").mkdir(parents=True, exist_ok=True)
    _write_jpg(raw / "train" / "images" / "a.jpg")
    _write_jpg(raw / "train" / "images" / "b.jpg")
    (raw / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "train" / "labels" / "b.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 2\nnames: ['cat','dog']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])


def test_balance_creates_new_dataset_and_passport(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(["--workspace", str(tmp_path), "--dataset", "ds_b", "--strategy", "oversample", "--target", "1.5"])
    out = tmp_path / "datasets" / "ds_b_balanced"
    assert out.is_dir()
    assert (out / "dataset_passport.json").is_file()
    p = json.loads((out / "dataset_passport.json").read_text(encoding="utf-8"))
    assert p["command"] == "balance"


def test_balance_name_increment_and_class_filter(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(["--workspace", str(tmp_path), "--dataset", "ds_b", "--class", "cat"])
    balance_main(["--workspace", str(tmp_path), "--dataset", "ds_b", "--classes", "cat,dog"])
    assert (tmp_path / "datasets" / "ds_b_balanced").is_dir()
    assert (tmp_path / "datasets" / "ds_b_balanced_2").is_dir()

