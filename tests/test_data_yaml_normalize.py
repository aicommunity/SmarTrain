from __future__ import annotations

import yaml
from pathlib import Path

from smartrain.services.datasets.data_yaml_normalize import (
    normalize_data_yaml_file,
    normalize_data_yaml_mapping,
    run_normalize,
    _rewrite_split_string,
)


def test_rewrite_foreign_absolute_when_split_dirs_exist(tmp_path: Path) -> None:
    ds = tmp_path / "mars_like"
    (ds / "train" / "images").mkdir(parents=True)
    (ds / "val" / "images").mkdir(parents=True)
    foreign_train = "/home/other/pc/datasets/mars_like/train/images"
    foreign_val = "/home/other/pc/datasets/mars_like/val/images"
    assert _rewrite_split_string(str(ds), foreign_train) == "train/images"
    assert _rewrite_split_string(str(ds), foreign_val) == "val/images"


def test_normalize_mapping_drops_path_and_strips_dot_slash() -> None:
    root = "/tmp/ds1"
    raw = {"path": "/abs/ds1", "train": "./images", "val": "./images", "nc": 1, "names": ["a"]}
    out = normalize_data_yaml_mapping(root, raw)
    assert "path" not in out
    assert out["train"] == "images"
    assert out["val"] == "images"


def test_normalize_file_writes_expected(tmp_path: Path) -> None:
    ds = tmp_path / "my_ds"
    ds.mkdir()
    (ds / "data.yaml").write_text(
        "path: /nope\n"
        "train: ./train/images\n"
        "val: ./val/images\n"
        "test: ./test/images\n"
        "nc: 1\n"
        "names: [x]\n",
        encoding="utf-8",
    )
    changed, msg = normalize_data_yaml_file(str(ds), dry_run=False)
    assert changed is True
    assert msg == "updated"
    cfg = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))
    assert "path" not in cfg
    assert cfg["train"] == "train/images"


def test_run_normalize_skips_subdirs_without_yaml(tmp_path: Path) -> None:
    (tmp_path / "datasets" / "empty").mkdir(parents=True)
    (tmp_path / "datasets" / "with").mkdir(parents=True)
    (tmp_path / "datasets" / "with" / "data.yaml").write_text(
        "train: images\nval: images\nnc: 1\nnames: [a]\n", encoding="utf-8"
    )
    code = run_normalize(str(tmp_path / "datasets"), dry_run=False, as_json=False)
    assert code == 0
