from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from smartrain.core.runtime.workspace_paths import CLASS_NAMES_FILE, DATASETS_INFO_FILE, deploy_workspace
from smartrain.services.datasets.dataset_split import main as split_main


def _write_jpg(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path, format="JPEG", quality=85)


def _write_split_bucket(
    base: Path,
    split: str,
    stems: list[str],
    class_idx: int = 0,
) -> None:
    for i, stem in enumerate(stems):
        img = base / split / "images" / f"{stem}.jpg"
        lbl = base / split / "labels" / f"{stem}.txt"
        seed = (sum(ord(ch) for ch in stem) + i * 17) % 255
        _write_jpg(img, color=(seed, (seed * 3) % 255, (seed * 7) % 255))
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text(f"{class_idx} 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def test_split_repartitions_split_structure_dataset(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    ds = sd / "my_ds"
    _write_split_bucket(ds, "train", [f"t{i}" for i in range(8)])
    _write_split_bucket(ds, "valid", [f"v{i}" for i in range(2)])
    _write_split_bucket(ds, "test", [f"e{i}" for i in range(2)])

    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps({"my_ds": {"classes": {"cat": 0}, "structure": "split"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (sd / CLASS_NAMES_FILE).write_text(json.dumps({"cat": "cat"}, ensure_ascii=False, indent=2), encoding="utf-8")

    split_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "my_ds",
            "--output-name",
            "my_ds_resplit",
            "--split-ratio",
            "1,0,0",
        ]
    )

    out = sd / "my_ds_resplit"
    assert out.is_dir()
    assert len(list((out / "train" / "images").glob("*.jpg"))) == 12
    assert len(list((out / "valid" / "images").glob("*.jpg"))) == 0
    assert len(list((out / "test" / "images").glob("*.jpg"))) == 0
    assert (out / "data.yaml").is_file()
    info = json.loads((sd / DATASETS_INFO_FILE).read_text(encoding="utf-8"))
    assert "my_ds_resplit" in info
    assert info["my_ds_resplit"]["structure"] == "split"


def test_split_flat_dataset(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    ds = sd / "flat_ds"
    for i in range(10):
        img = ds / "images" / f"img{i}.jpg"
        lbl = ds / "labels" / f"img{i}.txt"
        _write_jpg(img, color=(i * 10, 20, 30))
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps({"flat_ds": {"classes": {"cat": 0}, "structure": "flat"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (sd / CLASS_NAMES_FILE).write_text(json.dumps({"cat": "cat"}, ensure_ascii=False, indent=2), encoding="utf-8")

    split_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "flat_ds",
            "--output-name",
            "flat_split",
            "--split-ratio",
            "0.8,0.1,0.1",
        ]
    )

    out = sd / "flat_split"
    train_n = len(list((out / "train" / "images").glob("*.jpg")))
    valid_n = len(list((out / "valid" / "images").glob("*.jpg")))
    test_n = len(list((out / "test" / "images").glob("*.jpg")))
    assert train_n + valid_n + test_n == 10
    assert train_n == 8
    assert valid_n == 1
    assert test_n == 1


def test_split_dry_run(tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    ds = sd / "my_ds"
    _write_split_bucket(ds, "train", ["a", "b", "c"])

    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps({"my_ds": {"classes": {"cat": 0}, "structure": "split"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (sd / CLASS_NAMES_FILE).write_text(json.dumps({"cat": "cat"}, ensure_ascii=False, indent=2), encoding="utf-8")

    split_main(["--workspace", str(tmp_path), "--dataset", "my_ds", "--dry-run"])
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert not (sd / "my_ds_split").exists()
