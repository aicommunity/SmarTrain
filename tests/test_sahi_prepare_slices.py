"""Smoke test for SAHI prepare-slices dataset prep."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace
from smartrain.workflows.datasets.datasets_json_former import main as scan_main
from smartrain.workflows.inference.sahi_cli import main as sahi_main


def _write_jpg(path: Path, size: tuple[int, int] = (800, 600)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(40, 80, 120)).save(path, format="JPEG", quality=85)


def test_sahi_prepare_slices_smoke(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_sahi"
    (raw / "train" / "images").mkdir(parents=True)
    (raw / "train" / "labels").mkdir(parents=True)
    (raw / "val" / "images").mkdir(parents=True)
    (raw / "val" / "labels").mkdir(parents=True)
    _write_jpg(raw / "train" / "images" / "a.jpg")
    _write_jpg(raw / "train" / "images" / "b.jpg")
    # bbox near center so it survives at least one 640 slice on 800x600
    (raw / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (raw / "train" / "labels" / "b.txt").write_text("0 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    _write_jpg(raw / "val" / "images" / "v.jpg", size=(640, 640))
    (raw / "val" / "labels" / "v.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 1\nnames: ['obj']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])

    sahi_main(
        [
            "prepare-slices",
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_sahi",
            "--slice-h",
            "640",
            "--slice-w",
            "640",
            "--overlap-h",
            "0.2",
            "--overlap-w",
            "0.2",
        ]
    )
    out = tmp_path / "datasets" / "ds_sahi_sahi_slices"
    assert out.is_dir()
    assert (out / "data.yaml").is_file()
    train_imgs = list((out / "train" / "images").glob("*.jpg"))
    train_lbls = list((out / "train" / "labels").glob("*.txt"))
    assert len(train_imgs) >= 2
    assert len(train_lbls) == len(train_imgs)
    # YOLO label lines: class + 4 floats
    sample = train_lbls[0].read_text(encoding="utf-8").strip().splitlines()
    for line in sample:
        parts = line.split()
        assert len(parts) == 5
        assert all(0.0 <= float(x) <= 1.0 for x in parts[1:])
