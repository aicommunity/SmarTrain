from __future__ import annotations

from pathlib import Path

from PIL import Image

from smartrain.workflows.datasets.dataset_augment import main as augment_main
from smartrain.workflows.datasets.datasets_json_former import main as scan_main
from smartrain.core.runtime.workspace_paths import deploy_workspace


def _write_jpg(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color=color).save(path, format="JPEG", quality=85)


def _prepare_two_images(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_a"
    (raw / "train" / "images").mkdir(parents=True, exist_ok=True)
    (raw / "train" / "labels").mkdir(parents=True, exist_ok=True)
    _write_jpg(raw / "train" / "images" / "a.jpg", (40, 40, 40))
    _write_jpg(raw / "train" / "images" / "b.jpg", (60, 60, 60))
    # a: there are ROI boxes (object and defect)
    (raw / "train" / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.6 0.6\n1 0.2 0.5 0.1 0.1\n",
        encoding="utf-8",
    )
    #b: empty markup -> no ROI
    (raw / "train" / "labels" / "b.txt").write_text("", encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 2\nnames: ['obj','defect']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])


def test_copy_paste_roi_mode_skips_images_without_roi(tmp_path: Path) -> None:
    _prepare_two_images(tmp_path)
    augment_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_a",
            "--enable-bbox-copy",
            "--placement-roi",
            "--copy-paste-count",
            "1",
                "--disable-center-rotate",
        ]
    )
    out_labels = tmp_path / "datasets" / "ds_a_aug" / "train" / "labels"
    # There is no ROI for b.jpg, which means augmentation is skipped for it
    assert not any(p.name.startswith("b__a-") for p in out_labels.glob("*.txt"))

