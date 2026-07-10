from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from smartrain.services.datasets.yolo_image_rotate import apply_orthogonal_rotate, rotate_image_k


def test_rotate_image_k_cw_swaps_dimensions() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = rotate_image_k(img, 1)
    assert out.shape == (200, 100, 3)


def test_apply_orthogonal_rotate_bbox_cw(tmp_path: Path) -> None:
    img_path = tmp_path / "a.jpg"
    lbl_path = tmp_path / "a.txt"
    out_img = tmp_path / "out.jpg"
    out_lbl = tmp_path / "out.txt"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img_path)
    lbl_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    labels = apply_orthogonal_rotate(
        str(img_path),
        str(lbl_path),
        str(out_img),
        str(out_lbl),
        direction="cw",
    )
    assert len(labels) == 1
    assert out_img.is_file()
    with Image.open(out_img) as im:
        assert im.size == (100, 100)


def test_apply_orthogonal_rotate_polygon(tmp_path: Path) -> None:
    img_path = tmp_path / "a.jpg"
    lbl_path = tmp_path / "a.txt"
    out_img = tmp_path / "out.jpg"
    out_lbl = tmp_path / "out.txt"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img_path)
    lbl_path.write_text(
        "0 0.20 0.20 0.80 0.20 0.80 0.80 0.20 0.80\n",
        encoding="utf-8",
    )

    apply_orthogonal_rotate(
        str(img_path),
        str(lbl_path),
        str(out_img),
        str(out_lbl),
        direction="ccw",
    )
    text = out_lbl.read_text(encoding="utf-8").strip()
    parts = text.split()
    assert int(parts[0]) == 0
    assert len(parts) == 1 + 8
