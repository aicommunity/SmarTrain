from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from smartrain.services.inference_segmentation_viz import render_segment_overlay, save_inference_segment_overlays


def test_render_segment_overlay(tmp_path: Path) -> None:
    img = tmp_path / "img.jpg"
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(img)
    out = tmp_path / "out.jpg"
    segments = [
        {
            "class_name": "obj",
            "confidence": 0.9,
            "polygon_roi_xy": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        }
    ]
    path = render_segment_overlay(str(img), segments, out_path=str(out))
    assert Path(path).is_file()
    assert Path(path).stat().st_size > 0


def test_save_inference_segment_overlays(tmp_path: Path) -> None:
    img = tmp_path / "src.jpg"
    Image.new("RGB", (64, 48), color=(5, 6, 7)).save(img)
    report = tmp_path / "inference_results.json"
    report.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_path": str(img),
                        "task_outputs": {
                            "segments": [
                                {
                                    "class_name": "cat",
                                    "confidence": 0.8,
                                    "polygon_roi_xy": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    saved = save_inference_segment_overlays(str(report))
    assert len(saved) == 1
    assert Path(saved[0]).is_file()
