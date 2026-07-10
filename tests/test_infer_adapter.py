from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.services.visualization.infer_adapter import run_inference_for_split


def test_run_inference_for_split_disables_dataset_export(monkeypatch, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    split_dir = ws / "images"
    split_dir.mkdir()
    img = split_dir / "a.jpg"
    Image.new("RGB", (32, 32)).save(img)
    weights = ws / "model.pt"
    weights.write_text("fake", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_run_inference_job(args, layout):
        captured["export_dataset"] = bool(args.export_dataset)
        captured["export_visualize"] = bool(args.export_visualize)
        captured["workspace"] = str(layout.root)
        out_dir = Path(layout.root) / "inference" / "model" / "20260101-120000-images"
        out_dir.mkdir(parents=True)
        report = {
            "images": [
                {
                    "image_path_absolute": str(img.resolve()),
                    "task_outputs": {
                        "detections": [
                            {
                                "bbox_original_xyxy": [1.0, 1.0, 10.0, 10.0],
                                "class_index": 0,
                                "confidence": 0.9,
                            }
                        ]
                    },
                }
            ]
        }
        (out_dir / "inference_results.json").write_text(json.dumps(report), encoding="utf-8")
        return 0, False

    monkeypatch.setattr(
        "smartrain.services.visualization.infer_adapter.run_inference_job",
        _fake_run_inference_job,
    )

    layout = WorkspaceLayout(root=str(ws))
    preds = run_inference_for_split(
        layout=layout,
        weights_path=weights,
        split_dir=split_dir,
        device="cpu",
        conf=0.25,
        limit=None,
    )

    assert captured["export_dataset"] is False
    assert captured["export_visualize"] is False
    assert captured["workspace"] == str(ws)
    assert str(img.resolve()) in preds
    assert len(preds[str(img.resolve())]) == 1
