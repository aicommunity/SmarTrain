from __future__ import annotations

import json
from pathlib import Path

from smartrain.services.datasets.dataset_stats import _render_after_augment


def test_render_after_augment_reads_manifest(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ds_ha"
    ds_dir.mkdir()
    manifest = {
        "strategy": "hybrid-aug",
        "class_counts_after_bbox": {"cat": 10, "dog": 2},
        "post_augment": {
            "class_counts_after_augment": {"cat": 11, "dog": 3},
            "train_bbox_sum_before_augment": 12,
            "train_bbox_sum_after_augment": 14,
        },
    }
    (ds_dir / "balance_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = _render_after_augment(["ds_ha"], {"ds_ha": str(ds_dir)})
    assert len(rows) == 1
    assert rows[0]["dataset"] == "ds_ha"
    classes = {c["class"]: c for c in rows[0]["classes"]}
    assert classes["cat"]["delta"] == 1
    assert classes["dog"]["delta"] == 1
