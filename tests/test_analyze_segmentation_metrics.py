from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from smartrain.services.analyze.format_compare import write_format_compare_artifacts
from smartrain.tasks.metric_columns import (
    metric_agg_columns,
    metric_agg_columns_with_fallback,
    metric_fields_from_row,
    read_run_task_type,
)


def test_metric_agg_columns_segmentation() -> None:
    cols = metric_agg_columns("segmentation")
    assert "mask_mAP50-95" in cols


def test_metric_agg_columns_with_fallback_to_box() -> None:
    cols = metric_agg_columns_with_fallback("segmentation", {"mAP50-95", "Box-F1"})
    assert "mAP50-95" in cols


def test_metric_fields_from_row_segmentation() -> None:
    row = {"mask_mAP50-95": 0.42, "Mask-F1": 0.7}
    out = metric_fields_from_row(row, "segmentation")
    assert out["mask_mAP50-95"] == 0.42
    assert out["Mask-F1"] == 0.7


def test_read_run_task_type_from_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_seg"
    run_dir.mkdir()
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"task_type": "segmentation"}}),
        encoding="utf-8",
    )
    assert read_run_task_type(str(run_dir)) == "segmentation"


def test_format_compare_writes_mask_columns_for_segmentation_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds_seg" / "run_seg"
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "models" / "yolo11s-seg.pt").write_bytes(b"pt")
    (run_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {
                    "task_type": "segmentation",
                    "model": "yolo11s-seg.pt",
                    "dataset": {"name": "ds_seg"},
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "tests").mkdir()
    (run_dir / "tests" / "test_metrics_pt.csv").write_text(
        "mask_mAP50-95,mask_mAP50,Mask-F1,Mask-P,Mask-R\n0.41,0.55,0.62,0.60,0.64\n",
        encoding="utf-8",
    )
    (run_dir / "tests" / "test_artifacts_manifest.json").write_text(
        json.dumps(
            {
                "formats": {
                    "pt": {
                        "artifacts": [
                            {
                                "target_path": "models/yolo11s-seg.pt",
                                "metrics_csv": "tests/test_metrics_pt.csv",
                                "status": "ok",
                                "backend": "ultralytics",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    session_root = tmp_path / "analytics" / "session_seg"
    session_root.mkdir(parents=True)

    out = write_format_compare_artifacts(str(session_root), [str(run_dir)])
    assert out is not None
    cmp_df = pd.read_csv(session_root / "artifacts" / "format_compare" / "format_metrics_compare_test.csv")
    pt_row = cmp_df[cmp_df["format"] == "pt"].iloc[0]
    assert float(pt_row["mask_mAP50-95"]) == 0.41
    assert float(pt_row["Mask-F1"]) == 0.62
