"""export_runs_table: best epoch and train metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_export_runs_table_best_epoch_differs_from_last(tmp_path: Path) -> None:
    from smartrain.services.analyze.table import export_runs_table

    run_dir = tmp_path / "run_a"
    train = run_dir / "train-ultralytics"
    train.mkdir(parents=True)
    pd.DataFrame(
        [
            {"epoch": 1, "metrics/mAP50-95(B)": 0.10},
            {"epoch": 2, "metrics/mAP50-95(B)": 0.50},
            {"epoch": 3, "metrics/mAP50-95(B)": 0.30},
        ]
    ).to_csv(train / "results.csv", index=False)
    out = tmp_path / "runs_summary.csv"

    def flat_row(_run: str) -> dict:
        return {"run_dir": _run, "run_name": "run_a", "train_image_size": 640, "val_imgsz": 640}

    rc = export_runs_table(
        runs=[str(run_dir)],
        out_path=str(out),
        latest_test_metrics_path=lambda _r: None,
        results_csv_path=lambda r: str(Path(r) / "train-ultralytics" / "results.csv"),
        pick_map_column=lambda df: "metrics/mAP50-95(B)" if "metrics/mAP50-95(B)" in df.columns else None,
        flat_row_for_run=flat_row,
    )
    assert rc == 0
    df = pd.read_csv(out)
    assert int(df["train_last_epoch"].iloc[0]) == 3
    assert float(df["train_last_metrics/mAP50-95(B)"].iloc[0]) == 0.30
    assert int(df["train_best_epoch"].iloc[0]) == 2
    assert float(df["train_best_metrics/mAP50-95(B)"].iloc[0]) == 0.50
