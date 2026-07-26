from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

from smartrain.core.analyze.run_metrics_discovery import (
    recomputed_metrics_write_path,
    resolve_recomputed_metrics_csv,
)
from smartrain.services.analyze.artifact_builders import write_speed_quality_artifacts
from smartrain.services.analyze.report_labels import build_run_display_labels, infer_short_model_name
from smartrain.services.analyze.run_query import read_test_metrics_for_run
from smartrain.services.analyze.table import export_runs_table


def _write_training_metadata(run_dir: Path) -> None:
    (run_dir / "training_metadata.json").write_text(
        '{"hyperparameters": {"epochs": 1}, "system_profile": {}}',
        encoding="utf-8",
    )


def test_resolve_recomputed_metrics_csv_prefers_tests_subdir(tmp_path: Path) -> None:
    run_dir = tmp_path / "promoted"
    run_dir.mkdir()
    tests = run_dir / "tests"
    tests.mkdir()
    csv_path = tests / "test_metrics_recomputed.csv"
    pd.DataFrame([{"Class": "all", "mAP50-95": 0.42}]).to_csv(csv_path, index=False)
    assert resolve_recomputed_metrics_csv(str(run_dir)) == str(csv_path.resolve())


def test_recomputed_metrics_write_path_uses_tests_when_present(tmp_path: Path) -> None:
    run_dir = tmp_path / "promoted"
    (run_dir / "tests").mkdir(parents=True)
    assert recomputed_metrics_write_path(str(run_dir)).endswith(
        os.path.join("tests", "test_metrics_recomputed.csv")
    )


def test_infer_short_model_name_from_args_yaml(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    train = run_dir / "train-ultralytics"
    train.mkdir(parents=True)
    _write_training_metadata(run_dir)
    yaml.safe_dump({"model": "yolo11n.pt"}, (train / "args.yaml").open("w", encoding="utf-8"))
    assert infer_short_model_name(str(run_dir)) == "yolo11n"


def test_build_run_display_labels_format(tmp_path: Path) -> None:
    run_a = tmp_path / "detect_yolo11n_20260704_183140"
    train_a = run_a / "train-ultralytics"
    train_a.mkdir(parents=True)
    _write_training_metadata(run_a)
    yaml.safe_dump({"model": "yolo11n.pt"}, (train_a / "args.yaml").open("w", encoding="utf-8"))

    run_b = tmp_path / "2026-07-04_10-34_ultralytics_yolov8n_640px_400epochs_b16-c78211ca"
    train_b = run_b / "train-ultralytics"
    train_b.mkdir(parents=True)
    _write_training_metadata(run_b)
    yaml.safe_dump({"model": "yolov8n.pt"}, (train_b / "args.yaml").open("w", encoding="utf-8"))

    labels = build_run_display_labels([str(run_a), str(run_b)], build_run_record_cb=None)
    assert labels[os.path.basename(str(run_a))] == "M1 yolo11n"
    assert labels[os.path.basename(str(run_b))] == "M2 yolov8n"


def test_resolve_run_model_identity_lists_sibling_formats(tmp_path: Path) -> None:
    from smartrain.services.analyze.report_labels import resolve_run_model_identity

    run_dir = tmp_path / "release_run"
    models = run_dir / "models"
    models.mkdir(parents=True)
    stem = "detect_yolo11m_20260716_100611_640px_400epochs_b16"
    (models / f"{stem}.pt").write_bytes(b"pt")
    (models / f"{stem}.onnx").write_bytes(b"onnx")
    (run_dir / "training_metadata.json").write_text(
        '{"paths": {"best_model": "%s.pt"}, "hyperparameters": {"epochs": 1}}' % stem,
        encoding="utf-8",
    )
    identity = resolve_run_model_identity(str(run_dir))
    assert identity.weight_stem == stem
    assert identity.model_files == (f"{stem}.pt", f"{stem}.onnx")


def test_resolve_run_class_names_ordered_from_data_yaml(tmp_path: Path) -> None:
    from smartrain.services.analyze.report_labels import resolve_run_class_names

    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    data_yaml = tmp_path / "data.yaml"
    yaml.safe_dump({"names": {2: "joint", 0: "construct", 1: "digits"}}, data_yaml.open("w", encoding="utf-8"))
    pairs = resolve_run_class_names(
        str(run_dir),
        run_data_yaml_map={str(run_dir): str(data_yaml)},
    )
    assert pairs == [(0, "construct"), (1, "digits"), (2, "joint")]


def test_write_speed_quality_includes_promoted_run_with_tests_recomputed(tmp_path: Path) -> None:
    promoted = tmp_path / "models" / "detect_yolo11n_20260704_183140"
    tests = promoted / "tests"
    tests.mkdir(parents=True)
    _write_training_metadata(promoted)
    pd.DataFrame([{"Class": "construct", "mAP50-95": 0.68177}]).to_csv(
        tests / "test_metrics_recomputed.csv", index=False
    )

    run_b = tmp_path / "runs" / "2026-07-04_10-34_ultralytics_yolov8n_640px_400epochs_b16-c78211ca"
    run_b.mkdir(parents=True)
    _write_training_metadata(run_b)
    pd.DataFrame([{"Class": "all", "mAP50-95": 0.8}]).to_csv(run_b / "test_metrics.csv", index=False)

    run_c = tmp_path / "runs" / "2026-07-04_18-41_ultralytics_yolo11s_640px_400epochs_b16-d3bc56f2"
    run_c.mkdir(parents=True)
    _write_training_metadata(run_c)
    pd.DataFrame([{"Class": "all", "mAP50-95": 0.78}]).to_csv(run_c / "test_metrics.csv", index=False)

    bench_csv = tmp_path / "benchmark.csv"
    pd.DataFrame(
        [
            {
                "model": promoted.name,
                "run_dir": str(promoted),
                "run_name": promoted.name,
                "avg_inference_ms_per_frame": 69.0,
                "benchmark_status": "ok",
            },
            {
                "model": run_b.name,
                "run_dir": str(run_b),
                "run_name": run_b.name,
                "avg_inference_ms_per_frame": 49.5,
                "benchmark_status": "ok",
            },
            {
                "model": run_c.name,
                "run_dir": str(run_c),
                "run_name": run_c.name,
                "avg_inference_ms_per_frame": 126.9,
                "benchmark_status": "ok",
            },
        ]
    ).to_csv(bench_csv, index=False)

    labels = {
        str(promoted.resolve()): "M1 yolo11n",
        promoted.name: "M1 yolo11n",
        str(run_b.resolve()): "M2 yolov8n",
        run_b.name: "M2 yolov8n",
        str(run_c.resolve()): "M3 yolo11s",
        run_c.name: "M3 yolo11s",
    }
    out = write_speed_quality_artifacts(
        session_root=str(tmp_path),
        inference_csv=str(bench_csv),
        requested_runs=[str(promoted), str(run_b), str(run_c)],
        metric_sources_payload=None,
        scatter_x="avg_inference_ms_per_frame",
        scatter_y="mAP50-95",
        run_data_yaml_map={},
        read_test_metrics_for_run=read_test_metrics_for_run,
        display_labels=labels,
    )
    assert out is not None
    sq = pd.read_csv(tmp_path / "artifacts" / "speed_quality" / "speed_quality.csv")
    assert len(sq) == 3
    assert set(sq["model"].tolist()) == {"M1 yolo11n", "M2 yolov8n", "M3 yolo11s"}


def test_export_runs_table_reads_tests_recomputed(tmp_path: Path) -> None:
    run_dir = tmp_path / "detect_yolo11n_20260704_183140"
    tests = run_dir / "tests"
    tests.mkdir(parents=True)
    _write_training_metadata(run_dir)
    pd.DataFrame([{"Class": "construct", "mAP50-95": 0.55, "Box-F1": 0.9}]).to_csv(
        tests / "test_metrics_recomputed.csv", index=False
    )

    def _flat_row(rd: str) -> dict:
        return {
            "run_dir": rd,
            "run_name": os.path.basename(rd),
            "model": os.path.basename(rd),
            "dataset_name": "ds",
        }

    out_csv = tmp_path / "runs_summary.csv"
    rc = export_runs_table(
        runs=[str(run_dir)],
        out_path=str(out_csv),
        latest_test_metrics_path=lambda _rd: None,
        results_csv_path=lambda _rd: None,
        pick_map_column=lambda _df: None,
        flat_row_for_run=_flat_row,
    )
    assert rc == 0
    df = pd.read_csv(out_csv)
    assert float(df.iloc[0]["test_mAP50-95"]) == 0.55
