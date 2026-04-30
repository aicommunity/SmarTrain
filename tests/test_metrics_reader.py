from __future__ import annotations

import json
from pathlib import Path

from smartrain.confidence_recommendation import write_not_available_recommendations
from smartrain.metrics_reader import (
    flatten_metadata,
    latest_test_metrics_path,
    read_test_metrics_by_format,
    read_test_metrics_row,
    read_test_performance_by_format_artifacts,
    read_test_system_profile_by_format_artifacts,
)


def test_flatten_metadata_falls_back_to_args_yaml_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "dataset_a" / "run1"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text(
        "model: /old/machine/runs/weights/yolo26x.pt\n", encoding="utf-8"
    )
    md = {"status": {"training": {"success": True}}}

    row = flatten_metadata(md, str(run_dir))
    assert row["model"] == "yolo26x"


def test_flatten_metadata_falls_back_to_parent_dir_dataset_name(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "dataset_b" / "run2"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"status": {"training": {"success": True}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    md = {"training_info": {"model": "yolo11x"}, "status": {"training": {"success": True}}}

    row = flatten_metadata(md, str(run_dir))
    assert row["dataset_name"] == "dataset_b"


def test_flatten_metadata_uses_run_name_when_args_model_is_last(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "dataset_c" / "2026-04-24_19-13_yolo26x_300epochs-eb38791f"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text(
        "model: /other/machine/run/train/weights/last.pt\n", encoding="utf-8"
    )
    md = {"status": {"training": {"success": True}}}

    row = flatten_metadata(md, str(run_dir))
    assert row["model"] == "yolo26x"


def test_read_test_metrics_by_format_reads_manifest_and_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "dataset_d" / "run_fmt"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "test").mkdir(parents=True, exist_ok=True)
    (run_dir / "test_onnx").mkdir(parents=True, exist_ok=True)
    (run_dir / "test_metrics.csv").write_text("mAP50-95\n0.5\n", encoding="utf-8")
    (run_dir / "test_metrics_onnx.csv").write_text("mAP50-95\n0.4\n", encoding="utf-8")
    write_not_available_recommendations(model_dir=str(run_dir), split="test", reason="stub")
    write_not_available_recommendations(model_dir=str(run_dir), split="val", reason="stub")
    (run_dir / "test_artifacts_manifest.json").write_text(
        json.dumps(
            {
                "formats": {
                    "pt": {"metrics_csv": "test_metrics.csv"},
                    "onnx": {"metrics_csv": "test_metrics_onnx.csv"},
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    by_format = read_test_metrics_by_format(str(run_dir))
    assert by_format["pt"].endswith("test_metrics.csv")
    assert by_format["onnx"].endswith("test_metrics_onnx.csv")


def test_latest_test_metrics_path_pt_does_not_fallback_to_format_specific(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "dataset_e" / "run_pt_path"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "test_metrics_onnx.csv").write_text("mAP50-95\n0.4\n", encoding="utf-8")
    assert latest_test_metrics_path(str(run_dir), "pt") is None


def test_read_test_metrics_row_pt_uses_manifest_metrics_csv_when_default_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "dataset_f" / "run_manifest_pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "custom_pt_metrics.csv").write_text("mAP50-95\n0.55\n", encoding="utf-8")
    (run_dir / "test_metrics_onnx.csv").write_text("mAP50-95\n0.4\n", encoding="utf-8")
    (run_dir / "test_artifacts_manifest.json").write_text(
        json.dumps({"formats": {"pt": {"metrics_csv": "custom_pt_metrics.csv"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    row = read_test_metrics_row(str(run_dir), "pt")
    assert float(row["mAP50-95"]) == 0.55


def test_read_test_performance_by_format_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "dataset_perf" / "run_perf"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "formats": {
            "onnx": {
                "artifacts": [
                    {
                        "target_path": "models/a.onnx",
                        "performance": {"throughput_img_s": 12.3, "latency_ms": {"steady": {"p50": 9.9}}},
                    }
                ]
            }
        }
    }
    (run_dir / "test_artifacts_manifest.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    out = read_test_performance_by_format_artifacts(str(run_dir))
    assert "onnx" in out
    assert out["onnx"][0]["performance"]["throughput_img_s"] == 12.3


def test_read_test_system_profile_by_format_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "dataset_env" / "run_env"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "formats": {
            "trt": {
                "artifacts": [
                    {
                        "target_path": "models/a.trt",
                        "test_system_profile": {"runtime": {"stage": "test", "format": "trt"}},
                    }
                ]
            }
        }
    }
    (run_dir / "test_artifacts_manifest.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    out = read_test_system_profile_by_format_artifacts(str(run_dir))
    assert "trt" in out
    assert out["trt"][0]["test_system_profile"]["runtime"]["format"] == "trt"
