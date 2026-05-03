from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

import smartrain.results_analyzer as results_analyzer
from smartrain.results_analyzer import main as analyze_main
from smartrain.run_artifacts import run_test_backend_dir


def _write_run(
    root: Path,
    dataset: str,
    run_name: str,
    *,
    model: str,
    map5095: float,
    box_f1: float,
    train_ok: bool = True,
    test_ok: bool = True,
) -> Path:
    run_dir = root / "runs" / dataset / run_name
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    md = {
        "training_info": {"model": model, "dataset": {"name": dataset}, "hyperparameters": {"epochs": 2}},
        "status": {"training": {"success": train_ok}, "testing": {"success": test_ok}},
        "timestamps": {"training": {"duration_seconds": 12.0}},
    }
    (run_dir / "training_metadata.json").write_text(json.dumps(md), encoding="utf-8")
    pd.DataFrame([{"mAP50-95": map5095, "Box-F1": box_f1, "avg_inference_fps": 45.0 + map5095 * 10}]).to_csv(
        run_dir / "test_metrics.csv", index=False
    )
    pd.DataFrame(
        [
            {"epoch": 0, "metrics/mAP50-95(B)": map5095 - 0.1},
            {"epoch": 1, "metrics/mAP50-95(B)": map5095},
        ]
    ).to_csv(run_dir / "train" / "results.csv", index=False)
    return run_dir


def _run_interactive(
    tmp_path: Path,
    *,
    output_dir: Path | None = None,
    preset: str = "quality",
    data_yaml: str | None = None,
    quality_metrics: str = "mAP50-95,Box-F1",
    **extra: object,
) -> None:
    ns = argparse.Namespace(
        models_root=str(tmp_path / "runs"),
        output_dir=str(output_dir) if output_dir is not None else None,
        metric_column="metrics/mAP50-95(B)",
        workspace=str(tmp_path),
        analytics_session=None,
        preset=preset,
        quality_metrics=quality_metrics,
        data_yaml=data_yaml,
        benchmark_split="test",
        benchmark_frames=100,
        benchmark_device="cpu",
        benchmark_half=False,
        speed_metric="avg_inference_ms_per_frame",
        recompute_missing_metrics=True,
        recompute_split="test",
        filter_dataset=None,
        filter_model=None,
        filter_training_ok=None,
        filter_testing_ok=None,
    )
    for k, v in extra.items():
        setattr(ns, k, v)
    results_analyzer.cmd_interactive(ns)


def test_compare_writes_insights_and_delta(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    other = _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)
    out_csv = tmp_path / "cmp.csv"
    out_png = tmp_path / "cmp.png"
    out_insights = tmp_path / "cmp_insights.txt"
    analyze_main(
        [
            "compare",
            "--models-root",
            str(tmp_path / "runs"),
            "--baseline",
            str(baseline),
            "--others",
            str(other),
            "--out-csv",
            str(out_csv),
            "--out-png",
            str(out_png),
            "--out-insights",
            str(out_insights),
        ]
    )
    assert out_csv.is_file()
    assert out_insights.is_file()
    text = out_insights.read_text(encoding="utf-8")
    assert "Baseline:" in text
    assert "better:" in text or "worse:" in text


def test_leaderboard_builds_composite_score(tmp_path: Path) -> None:
    _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.62, box_f1=0.70)
    out_csv = tmp_path / "leaderboard.csv"
    analyze_main(
        [
            "leaderboard",
            "--models-root",
            str(tmp_path / "runs"),
            "--out-csv",
            str(out_csv),
            "--quality-metric",
            "mAP50-95",
            "--speed-metric",
            "avg_inference_fps",
        ]
    )
    assert out_csv.is_file()
    df = pd.read_csv(out_csv)
    assert "composite_score" in df.columns
    assert len(df) == 2
    assert float(df.iloc[0]["composite_score"]) >= float(df.iloc[1]["composite_score"])


def test_interactive_quality_preset_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)

    answers = iter(["1", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    out_dir = tmp_path / "out"
    _run_interactive(tmp_path, output_dir=out_dir, preset="quality", quality_metrics="mAP50-95,Box-F1")

    assert (out_dir / "compare_run_a.csv").is_file()
    assert (out_dir / "compare_run_a.png").is_file()
    assert (out_dir / "compare_run_a_insights.txt").is_file()

    charts = sorted(out_dir.glob("test_metrics_*_*.png"))
    assert charts, "quality preset should generate test metrics charts"


def test_interactive_speed_preset_calls_benchmark_and_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)

    calls: list[str] = []

    def _fake_benchmark(args):
        calls.append("benchmark")
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"model": "run_a", "avg_inference_ms_per_frame": 10.0}]).to_csv(
            args.out_csv, index=False
        )

    def _fake_plot(args):
        calls.append("plot")
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_png).write_bytes(b"fakepng")

    monkeypatch.setattr(results_analyzer, "cmd_inference_benchmark", _fake_benchmark)
    monkeypatch.setattr(results_analyzer, "cmd_inference_plot", _fake_plot)

    answers = iter(["1", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    out_dir = tmp_path / "out"
    _run_interactive(
        tmp_path,
        output_dir=out_dir,
        preset="speed",
        data_yaml=str(tmp_path / "datasets" / "ds_a" / "data.yaml"),
    )

    assert "benchmark" in calls
    assert "plot" in calls
    assert any(p.name.startswith("inference_") and p.suffix == ".csv" for p in out_dir.iterdir())
    assert any(p.name.startswith("inference_") and p.suffix == ".png" for p in out_dir.iterdir())


def test_interactive_full_preset_calls_quality_speed_and_pr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)

    calls: list[str] = []

    def _fake_benchmark(args):
        calls.append("benchmark")
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"model": "run_a", "avg_inference_ms_per_frame": 10.0}]).to_csv(
            args.out_csv, index=False
        )

    def _fake_plot(args):
        calls.append("plot")
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_png).write_bytes(b"fakepng")

    def _fake_pr(args):
        calls.append("pr")
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_png).write_bytes(b"fakepng")

    monkeypatch.setattr(results_analyzer, "cmd_inference_benchmark", _fake_benchmark)
    monkeypatch.setattr(results_analyzer, "cmd_inference_plot", _fake_plot)
    monkeypatch.setattr(results_analyzer, "cmd_pr_curves", _fake_pr)

    answers = iter(["1", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    out_dir = tmp_path / "out"
    _run_interactive(
        tmp_path,
        output_dir=out_dir,
        preset="full",
        data_yaml=str(tmp_path / "datasets" / "ds_a" / "data.yaml"),
        quality_metrics="mAP50-95,Box-F1",
    )

    assert set(calls) >= {"benchmark", "plot", "pr"}
    assert (out_dir / "compare_run_a_insights.txt").is_file()
    assert sorted(out_dir.glob("test_metrics_*_*.png")), "full preset should include quality charts"


def test_interactive_speed_without_data_yaml_skips_speed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)

    answers = iter(["1", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    out_dir = tmp_path / "out"
    _run_interactive(tmp_path, output_dir=out_dir, preset="speed")

    out = capsys.readouterr().out
    assert "speed/full selected but --data-yaml is missing" in out
    assert (out_dir / "compare_run_a.csv").is_file()


def test_interactive_filters_reduce_visible_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    _write_run(tmp_path, "ds_b", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)
    _write_run(tmp_path, "ds_a", "run_c", model="yolo11n.pt", map5095=0.50, box_f1=0.60, test_ok=False)

    answers = iter(["1", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    out_dir = tmp_path / "out"
    _run_interactive(
        tmp_path,
        output_dir=out_dir,
        preset="quality",
        quality_metrics="mAP50-95,Box-F1",
        filter_dataset="ds_a",
        filter_model="yolo11n.pt",
    )

    assert (out_dir / "compare_run_a.csv").is_file()


def test_interactive_filters_can_leave_single_run_and_fail_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    _write_run(tmp_path, "ds_a", "run_c", model="yolo11n.pt", map5095=0.50, box_f1=0.60, test_ok=False)

    answers = iter(["1", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    with pytest.raises(SystemExit) as exc_info:
        _run_interactive(
            tmp_path,
            preset="quality",
            filter_dataset="ds_a",
            filter_model="yolo11n.pt",
            filter_testing_ok=True,
        )
    assert exc_info.value.code == 1


def test_interactive_default_output_goes_to_analytics_metrics_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)

    answers = iter(["1", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    _run_interactive(
        tmp_path,
        preset="quality",
        quality_metrics="mAP50-95,Box-F1",
    )

    roots = sorted((tmp_path / "analytics" / "analyze-reports").glob("analyze_*"))
    assert roots
    out_dir = roots[-1] / "artifacts" / "compare"
    assert (out_dir / "compare_run_a.csv").is_file()
    assert (out_dir / "compare_run_a.png").is_file()
    assert (out_dir / "compare_run_a_insights.txt").is_file()


def test_compare_without_required_flags_runs_interactive_in_tty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["1", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    analyze_main(
        [
            "compare",
            "--workspace",
            str(tmp_path),
            "--models-root",
            str(tmp_path / "runs"),
        ]
    )

    roots = sorted((tmp_path / "analytics" / "analyze-reports").glob("analyze_*"))
    assert roots
    out_dir = roots[-1] / "artifacts" / "compare"
    assert (out_dir / "compare_run_a.csv").is_file()
    assert (out_dir / "compare_run_a_insights.txt").is_file()


def test_test_metrics_plot_skips_single_bar_comparison(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"mAP50-95": 0.55}]).to_csv(run_dir / "test_metrics.csv", index=False)

    analyze_main(
        [
            "test-metrics-plot",
            "--workspace",
            str(tmp_path),
            "--runs-group-dir",
            str(tmp_path / "runs" / "ds_a"),
            "--metrics",
            "mAP50-95",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    out = (captured.out or "") + (captured.err or "")
    assert "only one run with numeric value" in out
    assert not list((tmp_path / "out").glob("*.png"))


def test_analyze_all_creates_session_manifest_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)

    calls: list[str] = []

    def _fake_benchmark(args):
        calls.append("benchmark")
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"model": "run_a", "avg_inference_ms_per_frame": 10.0}]).to_csv(args.out_csv, index=False)

    def _fake_plot(args):
        calls.append("plot")
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_png).write_bytes(b"fakepng")

    def _fake_pr(args):
        calls.append("pr")
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_png).write_bytes(b"fakepng")

    monkeypatch.setattr(results_analyzer, "cmd_inference_benchmark", _fake_benchmark)
    monkeypatch.setattr(results_analyzer, "cmd_inference_plot", _fake_plot)
    monkeypatch.setattr(results_analyzer, "cmd_pr_curves", _fake_pr)

    answers = iter(
        [
            "1",  # baseline
            "2",  # others
            "full",  # profile
            str(tmp_path / "datasets" / "ds_a" / "data.yaml"),  # data yaml
        ]
    )
    monkeypatch.setattr("smartrain.results_analyzer.prompt_int", lambda *_a, **_k: int(next(answers)))
    monkeypatch.setattr("smartrain.results_analyzer.prompt_text", lambda *_a, **_k: str(next(answers)))
    monkeypatch.setattr("smartrain.results_analyzer.prompt_choice", lambda *_a, **_k: str(next(answers)))

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    analyze_main(
        [
            "all",
            "--workspace",
            str(tmp_path),
            "--models-root",
            str(tmp_path / "runs"),
            "--analytics-session",
            "session_x",
            "--no-pdf",
            "--no-odt",
        ]
    )

    session_root = tmp_path / "analytics" / "analyze-reports" / "session_x"
    assert (session_root / "session.json").is_file()
    assert (session_root / "ru" / "index.md").is_file()
    assert (session_root / "en" / "index.md").is_file()
    manifest = json.loads((session_root / "session.json").read_text(encoding="utf-8"))
    assert manifest.get("profile") == "full"
    assert "artifacts" in manifest
    assert "metric_sources" in manifest
    assert "tables" in manifest
    assert "images" in manifest
    assert "format_comparison" in manifest
    assert "artifacts/format_compare/format_metrics_compare_test.csv" in manifest.get("tables", [])
    assert (session_root / "artifacts" / "metrics" / "metric_sources.json").is_file()
    assert (session_root / "artifacts" / "table" / "system_profile_compare.csv").is_file()
    assert (session_root / "artifacts" / "format_compare" / "format_metrics_compare_test.csv").is_file()
    assert "artifacts/table/system_profile_compare.csv" in manifest.get("tables", [])
    assert set(calls) >= {"benchmark", "plot", "pr"}


def test_write_test_system_profile_compare_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds_env" / "run_env"
    (run_dir / "tests").mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {"model": "yolo11n", "dataset": {"name": "ds_env"}},
                "status": {"training": {"success": True}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "tests" / "test_artifacts_manifest.json").write_text(
        json.dumps(
            {
                "formats": {
                    "onnx": {
                        "artifacts": [
                            {
                                "target_path": "models/a.onnx",
                                "test_system_profile": {
                                    "cpu": {"model": "CPU-X", "logical_cores": 16},
                                    "ram": {"total_gb": 64.0},
                                    "gpu": {
                                        "cuda_available": True,
                                        "total_vram_gb": 24.0,
                                        "devices": [{"name": "GPU-0", "total_vram_gb": 24.0}],
                                    },
                                    "platform": {"os": "Linux", "os_release": "6.8", "python_version": "3.10"},
                                    "runtime": {"stage": "test", "format": "onnx", "backend": "onnxruntime"},
                                },
                            }
                        ]
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    out_csv = tmp_path / "test_system_profile_compare.csv"
    written = results_analyzer._write_test_system_profile_compare_csv([str(run_dir)], str(out_csv))
    assert written is not None
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert str(df.iloc[0]["format"]) == "onnx"
    assert str(df.iloc[0]["test_backend"]) == "onnxruntime"
    assert str(df.iloc[0]["test_provider"]) == "onnxruntime"


def test_leaderboard_uses_performance_fallback_for_speed_metric(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "ds_lb", "run_speed_fallback", model="yolo11n.pt", map5095=0.62, box_f1=0.70)
    pd.DataFrame([{"mAP50-95": 0.62, "Box-F1": 0.70}]).to_csv(run_dir / "test_metrics.csv", index=False)
    (run_dir / "tests").mkdir(parents=True, exist_ok=True)
    (run_dir / "tests" / "test_artifacts_manifest.json").write_text(
        json.dumps(
            {
                "formats": {
                    "onnx": {
                        "artifacts": [
                            {
                                "target_path": "models/a.onnx",
                                "performance": {"throughput_img_s": 77.7},
                                "status": "ok",
                                "backend": "onnxruntime",
                            }
                        ]
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_csv = tmp_path / "leaderboard_fallback.csv"
    analyze_main(
        [
            "leaderboard",
            "--models-root",
            str(tmp_path / "runs"),
            "--out-csv",
            str(out_csv),
            "--quality-metric",
            "mAP50-95",
            "--speed-metric",
            "avg_inference_fps",
        ]
    )
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert float(df.iloc[0]["speed_metric"]) == 77.7


def test_format_compare_marks_invalid_zero_metrics_as_issue(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "ds_inv", "run_invalid_zero", model="yolo11n.pt", map5095=0.51, box_f1=0.60)
    (run_dir / "engine_ok.engine").write_bytes(b"engine")
    (run_dir / "tests").mkdir(parents=True, exist_ok=True)
    (run_dir / "tests" / "test_metrics_engine.csv").write_text(
        "mAP50-95,mAP50,Box-F1,Box-P,Box-R\n0.0,0.0,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    (run_dir / "tests" / "test_artifacts_manifest.json").write_text(
        json.dumps(
            {
                "formats": {
                    "engine": {
                        "artifacts": [
                            {
                                "target_path": "engine_ok.engine",
                                "metrics_csv": "tests/test_metrics_engine.csv",
                                "status": "ok",
                                "backend": "tensorrt",
                            }
                        ]
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    session_root = tmp_path / "analytics" / "analyze-reports" / "session_invalid_metrics"
    session_root.mkdir(parents=True, exist_ok=True)

    out = results_analyzer._write_format_compare_artifacts(str(session_root), [str(run_dir)])
    assert out is not None
    cmp_df = pd.read_csv(session_root / "artifacts" / "format_compare" / "format_metrics_compare_test.csv")
    row_engine = cmp_df[cmp_df["format"] == "engine"].iloc[0]
    assert pd.isna(row_engine["mAP50-95"])
    issues_rel = out.get("issues_json")
    assert issues_rel
    issues_payload = json.loads((session_root / issues_rel).read_text(encoding="utf-8"))
    assert any(item.get("reason_code") == "invalid_metrics" and item.get("format") == "engine" for item in issues_payload)


def test_collect_ultralytics_test_artifacts_prefers_new_layout(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "ds_a", "run_new_layout", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    canonical = run_test_backend_dir(str(run_dir), "ultralytics")
    canonical.mkdir(parents=True, exist_ok=True)
    (canonical / "pr.csv").write_text("recall,precision\n0.5,0.6\n", encoding="utf-8")
    rows, _arts = results_analyzer._collect_ultralytics_test_artifacts(
        str(tmp_path / "analytics" / "analyze-reports" / "s"),
        [str(run_dir)],
        {"run_new_layout": "R1"},
    )
    assert rows and rows[0]["exists"] is True
    assert str(rows[0]["test_dir"]).endswith("tests/test-ultralytics")


def test_analyze_all_does_not_prompt_for_missing_metrics_and_auto_recomputes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    run_b = _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)
    pd.DataFrame([{"mAP50-95": 0.56}]).to_csv(run_b / "test_metrics.csv", index=False)

    monkeypatch.setattr(results_analyzer, "cmd_inference_benchmark", lambda _args: None)
    monkeypatch.setattr(results_analyzer, "cmd_inference_plot", lambda _args: None)
    monkeypatch.setattr(results_analyzer, "cmd_pr_curves", lambda _args: None)

    recompute_calls = {"n": 0}
    orig_cmd_tm = results_analyzer.cmd_test_metrics_plot

    def _wrapped_cmd_tm(args):
        if getattr(args, "recompute_missing_metrics", False):
            recompute_calls["n"] += 1
        return orig_cmd_tm(args)

    monkeypatch.setattr(results_analyzer, "cmd_test_metrics_plot", _wrapped_cmd_tm)

    answers = iter(
        [
            "1",  # baseline
            "2",  # others
            "full",  # profile
            str(tmp_path / "datasets" / "ds_a" / "data.yaml"),  # data yaml
        ]
    )
    monkeypatch.setattr("smartrain.results_analyzer.prompt_int", lambda *_a, **_k: int(next(answers)))
    monkeypatch.setattr("smartrain.results_analyzer.prompt_text", lambda *_a, **_k: str(next(answers)))
    monkeypatch.setattr("smartrain.results_analyzer.prompt_choice", lambda *_a, **_k: str(next(answers)))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    analyze_main(
        [
            "all",
            "--workspace",
            str(tmp_path),
            "--models-root",
            str(tmp_path / "runs"),
            "--analytics-session",
            "session_auto_recompute",
            "--no-pdf",
            "--no-odt",
        ]
    )
    out = capsys.readouterr().out
    assert "Found missing metrics. Recompute from best.pt + detected data.yaml?" not in out
    assert recompute_calls["n"] >= 1


def test_format_compare_falls_back_backend_for_legacy_pt_metrics(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "ds_a", "run_legacy_pt", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    session_root = tmp_path / "analytics" / "analyze-reports" / "session_legacy"
    session_root.mkdir(parents=True, exist_ok=True)

    out = results_analyzer._write_format_compare_artifacts(str(session_root), [str(run_dir)])
    assert out is not None
    out_csv = session_root / "artifacts" / "format_compare" / "format_metrics_compare_test.csv"
    assert out_csv.is_file()
    df = pd.read_csv(out_csv)
    row_pt = df[df["format"] == "pt"].iloc[0]
    assert row_pt["backend_status"] == "ultralytics"

    eval_csv = session_root / "artifacts" / "format_compare" / "format_eval_settings.csv"
    assert eval_csv.is_file()
    eval_df = pd.read_csv(eval_csv)
    assert "inference_source" in eval_df.columns
    assert "gt_source" in eval_df.columns
    assert "nms_profile" in eval_df.columns


def test_format_compare_omits_rows_when_format_model_missing(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "ds_a", "run_only_pt", model="yolo11n.pt", map5095=0.51, box_f1=0.60)
    session_root = tmp_path / "analytics" / "analyze-reports" / "session_no_missing_formats"
    session_root.mkdir(parents=True, exist_ok=True)

    out = results_analyzer._write_format_compare_artifacts(str(session_root), [str(run_dir)])
    assert out is not None
    out_csv = session_root / "artifacts" / "format_compare" / "format_metrics_compare_test.csv"
    df = pd.read_csv(out_csv)
    assert set(df["format"].tolist()) == {"pt"}


def test_format_compare_does_not_use_onnx_csv_as_pt_source(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds_a" / "run_no_pt_csv"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_dir / "model.onnx").write_bytes(b"onnx")
    (run_dir / "tests").mkdir(parents=True, exist_ok=True)
    (run_dir / "tests" / "test_metrics_onnx.csv").write_text(
        "mAP50-95,mAP50,Box-F1,Box-P,Box-R\n0.4000,0.5,0.6,0.7,0.8\n",
        encoding="utf-8",
    )
    (run_dir / "tests" / "test_artifacts_manifest.json").write_text(
        json.dumps(
            {
                "formats": {
                    "onnx": {
                        "artifacts": [
                            {
                                "target_path": "model.onnx",
                                "metrics_csv": "tests/test_metrics_onnx.csv",
                                "status": "ok",
                                "backend": "onnxruntime",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    session_root = tmp_path / "analytics" / "analyze-reports" / "session_no_pt_fallback"
    session_root.mkdir(parents=True, exist_ok=True)

    out = results_analyzer._write_format_compare_artifacts(str(session_root), [str(run_dir)])
    assert out is not None
    out_csv = session_root / "artifacts" / "format_compare" / "format_metrics_compare_test.csv"
    df = pd.read_csv(out_csv)
    row_onnx = df[df["format"] == "onnx"].iloc[0]
    assert float(row_onnx["mAP50-95"]) == 0.4
    if "pt" in set(df["format"].tolist()):
        row_pt = df[df["format"] == "pt"].iloc[0]
        assert pd.isna(row_pt["mAP50-95"])
    sources_json = session_root / "artifacts" / "format_compare" / "format_metrics_sources.json"
    assert sources_json.is_file()


def test_format_compare_issues_include_reason_code(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "ds_a", "run_issue_code", model="yolo11n.pt", map5095=0.51, box_f1=0.60)
    (run_dir / "model.onnx").write_bytes(b"fake")
    manifest = {
        "formats": {
            "onnx": {
                "backend": "onnxruntime",
                "status": "unavailable",
                "target_path": str(run_dir / "model.onnx"),
                "error": "[oom_gpu] CUDA out of memory during split",
            }
        }
    }
    (run_dir / "test_artifacts_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    session_root = tmp_path / "analytics" / "analyze-reports" / "session_issue_code"
    session_root.mkdir(parents=True, exist_ok=True)

    out = results_analyzer._write_format_compare_artifacts(str(session_root), [str(run_dir)])
    assert out is not None
    issues_rel = out.get("issues_json")
    assert issues_rel
    issues_payload = json.loads((session_root / issues_rel).read_text(encoding="utf-8"))
    assert isinstance(issues_payload, list) and issues_payload
    onnx_issue = next(item for item in issues_payload if item.get("format") == "onnx")
    assert onnx_issue["reason_code"] == "oom_gpu"


def test_format_compare_builds_alias_legend_for_variants(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "ds_a", "run_alias", model="yolo11n.pt", map5095=0.51, box_f1=0.60)
    (run_dir / "v1.onnx").write_bytes(b"onnx1")
    (run_dir / "v2.onnx").write_bytes(b"onnx2")
    pd.DataFrame([{"mAP50-95": 0.41, "mAP50": 0.5, "Box-F1": 0.6, "Box-P": 0.7, "Box-R": 0.8}]).to_csv(
        run_dir / "test_metrics_onnx.csv", index=False
    )
    manifest = {
        "formats": {
            "onnx": {
                "artifacts": [
                    {"target_path": "v1.onnx", "metrics_csv": "test_metrics_onnx.csv", "status": "ok", "backend": "onnxruntime"},
                    {"target_path": "v2.onnx", "metrics_csv": "test_metrics_onnx.csv", "status": "ok", "backend": "onnxruntime"},
                ]
            }
        }
    }
    (run_dir / "test_artifacts_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    session_root = tmp_path / "analytics" / "analyze-reports" / "session_aliases"
    session_root.mkdir(parents=True, exist_ok=True)

    out = results_analyzer._write_format_compare_artifacts(str(session_root), [str(run_dir)])
    assert out is not None
    alias_rel = out.get("alias_legend_csv")
    assert alias_rel
    alias_df = pd.read_csv(session_root / alias_rel)
    assert any(str(v).startswith("ONNX") for v in alias_df["alias"].tolist())
    cmp_df = pd.read_csv(session_root / "artifacts" / "format_compare" / "format_metrics_compare_test.csv")
    assert "alias" in cmp_df.columns


def test_format_compare_prefers_variants_with_metrics_and_avoids_duplicate_missing(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "ds_a", "run_dedup", model="yolo11n.pt", map5095=0.51, box_f1=0.60)
    (run_dir / "v1.onnx").write_bytes(b"onnx1")
    (run_dir / "v2.onnx").write_bytes(b"onnx2")
    pd.DataFrame([{"mAP50-95": 0.41, "mAP50": 0.5, "Box-F1": 0.6, "Box-P": 0.7, "Box-R": 0.8}]).to_csv(
        run_dir / "test_metrics_onnx.csv", index=False
    )
    manifest = {
        "formats": {
            "onnx": {
                "artifacts": [
                    {"target_path": "v1.onnx", "metrics_csv": "", "status": "ok", "backend": "onnxruntime", "error": "metrics missing"},
                    {"target_path": "v2.onnx", "metrics_csv": "test_metrics_onnx.csv", "status": "ok", "backend": "onnxruntime"},
                ]
            }
        }
    }
    (run_dir / "tests").mkdir(parents=True, exist_ok=True)
    (run_dir / "tests" / "test_artifacts_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    session_root = tmp_path / "analytics" / "analyze-reports" / "session_dedup"
    session_root.mkdir(parents=True, exist_ok=True)

    out = results_analyzer._write_format_compare_artifacts(str(session_root), [str(run_dir)])
    assert out is not None
    issues_rel = out.get("issues_json")
    if issues_rel:
        issues_payload = json.loads((session_root / issues_rel).read_text(encoding="utf-8"))
        assert not any(
            item.get("format") == "onnx"
            and item.get("split") == "test"
            and item.get("reason_code") == "missing_artifact"
            for item in issues_payload
        )


def test_format_compare_skips_missing_metrics_without_explicit_failure(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "ds_a", "run_skip_missing_no_fail", model="yolo11n.pt", map5095=0.51, box_f1=0.60)
    (run_dir / "model.onnx").write_bytes(b"onnx")
    (run_dir / "tests").mkdir(parents=True, exist_ok=True)
    (run_dir / "tests" / "test_artifacts_manifest.json").write_text(
        json.dumps(
            {
                "formats": {
                    "onnx": {
                        "artifacts": [
                            {
                                "target_path": "model.onnx",
                                "metrics_csv": "",
                                "status": "ok",
                                "backend": "onnxruntime",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    session_root = tmp_path / "analytics" / "analyze-reports" / "session_skip_missing_no_fail"
    session_root.mkdir(parents=True, exist_ok=True)

    out = results_analyzer._write_format_compare_artifacts(str(session_root), [str(run_dir)])
    assert out is not None
    cmp_df = pd.read_csv(session_root / "artifacts" / "format_compare" / "format_metrics_compare_test.csv")
    assert "onnx" not in set(cmp_df["format"].tolist())


def test_format_compare_val_skips_missing_metrics_without_failure(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "ds_a", "run_val_skip_missing", model="yolo11n.pt", map5095=0.51, box_f1=0.60)
    (run_dir / "model.onnx").write_bytes(b"onnx")
    (run_dir / "tests").mkdir(parents=True, exist_ok=True)
    (run_dir / "tests" / "test_metrics_onnx.csv").write_text(
        "mAP50-95,mAP50,Box-F1,Box-P,Box-R\n0.44,0.55,0.66,0.77,0.88\n",
        encoding="utf-8",
    )
    (run_dir / "tests" / "test_artifacts_manifest.json").write_text(
        json.dumps(
            {
                "formats": {
                    "onnx": {
                        "artifacts": [
                            {
                                "target_path": "model.onnx",
                                "metrics_csv": "tests/test_metrics_onnx.csv",
                                "status": "ok",
                                "backend": "onnxruntime",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    session_root = tmp_path / "analytics" / "analyze-reports" / "session_val_skip_missing"
    session_root.mkdir(parents=True, exist_ok=True)

    out = results_analyzer._write_format_compare_artifacts(str(session_root), [str(run_dir)])
    assert out is not None
    val_csv = session_root / "artifacts" / "format_compare" / "format_metrics_compare_val.csv"
    if val_csv.is_file():
        val_df = pd.read_csv(val_csv)
        assert "onnx" not in set(val_df["format"].tolist())


def test_format_compare_prefers_existing_target_path_variant(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "ds_a", "run_target_prefer_existing", model="yolo11n.pt", map5095=0.51, box_f1=0.60)
    (run_dir / "existing.onnx").write_bytes(b"onnx")
    (run_dir / "tests").mkdir(parents=True, exist_ok=True)
    (run_dir / "tests" / "test_metrics_onnx.csv").write_text(
        "mAP50-95,mAP50,Box-F1,Box-P,Box-R\n0.42,0.53,0.64,0.75,0.86\n",
        encoding="utf-8",
    )
    (run_dir / "tests" / "test_artifacts_manifest.json").write_text(
        json.dumps(
            {
                "formats": {
                    "onnx": {
                        "artifacts": [
                            {
                                "target_path": "missing.onnx",
                                "metrics_csv": "tests/test_metrics_onnx.csv",
                                "status": "ok",
                                "backend": "onnxruntime",
                            },
                            {
                                "target_path": "existing.onnx",
                                "metrics_csv": "tests/test_metrics_onnx.csv",
                                "status": "ok",
                                "backend": "onnxruntime",
                            },
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    session_root = tmp_path / "analytics" / "analyze-reports" / "session_target_prefer_existing"
    session_root.mkdir(parents=True, exist_ok=True)

    out = results_analyzer._write_format_compare_artifacts(str(session_root), [str(run_dir)])
    assert out is not None
    cmp_df = pd.read_csv(session_root / "artifacts" / "format_compare" / "format_metrics_compare_test.csv")
    onnx_rows = cmp_df[cmp_df["format"] == "onnx"]
    assert len(onnx_rows) == 1
    assert onnx_rows.iloc[0]["target_path"] == "existing.onnx"


def test_analyze_all_allows_single_run_without_compare_and_shows_relative_run_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_a = _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)

    monkeypatch.setattr(results_analyzer, "cmd_inference_benchmark", lambda _args: None)
    monkeypatch.setattr(results_analyzer, "cmd_inference_plot", lambda _args: None)
    monkeypatch.setattr(results_analyzer, "cmd_pr_curves", lambda _args: None)

    prompt_defaults: dict[str, str] = {}

    def _fake_prompt_text(label: str, default: str = "", **_kwargs) -> str:
        prompt_defaults[label] = default
        if label == "Other run numbers (comma-separated)":
            return ""
        if label == "Path to data.yaml (required for speed/full)":
            return str(tmp_path / "datasets" / "ds_a" / "data.yaml")
        return default

    answers = iter(["1", "full"])
    monkeypatch.setattr("smartrain.results_analyzer.prompt_int", lambda *_a, **_k: int(next(answers)))
    monkeypatch.setattr("smartrain.results_analyzer.prompt_choice", lambda *_a, **_k: str(next(answers)))
    monkeypatch.setattr("smartrain.results_analyzer.prompt_text", _fake_prompt_text)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    analyze_main(
        [
            "all",
            "--workspace",
            str(tmp_path),
            "--models-root",
            str(tmp_path / "runs"),
            "--analytics-session",
            "session_single_run",
            "--no-pdf",
            "--no-odt",
        ]
    )
    out = capsys.readouterr().out

    assert prompt_defaults.get("Other run numbers (comma-separated)") == ""
    assert "run_dir (relative to runs root)" in out
    first_row = next((line for line in out.splitlines() if line.strip().startswith("1  ")), "")
    assert "ds_a/run_a" in first_row
    assert "/runs/" not in first_row
    assert "No candidate runs selected: compare artifacts are skipped" in out

    session_root = tmp_path / "analytics" / "analyze-reports" / "session_single_run"
    manifest = json.loads((session_root / "session.json").read_text(encoding="utf-8"))
    assert manifest.get("others") == []
    compare_roles = {a.get("role") for a in manifest.get("artifacts", []) if isinstance(a, dict)}
    assert "compare_csv" not in compare_roles
    assert "compare_png" not in compare_roles
    assert "compare_insights" not in compare_roles
    assert (session_root / "ru" / "index.md").is_file()
    assert (session_root / "en" / "index.md").is_file()


def test_analyze_report_includes_images_and_tables_from_manifest(tmp_path: Path) -> None:
    from smartrain.analyze_report import write_analysis_report

    (tmp_path / "artifacts" / "speed_quality").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "compare").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "pr").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "table").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "format_compare").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": "a", "scatter_x_value": 10.0, "scatter_y_value": 0.61}]).to_csv(
        tmp_path / "artifacts" / "speed_quality" / "speed_quality.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "run_dir": "runs/ds_a/run_a",
                "run_name": "run_a",
                "test_mAP50-95": 0.61,
                "test_mAP50": 0.72,
                "test_Box-F1": 0.80,
                "test_Box-P": 0.81,
                "test_Box-R": 0.79,
            }
        ]
    ).to_csv(tmp_path / "artifacts" / "table" / "runs_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "alias": "PT1",
                "run_name": "run_a",
                "format": "pt",
                "artifact_status": "ok",
                "backend_status": "ultralytics",
                "mAP50-95": 0.61,
            }
        ]
    ).to_csv(tmp_path / "artifacts" / "format_compare" / "format_metrics_compare_test.csv", index=False)
    pd.DataFrame(
        [{"alias": "PT1", "format": "pt", "run_name": "run_a", "target_path": "models/run_a.pt"}]
    ).to_csv(tmp_path / "artifacts" / "format_compare" / "format_alias_legend.csv", index=False)
    (tmp_path / "artifacts" / "compare" / "compare_curves.png").write_bytes(b"fakepng")
    (tmp_path / "artifacts" / "pr" / "pr_all_classes.png").write_bytes(b"fakepng")

    manifest = {
        "session_name": "s1",
        "profile": "full",
        "baseline": "run_a",
        "others": ["run_b"],
        "tables": [
            "artifacts/speed_quality/speed_quality.csv",
            "artifacts/table/runs_summary.csv",
            "artifacts/format_compare/format_metrics_compare_test.csv",
        ],
        "images": ["artifacts/compare/compare_curves.png", "artifacts/pr/pr_all_classes.png"],
        "artifacts": [{"role": "compare_png", "path": "artifacts/compare/compare_curves.png"}],
        "speed_quality": {"csv": "artifacts/speed_quality/speed_quality.csv"},
        "format_comparison": {
            "test_csv": "artifacts/format_compare/format_metrics_compare_test.csv",
            "alias_legend_csv": "artifacts/format_compare/format_alias_legend.csv",
        },
        "abbreviations": {"a": "M1", "ds_a": "D1"},
        "ultralytics_test": [
            {
                "run_code": "R1",
                "csv": {"pr.csv": "artifacts/ultralytics-test/R1/pr.csv"},
                "images": ["artifacts/pr/pr_all_classes.png"],
            }
        ],
    }
    out = write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    assert "md_ru" in out and "md_en" in out
    ru_md = (tmp_path / "ru" / "index.md").read_text(encoding="utf-8")
    en_md = (tmp_path / "en" / "index.md").read_text(encoding="utf-8")
    assert "![](../artifacts/compare/compare_curves.png)" in ru_md
    assert "Executive Summary" in en_md
    assert "Conclusions and Actions" in en_md
    assert "Table 1." in en_md
    assert "Рисунок 1." in ru_md
    assert "PR-кривые (все классы)" in ru_md
    assert "Ultralytics Test Results" in en_md
    assert "Ultralytics test metrics summary" in en_md
    assert "## 1. Comparison Context" in en_md
    assert "## 2. Quality Analysis" in en_md
    assert "## 4. Model Format Comparison" in en_md
    assert "Format alias legend" in en_md
    assert "| alias | run_name | target_path |" in en_md
    assert "## 8. Executive Summary" in en_md
    assert "Datasets: D1 = ds_a" in en_md
    assert "### 6.1 Run R1" in en_md


def test_analyze_report_replaces_nan_with_dash_in_tables(tmp_path: Path) -> None:
    from smartrain.analyze_report import write_analysis_report

    (tmp_path / "artifacts" / "format_compare").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_name": "run_a",
                "format": "pt",
                "artifact_status": np.nan,
                "backend_status": np.nan,
                "mAP50-95": 0.61,
            }
        ]
    ).to_csv(tmp_path / "artifacts" / "format_compare" / "format_metrics_compare_test.csv", index=False)
    manifest = {
        "session_name": "s_nan",
        "profile": "quality",
        "baseline": "run_a",
        "others": [],
        "tables": ["artifacts/format_compare/format_metrics_compare_test.csv"],
        "images": [],
        "artifacts": [],
        "format_comparison": {"test_csv": "artifacts/format_compare/format_metrics_compare_test.csv"},
        "abbreviations": {},
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    ru_md = (tmp_path / "ru" / "index.md").read_text(encoding="utf-8").lower()
    en_md = (tmp_path / "en" / "index.md").read_text(encoding="utf-8").lower()
    assert " nan " not in ru_md
    assert " nan " not in en_md


def test_analyze_report_format_section_contains_perf_subsection(tmp_path: Path) -> None:
    from smartrain.analyze_report import write_analysis_report

    (tmp_path / "artifacts" / "format_compare").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"alias": "ONNX1", "run_name": "run_a", "split": "test", "format": "onnx", "mAP50-95": 0.5}]
    ).to_csv(tmp_path / "artifacts" / "format_compare" / "format_metrics_compare_test.csv", index=False)
    pd.DataFrame(
        [{"alias": "ONNX1", "run_name": "run_a", "split": "test", "format": "onnx", "throughput_img_s": 20.0, "latency_p50_ms": 12.0, "latency_p95_ms": 20.0}]
    ).to_csv(tmp_path / "artifacts" / "format_compare" / "format_performance_compare_test.csv", index=False)
    manifest = {
        "session_name": "s_perf",
        "profile": "quality",
        "baseline": "run_a",
        "others": [],
        "tables": ["artifacts/format_compare/format_metrics_compare_test.csv"],
        "images": [],
        "artifacts": [],
        "format_comparison": {
            "test_csv": "artifacts/format_compare/format_metrics_compare_test.csv",
            "perf_test_csv": "artifacts/format_compare/format_performance_compare_test.csv",
        },
        "abbreviations": {},
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    en_md = (tmp_path / "en" / "index.md").read_text(encoding="utf-8")
    assert "### 4.1 Quality metrics comparison" in en_md
    assert "### 4.2 Performance comparison" in en_md
    assert "Format performance comparison (test)" in en_md


def test_analyze_report_perf_section_uses_benchmark_fallback(tmp_path: Path) -> None:
    from smartrain.analyze_report import write_analysis_report

    (tmp_path / "artifacts" / "inference").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model": "run_a",
                "run_dir": "runs/ds_a/run_a",
                "device": "cpu",
                "avg_total_ms_per_frame": 10.0,
                "avg_inference_ms_per_frame": 8.0,
                "avg_total_fps": 100.0,
                "avg_inference_fps": 125.0,
            }
        ]
    ).to_csv(tmp_path / "artifacts" / "inference" / "benchmark.csv", index=False)
    manifest = {
        "session_name": "s_perf_fallback",
        "profile": "quality",
        "baseline": "run_a",
        "others": [],
        "tables": ["artifacts/inference/benchmark.csv"],
        "images": [],
        "artifacts": [],
        "format_comparison": {},
        "abbreviations": {},
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    ru_md = (tmp_path / "ru" / "index.md").read_text(encoding="utf-8")
    assert "Fallback: скорость инференса по benchmark" in ru_md


def test_analyze_report_hides_sparse_system_profile_table(tmp_path: Path) -> None:
    from smartrain.analyze_report import write_analysis_report

    (tmp_path / "artifacts" / "table").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_name": "run_a",
                "model": "yolo",
                "dataset_name": "very_long_dataset_name_a",
                "sys_cpu_model": None,
                "sys_cpu_logical_cores": None,
                "sys_ram_total_gb": None,
                "sys_gpu_0_name": None,
            }
        ]
    ).to_csv(tmp_path / "artifacts" / "table" / "system_profile_compare.csv", index=False)
    manifest = {
        "session_name": "s2",
        "profile": "quality",
        "baseline": "run_a",
        "others": [],
        "tables": ["artifacts/table/system_profile_compare.csv"],
        "images": [],
        "artifacts": [],
        "abbreviations": {"very_long_dataset_name_a": "D1"},
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    ru_md = (tmp_path / "ru" / "index.md").read_text(encoding="utf-8")
    assert "Системный профиль не показан" in ru_md
    assert "| run_name | model | dataset_name |" not in ru_md


def test_analyze_report_renders_test_system_profile_table(tmp_path: Path) -> None:
    from smartrain.analyze_report import write_analysis_report

    (tmp_path / "artifacts" / "table").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_name": "run_a",
                "model": "yolo",
                "dataset_name": "ds_a",
                "format": "onnx",
                "test_backend": "onnxruntime",
                "test_provider": "onnxruntime-session",
                "sys_cpu_model": "CPU-X",
                "sys_ram_total_gb": 64.0,
            }
        ]
    ).to_csv(tmp_path / "artifacts" / "table" / "test_system_profile_compare.csv", index=False)
    manifest = {
        "session_name": "s2_test_env",
        "profile": "quality",
        "baseline": "run_a",
        "others": [],
        "tables": ["artifacts/table/test_system_profile_compare.csv"],
        "images": [],
        "artifacts": [],
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    ru_md = (tmp_path / "ru" / "index.md").read_text(encoding="utf-8")
    assert "Сравнение окружения тестирования (железо)" in ru_md


def test_analyze_report_per_class_headers_and_human_readable_ultralytics_captions(tmp_path: Path) -> None:
    from smartrain.analyze_report import write_analysis_report

    (tmp_path / "artifacts" / "pr" / "per_class").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "ultralytics-test" / "R1").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"model": "run_a", "class_name": "aluminium", "ap": 0.9},
            {"model": "run_b", "class_name": "aluminium", "ap": 0.7},
        ]
    ).to_csv(tmp_path / "artifacts" / "pr" / "per_class" / "pr_per_class.csv", index=False)
    (tmp_path / "artifacts" / "pr" / "per_class" / "pr_class_0_aluminium.png").write_bytes(b"fakepng")
    (tmp_path / "artifacts" / "ultralytics-test" / "R1" / "BoxPR_curve.png").write_bytes(b"fakepng")
    manifest = {
        "session_name": "s3",
        "profile": "full",
        "baseline": "run_a",
        "others": ["run_b"],
        "tables": ["artifacts/pr/per_class/pr_per_class.csv"],
        "images": [
            "artifacts/pr/per_class/pr_class_0_aluminium.png",
            "artifacts/ultralytics-test/R1/BoxPR_curve.png",
        ],
        "artifacts": [],
        "abbreviations": {"run_a": "R1", "run_b": "R2"},
        "pr_per_class": {"csv": "artifacts/pr/per_class/pr_per_class.csv"},
        "ultralytics_test": [
            {
                "run_code": "R1",
                "csv": {
                    "pr.csv": "artifacts/ultralytics-test/R1/pr.csv",
                    "pr_per_class.csv": "artifacts/ultralytics-test/R1/pr_per_class.csv",
                },
                "images": ["artifacts/ultralytics-test/R1/BoxPR_curve.png"],
                "run_info": {
                    "model": "yolo26x",
                    "epochs": 300,
                    "batch_size": 4,
                    "train_image_size": 1280,
                    "val_imgsz": 1280,
                },
                "machine_info": {
                    "sys_cpu_model": "AMD EPYC",
                    "sys_cpu_logical_cores": 32,
                    "sys_ram_total_gb": 128,
                    "sys_gpu_0_name": "NVIDIA RTX",
                    "sys_gpu_0_vram_gb": 24,
                    "sys_os": "Ubuntu",
                    "sys_os_release": "22.04",
                },
            }
        ],
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    en_md = (tmp_path / "en" / "index.md").read_text(encoding="utf-8")
    assert "| class_name | best_run | best_ap | worst_run | worst_ap | ap_gap |" in en_md
    assert "Per-class PR curve: aluminium" in en_md
    assert "Precision-Recall curve" in en_md
    assert "*Figure" in en_md
    assert "BoxPR_curve.png*" not in en_md
    assert "### 6.1 Run R1" in en_md
    assert "Run config: model=yolo26x, dataset=-, epochs=300, batch=4, imgsz_train=1280, imgsz_val=1280." in en_md
    assert "Machine: CPU=AMD EPYC (32 cores), RAM=128 GB, GPU=NVIDIA RTX (24 GB), OS=Ubuntu 22.04" in en_md
    assert "Data source for PR-curve table (precision/recall across confidence thresholds): `artifacts/ultralytics-test/R1/pr.csv`" in en_md
    assert "Data source for per-class PR table: `artifacts/ultralytics-test/R1/pr_per_class.csv`" in en_md
    assert "R1: data source pr.csv" not in en_md


def test_analyze_report_hides_empty_run_machine_lines_in_ultralytics_section(tmp_path: Path) -> None:
    from smartrain.analyze_report import write_analysis_report

    (tmp_path / "artifacts" / "ultralytics-test" / "R1").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "ultralytics-test" / "R1" / "BoxPR_curve.png").write_bytes(b"fakepng")
    manifest = {
        "session_name": "s4",
        "profile": "full",
        "baseline": "run_a",
        "others": [],
        "tables": [],
        "images": ["artifacts/ultralytics-test/R1/BoxPR_curve.png"],
        "artifacts": [],
        "ultralytics_test": [
            {
                "run_code": "R1",
                "csv": {},
                "images": ["artifacts/ultralytics-test/R1/BoxPR_curve.png"],
                "run_info": {},
                "machine_info": {},
            }
        ],
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    ru_md = (tmp_path / "ru" / "index.md").read_text(encoding="utf-8")
    assert "модель=-" not in ru_md
    assert "CPU=-" not in ru_md


def test_analyze_report_confidence_tables_titles_columns_and_per_run_split(tmp_path: Path) -> None:
    from smartrain.analyze_report import write_analysis_report

    (tmp_path / "artifacts" / "confidence").mkdir(parents=True, exist_ok=True)
    rows_a = [
        {
            "run_name": "run_a",
            "split": "test",
            "objective": "A",
            "level": "global",
            "class_id": -1,
            "class_name": "all",
            "recommended_conf": 0.42,
            "target_metric": 0.81,
            "precision": 0.83,
            "recall": 0.79,
            "f1": 0.81,
            "status": "ok",
        },
        {
            "run_name": "run_a",
            "split": "test",
            "objective": "A",
            "level": "class",
            "class_id": 0,
            "class_name": "metal",
            "recommended_conf": 0.44,
            "target_metric": 0.82,
            "precision": 0.84,
            "recall": 0.80,
            "f1": 0.82,
            "status": "ok",
        },
        {
            "run_name": "run_b",
            "split": "test",
            "objective": "A",
            "level": "class",
            "class_id": 0,
            "class_name": "metal",
            "recommended_conf": 0.47,
            "target_metric": 0.79,
            "precision": 0.81,
            "recall": 0.77,
            "f1": 0.79,
            "status": "ok",
        },
    ]
    rows_b = [
        {
            "run_name": "run_a",
            "split": "test",
            "objective": "B",
            "level": "global",
            "class_id": -1,
            "class_name": "all",
            "recommended_conf": 0.31,
            "target_metric": 0.78,
            "precision": 0.75,
            "recall": 0.82,
            "f1": 0.78,
            "status": "ok",
        }
    ]
    rows_c = [
        {
            "run_name": "run_a",
            "split": "test",
            "objective": "C",
            "level": "global",
            "class_id": -1,
            "class_name": "all",
            "recommended_conf": 0.58,
            "target_metric": 0.76,
            "precision": 0.88,
            "recall": 0.67,
            "f1": 0.76,
            "status": "ok",
        }
    ]
    pd.DataFrame(rows_a).to_csv(
        tmp_path / "artifacts" / "confidence" / "confidence_recommendations_A.csv",
        index=False,
    )
    pd.DataFrame(rows_b).to_csv(
        tmp_path / "artifacts" / "confidence" / "confidence_recommendations_B.csv",
        index=False,
    )
    pd.DataFrame(rows_c).to_csv(
        tmp_path / "artifacts" / "confidence" / "confidence_recommendations_C.csv",
        index=False,
    )

    manifest = {
        "session_name": "s_conf",
        "profile": "full",
        "baseline": "run_a",
        "others": ["run_b"],
        "tables": [
            "artifacts/confidence/confidence_recommendations_A.csv",
            "artifacts/confidence/confidence_recommendations_B.csv",
            "artifacts/confidence/confidence_recommendations_C.csv",
        ],
        "images": [],
        "artifacts": [],
        "confidence_recommendations": {
            "A": "artifacts/confidence/confidence_recommendations_A.csv",
            "B": "artifacts/confidence/confidence_recommendations_B.csv",
            "C": "artifacts/confidence/confidence_recommendations_C.csv",
        },
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    ru_md = (tmp_path / "ru" / "index.md").read_text(encoding="utf-8")

    assert "Рекомендации confidence (A: максимум F1)" in ru_md
    assert "Рекомендации confidence (B: F-beta (приоритет Recall))" in ru_md
    assert "Рекомендации confidence (C: F-beta (приоритет Precision))" in ru_md

    # quality confidence tables: no objective/level/class_id/class_name
    assert "| split | recommended_conf | target_metric | precision | recall | f1 | status |" in ru_md
    assert "| objective |" not in ru_md
    assert "| level |" not in ru_md

    # per-class confidence tables split by run and class_name first
    assert "Рекомендации confidence (A: максимум F1) — run run_a" in ru_md
    assert "Рекомендации confidence (A: максимум F1) — run run_b" in ru_md
    assert "| class_name | split | class_id | recommended_conf | target_metric | precision | recall | f1 | status |" in ru_md


def test_interactive_full_auto_detects_data_yaml_from_runtime_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_a = _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)
    runtime_yaml = run_a / "_runtime_data_train.yaml"
    runtime_yaml.write_text("path: datasets/ds_a\ntrain: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")

    calls: list[str] = []

    def _fake_benchmark(args):
        calls.append("benchmark")
        assert str(args.data_yaml).endswith("_runtime_data_train.yaml")
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"model": "run_a", "avg_inference_ms_per_frame": 10.0}]).to_csv(args.out_csv, index=False)

    def _fake_plot(args):
        calls.append("plot")
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_png).write_bytes(b"fakepng")

    def _fake_pr(args):
        calls.append("pr")
        assert str(args.data_yaml).endswith("_runtime_data_train.yaml")
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_png).write_bytes(b"fakepng")

    monkeypatch.setattr(results_analyzer, "cmd_inference_benchmark", _fake_benchmark)
    monkeypatch.setattr(results_analyzer, "cmd_inference_plot", _fake_plot)
    monkeypatch.setattr(results_analyzer, "cmd_pr_curves", _fake_pr)
    answers = iter(["1", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    out_dir = tmp_path / "out"
    _run_interactive(
        tmp_path,
        output_dir=out_dir,
        preset="full",
        quality_metrics="mAP50-95,Box-F1",
    )
    assert set(calls) >= {"benchmark", "plot", "pr"}


def test_auto_detect_prints_data_yaml_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_a = _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)
    runtime_yaml = run_a / "_runtime_data_train.yaml"
    runtime_yaml.write_text("path: datasets/ds_a\ntrain: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")

    monkeypatch.setattr(results_analyzer, "cmd_inference_benchmark", lambda _args: None)
    monkeypatch.setattr(results_analyzer, "cmd_inference_plot", lambda _args: None)
    monkeypatch.setattr(results_analyzer, "cmd_pr_curves", lambda _args: None)

    answers = iter(["1", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    _run_interactive(
        tmp_path,
        output_dir=(tmp_path / "out"),
        preset="full",
        quality_metrics="mAP50-95,Box-F1",
    )
    out = capsys.readouterr().out
    assert "Auto-detected data.yaml:" in out
    assert "source:" in out


def test_auto_detect_multiple_candidates_prints_single_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_a = _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    run_b = _write_run(tmp_path, "ds_b", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)
    (run_a / "_runtime_data_train.yaml").write_text(
        "path: datasets/ds_a\ntrain: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )
    (run_b / "_runtime_data_train.yaml").write_text(
        "path: datasets/ds_b\ntrain: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(results_analyzer, "cmd_inference_benchmark", lambda _args: None)
    monkeypatch.setattr(results_analyzer, "cmd_inference_plot", lambda _args: None)
    monkeypatch.setattr(results_analyzer, "cmd_pr_curves", lambda _args: None)
    monkeypatch.setattr(
        "smartrain.results_analyzer.prompt_choice",
        lambda _label, options, default=None, **_kw: default or options[0],
    )

    answers = iter(["1", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    _run_interactive(
        tmp_path,
        output_dir=(tmp_path / "out"),
        preset="full",
        quality_metrics="mAP50-95,Box-F1",
    )
    out = capsys.readouterr().out
    assert "Multiple data.yaml candidates detected:" in out
    assert "Options for Select data.yaml" not in out


def test_test_metrics_plot_recomputes_missing_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_a = tmp_path / "runs" / "ds_a" / "run_a"
    run_b = tmp_path / "runs" / "ds_a" / "run_b"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    (run_a / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {
                    "dataset": {"name": "ds_a"},
                }
            }
        ),
        encoding="utf-8",
    )
    (run_b / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {
                    "dataset": {"name": "ds_a"},
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "datasets" / "ds_a").mkdir(parents=True, exist_ok=True)
    ((tmp_path / "datasets" / "ds_a") / "data.yaml").write_text("train: train/images\nval: val/images\ntest: test/images\n", encoding="utf-8")
    (run_a / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_b / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_a / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_b / "train" / "weights" / "best.pt").write_bytes(b"fake")
    pd.DataFrame([{"mAP50-95": 0.55}]).to_csv(run_a / "test_metrics.csv", index=False)
    pd.DataFrame([{"mAP50-95": 0.60, "Box-F1": 0.92}]).to_csv(run_b / "test_metrics.csv", index=False)

    monkeypatch.setattr(
        results_analyzer,
        "_recompute_run_test_metrics",
        lambda *_a, **_k: {"mAP50-95": 0.55, "Box-F1": 0.91},
    )
    analyze_main(
        [
            "test-metrics-plot",
            "--workspace",
            str(tmp_path),
            "--runs-group-dir",
            str(tmp_path / "runs" / "ds_a"),
            "--metrics",
            "mAP50-95",
            "Box-F1",
            "--out-dir",
            str(tmp_path / "out"),
            "--recompute-missing-metrics",
        ]
    )
    assert list((tmp_path / "out").glob("*.png"))


def test_test_metrics_plot_respects_selected_run_scope_and_writes_scope_payload(
    tmp_path: Path,
) -> None:
    run_a = tmp_path / "runs" / "ds_a" / "run_a"
    run_b = tmp_path / "runs" / "ds_a" / "run_b"
    run_c = tmp_path / "runs" / "ds_a" / "run_c"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    run_c.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"mAP50-95": 0.55}]).to_csv(run_a / "test_metrics.csv", index=False)
    pd.DataFrame([{"mAP50-95": 0.60}]).to_csv(run_b / "test_metrics.csv", index=False)
    pd.DataFrame([{"mAP50-95": 0.10}]).to_csv(run_c / "test_metrics.csv", index=False)
    out_dir = tmp_path / "out"
    sources_path = out_dir / "metric_sources.json"

    results_analyzer.cmd_test_metrics_plot(
        argparse.Namespace(
            runs_group_dir=str(tmp_path / "runs" / "ds_a"),
            selected_run_dirs=[str(run_a), str(run_b)],
            metrics=["mAP50-95"],
            out_dir=str(out_dir),
            workspace=str(tmp_path),
            recompute_missing_metrics=False,
            recompute_split="test",
            metric_sources_out=str(sources_path),
            val_batch=1,
            val_imgsz=640,
            val_half=True,
            gpu_only_val=True,
        )
    )

    payload = json.loads(sources_path.read_text(encoding="utf-8"))
    scoped_sources = payload.get("sources", {})
    assert set(scoped_sources.keys()) == {str(run_a), str(run_b)}
    assert payload.get("scope", {}).get("mode") == "selected_runs"
    assert sorted(payload.get("scope", {}).get("selected_run_dirs", [])) == sorted([str(run_a), str(run_b)])


def test_inference_and_pr_respect_selected_run_scope_in_analyze_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_a = _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    run_b = _write_run(tmp_path, "ds_a", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)

    captured: dict[str, list[str]] = {"benchmark": [], "pr": []}

    def _fake_benchmark(args):
        captured["benchmark"] = list(getattr(args, "selected_run_dirs", []) or [])
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"model": "run_a", "run_dir": str(run_a), "avg_inference_ms_per_frame": 10.0},
                {"model": "run_b", "run_dir": str(run_b), "avg_inference_ms_per_frame": 11.0},
            ]
        ).to_csv(args.out_csv, index=False)

    def _fake_plot(args):
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_png).write_bytes(b"fakepng")

    def _fake_pr(args):
        captured["pr"] = list(getattr(args, "selected_run_dirs", []) or [])
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_png).write_bytes(b"fakepng")

    def _fake_leaderboard(args):
        captured["leaderboard"] = list(getattr(args, "selected_run_dirs", []) or [])
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"model": "run_a", "composite_score": 1.0, "run_dir": str(run_a)}]).to_csv(args.out_csv, index=False)

    monkeypatch.setattr(results_analyzer, "cmd_inference_benchmark", _fake_benchmark)
    monkeypatch.setattr(results_analyzer, "cmd_inference_plot", _fake_plot)
    monkeypatch.setattr(results_analyzer, "cmd_pr_curves", _fake_pr)
    monkeypatch.setattr(results_analyzer, "cmd_test_metrics_plot", lambda _args: None)
    monkeypatch.setattr(results_analyzer, "cmd_export_table", lambda args: Path(args.output).parent.mkdir(parents=True, exist_ok=True) or pd.DataFrame([{"run_dir": str(run_a), "model": "yolo11n.pt"}]).to_csv(args.output, index=False))
    monkeypatch.setattr(results_analyzer, "cmd_leaderboard", _fake_leaderboard)
    monkeypatch.setattr(results_analyzer, "_collect_ultralytics_test_artifacts", lambda *_a, **_k: ([], []))

    answers = iter(["1", "2", "full", str(tmp_path / "datasets" / "ds_a" / "data.yaml")])
    monkeypatch.setattr("smartrain.results_analyzer.prompt_int", lambda *_a, **_k: int(next(answers)))
    monkeypatch.setattr("smartrain.results_analyzer.prompt_text", lambda *_a, **_k: str(next(answers)))
    monkeypatch.setattr("smartrain.results_analyzer.prompt_choice", lambda *_a, **_k: str(next(answers)))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    analyze_main(
        [
            "all",
            "--workspace",
            str(tmp_path),
            "--models-root",
            str(tmp_path / "runs"),
            "--analytics-session",
            "session_scope",
            "--no-pdf",
            "--no-odt",
        ]
    )
    assert sorted(captured["benchmark"]) == sorted([str(run_a), str(run_b)])
    assert sorted(captured["pr"]) == sorted([str(run_a), str(run_b)])
    assert sorted(captured["leaderboard"]) == sorted([str(run_a), str(run_b)])


def test_runs_with_missing_metrics_uses_run_resolved_yaml_for_unresolved_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"mAP50-95": 0.55}]).to_csv(run_dir / "test_metrics.csv", index=False)

    run_yaml = str(tmp_path / "datasets" / "ds_a" / "data.yaml")
    session_yaml = str(tmp_path / "datasets" / "other" / "data.yaml")

    monkeypatch.setattr(
        results_analyzer,
        "_resolve_data_yaml_for_run",
        lambda *_a, **_k: (run_yaml, "mock"),
    )

    def _fake_load_status(_run_dir: str, data_yaml: str, _split: str, _metrics: list[str]):
        assert data_yaml == run_yaml
        return {"unresolved_metrics": ["Box-F1"]}

    monkeypatch.setattr(results_analyzer, "_load_recompute_status", _fake_load_status)

    missing = results_analyzer._runs_with_missing_metrics(
        [str(run_dir)],
        ["mAP50-95", "Box-F1"],
        data_yaml=session_yaml,
        workspace=str(tmp_path),
        split="test",
    )
    assert missing == []


def test_runs_with_missing_metrics_skips_prompt_without_resolved_data_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"mAP50-95": 0.55}]).to_csv(run_dir / "test_metrics.csv", index=False)

    monkeypatch.setattr(
        results_analyzer,
        "_resolve_data_yaml_for_run",
        lambda *_a, **_k: ("", "none"),
    )
    monkeypatch.setattr(results_analyzer, "_load_recompute_status", lambda *_a, **_k: None)

    missing = results_analyzer._runs_with_missing_metrics(
        [str(run_dir)],
        ["mAP50-95", "Box-F1"],
        data_yaml=None,
        workspace=str(tmp_path),
        split="test",
    )
    assert missing == []


def test_runs_with_missing_metrics_skips_prompt_without_best_pt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"mAP50-95": 0.55}]).to_csv(run_dir / "test_metrics.csv", index=False)
    run_yaml = str(tmp_path / "datasets" / "ds_a" / "data.yaml")

    monkeypatch.setattr(
        results_analyzer,
        "_resolve_data_yaml_for_run",
        lambda *_a, **_k: (run_yaml, "mock"),
    )
    monkeypatch.setattr(results_analyzer, "_load_recompute_status", lambda *_a, **_k: None)

    missing = results_analyzer._runs_with_missing_metrics(
        [str(run_dir)],
        ["mAP50-95", "Box-F1"],
        data_yaml=run_yaml,
        workspace=str(tmp_path),
        split="test",
    )
    assert missing == []


def test_resolve_selected_run_dirs_allows_explicit_runs_outside_group(tmp_path: Path) -> None:
    run_a = _write_run(tmp_path, "ds_a", "run_a", model="yolo11n.pt", map5095=0.52, box_f1=0.61)
    run_b = _write_run(tmp_path, "ds_b", "run_b", model="yolo11s.pt", map5095=0.56, box_f1=0.65)
    scoped = results_analyzer._resolve_selected_run_dirs(
        str(tmp_path / "runs" / "ds_a"),
        [str(run_a), str(run_b)],
    )
    assert scoped == [str(run_a), str(run_b)]


def test_auto_select_data_yaml_prefers_candidate_with_existing_split_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "datasets" / "ds_a" / "test" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "datasets" / "ds_a" / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )

    runtime_yaml = run_dir / "_runtime_data_train.yaml"
    runtime_yaml.write_text(
        "path: /home/user/MarsSmarTrain/runs\ntrain: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )
    # This points to runtime yaml first; fallback candidate should come from metadata.
    (run_dir / "train" / "args.yaml").write_text(f"data: {runtime_yaml}\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {
                    "dataset": {"name": "ds_a", "path_under_workspace": "datasets/ds_a"},
                }
            }
        ),
        encoding="utf-8",
    )

    selected = results_analyzer._auto_select_data_yaml(
        str(run_dir),
        [],
        str(tmp_path),
        preferred_split="test",
    )
    assert selected == str(tmp_path / "datasets" / "ds_a" / "data.yaml")


def test_test_metrics_plot_saves_unresolved_status_on_recompute_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds_a"}}}),
        encoding="utf-8",
    )
    pd.DataFrame([{"mAP50-95": 0.55}]).to_csv(run_dir / "test_metrics.csv", index=False)
    (tmp_path / "datasets" / "ds_a").mkdir(parents=True, exist_ok=True)
    ((tmp_path / "datasets" / "ds_a") / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        results_analyzer,
        "_recompute_run_test_metrics",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    saved: list[dict[str, object]] = []

    def _fake_save_status(
        run_dir: str,
        data_yaml: str,
        split: str,
        requested_metrics: list[str],
        *,
        resolved: list[str],
        unresolved: list[str],
        status: str,
    ) -> None:
        saved.append(
            {
                "run_dir": run_dir,
                "data_yaml": data_yaml,
                "split": split,
                "requested_metrics": list(requested_metrics),
                "resolved": list(resolved),
                "unresolved": list(unresolved),
                "status": status,
            }
        )

    monkeypatch.setattr(results_analyzer, "_save_recompute_status", _fake_save_status)

    analyze_main(
        [
            "test-metrics-plot",
            "--workspace",
            str(tmp_path),
            "--runs-group-dir",
            str(tmp_path / "runs" / "ds_a"),
            "--metrics",
            "mAP50-95",
            "Box-F1",
            "--out-dir",
            str(tmp_path / "out"),
            "--recompute-missing-metrics",
        ]
    )
    assert saved
    assert any(item.get("status") == "error" and "Box-F1" in item.get("unresolved", []) for item in saved)

