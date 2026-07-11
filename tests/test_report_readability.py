"""Readability improvements for analyze reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def _write_training_run(tmp_path: Path, name: str, *, model: str, dataset: str, epochs: int) -> Path:
    run_dir = tmp_path / name
    train = run_dir / "train-ultralytics"
    train.mkdir(parents=True)
    (run_dir / "training_metadata.json").write_text(
        '{"hyperparameters": {"epochs": '
        + str(epochs)
        + '}, "system_profile": {}, "training_info": {"model": "'
        + model
        + '", "dataset": {"name": "'
        + dataset
        + '"}, "hyperparameters": {"epochs": '
        + str(epochs)
        + ', "batch_size": 16}}}',
        encoding="utf-8",
    )
    yaml.safe_dump({"model": f"{model}.pt", "epochs": epochs, "batch": 16}, (train / "args.yaml").open("w", encoding="utf-8"))
    return run_dir


def test_collision_labels_include_dataset(tmp_path: Path) -> None:
    from smartrain.services.analyze.report_labels import build_run_legend_rows

    run_a = _write_training_run(tmp_path, "run_a", model="yolo11m", dataset="ds_a", epochs=200)
    run_b = _write_training_run(tmp_path, "run_b", model="yolo11m", dataset="ds_b", epochs=100)
    legend = build_run_legend_rows([str(run_a), str(run_b)])
    assert legend[0].short_label != legend[1].short_label
    assert "D1" in legend[0].short_label or "ds_a" in legend[0].short_label
    assert "D2" in legend[1].short_label or "ds_b" in legend[1].short_label


def test_figure_caption_ultralytics_includes_run(tmp_path: Path) -> None:
    from smartrain.services.analyze.report_sections.report_figures import _figure_caption

    manifest = {
        "ultralytics_test": [
            {
                "run_code": "M1_yolo11m",
                "run_info": {"model": "yolo11m", "dataset_name": "ds_a", "epochs": 200, "batch_size": 16, "val_imgsz": 640},
            }
        ],
        "run_legend": [{"short_label": "M1 · yolo11m · D1 · 200ep · b16", "index": 1}],
    }
    rel = "artifacts/ultralytics-test/M1_yolo11m/BoxPR_curve.png"
    caption = _figure_caption(rel, 1, {"M1_yolo11m": "M1 · yolo11m · D1 · 200ep · b16"}, manifest, True)
    assert "M1" in caption
    assert "yolo11m" in caption


def test_executive_summary_is_first_section(tmp_path: Path) -> None:
    from smartrain.services.analyze.report_writer import write_analysis_report

    (tmp_path / "artifacts" / "leaderboard").mkdir(parents=True, exist_ok=True)
    lb = tmp_path / "artifacts" / "leaderboard" / "leaderboard.csv"
    pd.DataFrame(
        [{"model": "run_a", "run_name": "run_a", "composite_score": 1.2, "quality_metric": 0.8, "speed_metric": 4.0}]
    ).to_csv(lb, index=False)
    manifest = {
        "session_name": "s_exec",
        "profile": "full",
        "baseline": "run_a",
        "others": [],
        "tables": ["artifacts/leaderboard/leaderboard.csv"],
        "images": [],
        "artifacts": [],
        "format_comparison": {},
        "abbreviations": {"run_a": "M1 yolo11n"},
        "run_legend": [
            {
                "index": 1,
                "short_label": "M1 yolo11n",
                "architecture": "yolo11n",
                "dataset_label": "D1",
                "epochs": "200",
                "batch": "16",
                "run_name": "run_a",
                "run_dir": "run_a",
                "role": "baseline",
            }
        ],
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    ru_md = (tmp_path / "ru" / "index.md").read_text(encoding="utf-8")
    pos_exec = ru_md.find("## 1. Краткое резюме")
    pos_context = ru_md.find("## 2. Контекст")
    pos_lb = ru_md.find("Рейтинг моделей (сводка)")
    assert pos_exec != -1 and pos_context != -1
    assert pos_exec < pos_context
    assert pos_lb != -1 and pos_lb < pos_context


def test_compare_delta_takeaways_use_run_labels(tmp_path: Path) -> None:
    from smartrain.services.analyze.report_markdown_formatting import _compare_delta_takeaways

    df = pd.DataFrame(
        [
            {"baseline": "M1 yolo11m", "other": "M2 yolo11m", "delta_mAP50-95": -0.1},
            {"baseline": "M1 yolo11m", "other": "M3 yolo11m", "delta_mAP50-95": 0.12},
        ]
    )
    lines = _compare_delta_takeaways(df, True, {})
    assert any("M3 yolo11m" in line for line in lines)
    assert not any("**0**" in line or " **1**" in line for line in lines)


def test_compact_ultralytics_defers_extra_images(tmp_path: Path) -> None:
    from smartrain.services.analyze.report_writer import write_analysis_report

    ultra_dir = tmp_path / "artifacts" / "ultralytics-test" / "M1_yolo11m"
    ultra_dir.mkdir(parents=True)
    for name in ("BoxPR_curve.png", "BoxF1_curve.png", "confusion_matrix_normalized.png", "val_batch0_pred.jpg"):
        (ultra_dir / name).write_bytes(b"x")
    images = [f"artifacts/ultralytics-test/M1_yolo11m/{n}" for n in ("BoxPR_curve.png", "BoxF1_curve.png", "confusion_matrix_normalized.png", "val_batch0_pred.jpg")]
    manifest = {
        "session_name": "s_ultra",
        "profile": "full",
        "baseline": "run_a",
        "others": ["run_b"],
        "single_run_mode": False,
        "tables": [],
        "images": images,
        "artifacts": [],
        "format_comparison": {},
        "abbreviations": {"run_a": "M1 yolo11m", "run_b": "M2 yolo11m"},
        "ultralytics_test": [
            {
                "run_code": "M1_yolo11m",
                "run_name": "run_a",
                "images": images,
                "csv": {},
                "run_info": {"model": "yolo11m", "dataset_name": "ds", "epochs": 200, "batch_size": 16},
            }
        ],
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    ru_md = (tmp_path / "ru" / "index.md").read_text(encoding="utf-8")
    ultra_pos = ru_md.find("## 6. Результаты Ultralytics test")
    appendix_pos = ru_md.find("## 7. Приложение")
    assert ultra_pos != -1 and appendix_pos != -1
    ultra_body = ru_md[ultra_pos:appendix_pos]
    assert ultra_body.count("![](../artifacts/ultralytics-test/") <= 3
    assert "BoxF1_curve" in ru_md[appendix_pos:]
