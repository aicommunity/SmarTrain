"""Executive summary content: metrics, confidence, ultralytics pairs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_executive_summary_includes_confidence_and_ultra_pairs(tmp_path: Path) -> None:
    from smartrain.services.analyze.report_writer import write_analysis_report

    (tmp_path / "artifacts" / "table").mkdir(parents=True)
    (tmp_path / "artifacts" / "confidence").mkdir(parents=True)
    ultra_dir = tmp_path / "artifacts" / "ultralytics-test" / "M1_yolo11m"
    ultra_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"run_name": "run_a", "test_mAP50-95": 0.91, "test_mAP50": 0.99, "test_Box-F1": 0.96}]
    ).to_csv(tmp_path / "artifacts" / "table" / "runs_summary.csv", index=False)
    for objective, conf, f1 in (("A", 0.55, 0.76), ("B", 0.31, 0.78), ("C", 0.58, 0.74)):
        pd.DataFrame(
            [
                {
                    "run_name": "run_a",
                    "split": "test",
                    "objective": objective,
                    "level": "global",
                    "class_id": -1,
                    "class_name": "all",
                    "recommended_conf": conf,
                    "target_metric": f1,
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": f1,
                    "status": "ok",
                }
            ]
        ).to_csv(tmp_path / "artifacts" / "confidence" / f"confidence_recommendations_{objective}.csv", index=False)
    (ultra_dir / "BoxPR_curve.png").write_bytes(b"png")
    (ultra_dir / "confusion_matrix_normalized.png").write_bytes(b"png")
    pr_rel = "artifacts/ultralytics-test/M1_yolo11m/BoxPR_curve.png"
    cm_rel = "artifacts/ultralytics-test/M1_yolo11m/confusion_matrix_normalized.png"
    manifest = {
        "session_name": "s_exec_full",
        "profile": "full",
        "baseline": "run_a",
        "others": [],
        "tables": [
            "artifacts/table/runs_summary.csv",
            "artifacts/confidence/confidence_recommendations_A.csv",
            "artifacts/confidence/confidence_recommendations_B.csv",
            "artifacts/confidence/confidence_recommendations_C.csv",
        ],
        "images": [pr_rel, cm_rel],
        "artifacts": [],
        "format_comparison": {},
        "abbreviations": {"run_a": "M1 yolo11m"},
        "confidence_recommendations": {
            "A": "artifacts/confidence/confidence_recommendations_A.csv",
            "B": "artifacts/confidence/confidence_recommendations_B.csv",
            "C": "artifacts/confidence/confidence_recommendations_C.csv",
        },
        "ultralytics_test": [
            {
                "run_code": "M1_yolo11m",
                "run_name": "run_a",
                "images": [pr_rel, cm_rel],
                "csv": {},
                "run_info": {"model": "yolo11m", "dataset_name": "ds_a", "epochs": 200, "batch_size": 16},
                "artifact_sources": {
                    "BoxPR_curve.png": "test",
                    "confusion_matrix_normalized.png": "test",
                },
            }
        ],
        "run_legend": [
            {
                "index": 1,
                "short_label": "M1 yolo11m",
                "architecture": "yolo11m",
                "dataset_label": "D1",
                "epochs": "200",
                "batch": "16",
                "image_size": "640",
                "run_name": "run_a",
                "run_dir": "run_a",
                "role": "baseline",
            }
        ],
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    ru_md = (tmp_path / "ru" / "index.md").read_text(encoding="utf-8")
    exec_end = ru_md.find("## 2. Контекст")
    exec_body = ru_md[:exec_end]
    assert "Основные метрики по запускам" in exec_body
    assert "Рекомендации confidence (A/B/C)" in exec_body
    assert "A: порог" in exec_body
    assert "run_a" in exec_body or "M1 yolo11m" in exec_body
    assert "PR-кривая и матрица ошибок" in exec_body
    assert pr_rel.split("/")[-1] in exec_body or "BoxPR_curve" in exec_body
    assert exec_body.count("|:---:|") >= 1
    ultra_pos = ru_md.find("## 6. Результаты Ultralytics test")
    if ultra_pos != -1:
        ultra_body = ru_md[ultra_pos:]
        assert ultra_body.count("![PR](../artifacts/ultralytics-test/") == 0
        assert ultra_body.count("![CM](../artifacts/ultralytics-test/") == 0
