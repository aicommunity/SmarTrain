"""Report section 6 rendering with enriched ultralytics_test manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_report_section6_renders_completeness_and_per_class_table(tmp_path: Path) -> None:
    from smartrain.services.analyze.report_writer import write_analysis_report

    ultra_dir = tmp_path / "artifacts" / "ultralytics-test" / "R1"
    ultra_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"class_name": "construct", "ap": 0.55},
            {"class_name": "digits", "ap": 0.77},
        ]
    ).to_csv(ultra_dir / "pr_per_class.csv", index=False)
    (ultra_dir / "BoxPR_curve.png").write_bytes(b"fakepng")

    manifest = {
        "session_name": "s_ultra",
        "profile": "full",
        "baseline": "run_a",
        "others": [],
        "tables": [],
        "images": ["artifacts/ultralytics-test/R1/BoxPR_curve.png"],
        "artifacts": [],
        "ultralytics_test": [
            {
                "run_code": "R1",
                "run_name": "run_a",
                "completeness": "train_val_fallback",
                "missing_files": ["args.yaml"],
                "artifact_sources": {"BoxPR_curve.png": "train_val_fallback", "pr_per_class.csv": "test"},
                "csv": {
                    "pr_per_class.csv": "artifacts/ultralytics-test/R1/pr_per_class.csv",
                },
                "images": ["artifacts/ultralytics-test/R1/BoxPR_curve.png"],
                "run_info": {
                    "model": "yolo11n",
                    "dataset_name": "ds",
                    "epochs": 400,
                    "batch_size": 16,
                    "train_image_size": 640,
                    "val_imgsz": 640,
                },
                "machine_info": {
                    "sys_cpu_model": "AMD",
                    "sys_cpu_logical_cores": 16,
                    "sys_ram_total_gb": 64,
                    "sys_gpu_0_name": "RTX",
                    "sys_gpu_0_vram_gb": 24,
                    "sys_os": "Linux",
                    "sys_os_release": "6.8",
                },
            }
        ],
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True, languages=["en"])
    en_md = (tmp_path / "en" / "index.md").read_text(encoding="utf-8")
    assert "### 6.1 Run R1" in en_md
    assert "Completeness:" in en_md
    assert "train-ultralytics" in en_md
    assert "epochs=400" in en_md
    assert "Machine: CPU=AMD" in en_md
    assert "Per-class AP (Ultralytics test)" in en_md
    assert "construct" in en_md
    assert "BoxPR_curve" not in en_md or "Precision-Recall" in en_md


def test_pr_curves_does_not_write_tests_test_ultralytics(tmp_path: Path, monkeypatch) -> None:
    import argparse

    from smartrain.services.analyze import pr_curves

    run_dir = tmp_path / "runs" / "ds" / "run_pr"
    run_dir.mkdir(parents=True)
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "models" / "run_pr.pt").write_bytes(b"pt")

    import numpy as np

    def _fake_val(*_a, **_k):
        class _M:
            pass

        return _M()

    class _FakeYOLO:
        def __init__(self, *_a, **_k):
            pass

    import sys
    import types

    ultra = types.ModuleType("ultralytics")
    ultra.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultra)

    args = argparse.Namespace(
        runs_group_dir=str(tmp_path / "runs" / "ds"),
        data_yaml=str(tmp_path / "data.yaml"),
        workspace=str(tmp_path),
        out_png=str(tmp_path / "out.png"),
        selected_run_dirs=None,
        pr_per_class=True,
        reuse_run_cache=False,
        val_batch=1,
        val_imgsz=640,
        val_half=True,
        gpu_only_val=True,
    )
    (tmp_path / "data.yaml").write_text("names: [a]\ntest: .\n", encoding="utf-8")

    pr_curves.run_pr_curves(
        args,
        prompt_text_cb=lambda *_a, **_k: "",
        resolve_selected_run_dirs_cb=lambda *_a, **_k: [str(run_dir)],
        load_dataset_class_names_cb=lambda *_a, **_k: {0: "a"},
        preferred_run_model_path_cb=lambda rd, _e: str(run_dir / "models" / "run_pr.pt"),
        run_cache_root_cb=lambda rd: str(run_dir / "cache"),
        compute_fingerprint_cb=lambda _p: "fp",
        data_yaml_hash_cb=lambda _p: "h",
        weights_hash_cb=lambda _p: "w",
        clear_gpu_memory_cb=lambda: None,
        resolve_run_val_profile_cb=lambda *_a, **_k: (1, 640, True),
        ultralytics_sidecar_dir_cb=lambda *_a, **_k: str(run_dir / "sidecar"),
        run_val_memory_safe_cb=_fake_val,
        extract_pr_curve_cb=lambda _m: (np.array([0.0, 1.0]), np.array([1.0, 0.0])),
        extract_pr_curve_per_class_cb=lambda _m: None,
        append_cache_entry_cb=lambda *_a, **_k: None,
        safe_name_cb=lambda x: x,
        resolve_workspace_root_cb=lambda _w: str(tmp_path),
        workspace_layout_cls=lambda root: type("L", (), {"analytics": str(tmp_path / "analytics")})(),
    )

    assert not (run_dir / "tests" / "test-ultralytics" / "pr.csv").is_file()
    assert (run_dir / "cache" / "pr" / "aggregate" / "pr_fp.csv").is_file()
