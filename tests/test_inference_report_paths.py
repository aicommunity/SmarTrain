from __future__ import annotations

import argparse
from pathlib import Path

from smartrain.core.runtime.workspace_paths import WorkspaceLayout, deploy_workspace
from smartrain.services.inference_runtime_helpers import build_report, source_descriptor


def _minimal_args() -> argparse.Namespace:
    return argparse.Namespace(
        data_mode="folder",
        dataset=None,
        split=None,
        task="detect",
        external_provider="",
        conf=0.25,
        iou=0.45,
        img_size=640,
        img_size_source="",
        device="cpu",
        half=False,
        batch_size=1,
        limit=0,
        roi_pre_detect=False,
        roi_weights=None,
        roi_conf=0.25,
        roi_policy="largest",
        roi_pad_px=0,
        roi_on_empty="full",
        roi_class_ids=None,
        export_dataset=True,
        export_label_conf_min=0.25,
        export_label_conf_max=1.0,
        export_visualize=None,
        export_split_dirs=True,
        export_files_per_dir=500,
        export_classes=None,
        export_class_ids=None,
    )


def test_source_descriptor_omits_abs_under_workspace(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    raw = tmp_path / "raw_data"
    raw.mkdir(exist_ok=True)
    desc = source_descriptor(_minimal_args(), str(raw.resolve()), "raw_data", layout)
    assert "path_absolute" not in desc
    assert desc.get("path_relative") == "raw_data"
    assert "\\" not in str(desc.get("path_relative") or "")


def test_build_report_workspace_and_source_portable(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    model_path = tmp_path / "models" / "ds" / "r1" / "models" / "a.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"pt")
    raw = tmp_path / "raw_data"
    raw.mkdir(exist_ok=True)
    out_root = str(tmp_path / "inference" / "out")
    Path(out_root).mkdir(parents=True)
    report_path = str(Path(out_root) / "report.json")
    report = build_report(
        args=_minimal_args(),
        layout=layout,
        model_source="models",
        model_name="a",
        model_path=model_path,
        source_abs=str(raw.resolve()),
        source_short="raw_data",
        out_root=out_root,
        report_path=report_path,
        images_input_count=0,
        image_rows=[],
        skipped=0,
    )
    assert "root_absolute" not in report.get("workspace", {})
    assert report["workspace"].get("root_relative") == "."
    src = report.get("source") or {}
    assert "path_absolute" not in src
    assert src.get("path_relative") == "raw_data"
    model = report.get("model") or {}
    assert "weights_absolute" not in model
    assert "\\" not in str(model.get("weights_relative") or "")
