from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from smartrain.core.runtime.workspace_paths import deploy_workspace
from smartrain.services.analyze.all_selection import _display_model_column
from smartrain.services.inference_runtime_helpers import list_workspace_detector_weights


def test_list_workspace_detector_weights_includes_nested_models(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    (tmp_path / "yolo11n.pt").write_bytes(b"root")
    release = tmp_path / "models" / "ds" / "detect_yolo11s_20260708"
    (release / "models").mkdir(parents=True)
    (release / "models" / "detect_yolo11s_20260708.pt").write_bytes(b"nested")
    entries = list_workspace_detector_weights(str(tmp_path))
    assert "yolo11n.pt" in entries
    assert any(e.endswith("detect_yolo11s_20260708.pt") for e in entries)


def test_display_model_column_prefers_detect_stem_for_r3(tmp_path: Path) -> None:
    run_folder = tmp_path / "models" / "ds" / "2026-07-14_20-42_ultralytics_yolo11s_640px_400epochs_b16-b1ef93cc"
    run_folder.mkdir(parents=True)
    pt = run_folder / "detect_yolo11s_20260714_204230_640px_400epochs_b16.pt"
    pt.write_bytes(b"pt")
    (run_folder / f"{pt.stem}.json").write_text(
        '{"artifacts":{"release_dir":"x","model_path":"y"},"source":{"source_run":"z"}}',
        encoding="utf-8",
    )
    label = _display_model_column(str(run_folder), SimpleNamespace(model=run_folder.name), width=40)
    assert label.startswith("detect_yolo11s_")
