from __future__ import annotations

from pathlib import Path

from smartrain.core.runtime.workspace_paths import WorkspaceLayout, deploy_workspace
from smartrain.services.analyze.report_sections.report_manifest import _path_for_report
from smartrain.services.visualization.pipeline import _portable_target_paths
from smartrain.services.visualization.target_resolution import _normalize_source_run_ref


def test_normalize_source_run_ref_relative_and_abs(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    layout = WorkspaceLayout(str(tmp_path))
    run = tmp_path / "runs" / "ds" / "r1"
    run.mkdir(parents=True)

    rel = _normalize_source_run_ref(layout, "runs/ds/r1")
    assert Path(rel).resolve() == run.resolve()

    abs_got = _normalize_source_run_ref(layout, str(run.resolve()))
    assert Path(abs_got).resolve() == run.resolve()

    backslash = _normalize_source_run_ref(layout, r"runs\ds\r1")
    assert Path(backslash).resolve() == run.resolve()


def test_path_for_report_keeps_relative_posix(tmp_path: Path) -> None:
    assert _path_for_report("runs/ds/r1", str(tmp_path)) == "runs/ds/r1"
    abs_run = tmp_path / "runs" / "ds" / "r1"
    abs_run.mkdir(parents=True)
    assert _path_for_report(str(abs_run), str(tmp_path)) == "runs/ds/r1"
    assert "\\" not in _path_for_report(str(abs_run), str(tmp_path))


def test_portable_target_paths_under_workspace(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "ds" / "r1"
    model = run / "models" / "a.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"pt")
    (tmp_path / "datasets" / "ds").mkdir(parents=True)
    out = _portable_target_paths(
        str(tmp_path),
        {
            "dataset_root": str(tmp_path / "datasets" / "ds"),
            "run_dir": str(run),
            "model_path": str(model),
        },
    )
    assert out["run_dir"] == "runs/ds/r1"
    assert out["model_path"] == "runs/ds/r1/models/a.pt"
    assert out["dataset_root"] == "datasets/ds"
    assert all("\\" not in str(v) for v in out.values() if v)
