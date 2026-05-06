from __future__ import annotations

import os

from smartrain.core.runtime.ultralytics_ephemeral import (
    best_effort_prune_runs_detect_near_run,
    prune_ultralytics_default_detect_under_runs,
    ultralytics_sidecar_dir,
)


def test_prune_removes_empty_detect_trees(tmp_path) -> None:
    runs = tmp_path / "runs"
    detect = runs / "detect"
    junk = detect / "predict"
    nested = junk / "sub"
    nested.mkdir(parents=True)
    keep = detect / "keep"
    keep.mkdir(parents=True)
    (keep / "x.txt").write_text("ok", encoding="utf-8")

    prune_ultralytics_default_detect_under_runs(str(runs))

    assert not junk.exists()
    assert keep.exists()
    assert (keep / "x.txt").is_file()
    assert detect.exists()


def test_prune_removes_detect_when_only_empty_children(tmp_path) -> None:
    runs = tmp_path / "runs"
    detect = runs / "detect"
    (detect / "val2").mkdir(parents=True)
    prune_ultralytics_default_detect_under_runs(str(runs))
    assert not detect.exists()


def test_prune_near_run_finds_runs_parent(tmp_path) -> None:
    ws = tmp_path / "ws"
    run_dir = ws / "runs" / "ds1" / "run_a"
    run_dir.mkdir(parents=True)
    detect = ws / "runs" / "detect"
    (detect / "predict").mkdir(parents=True)

    best_effort_prune_runs_detect_near_run(str(run_dir))

    assert not detect.exists()


def test_ultralytics_sidecar_dir_creates_path(tmp_path) -> None:
    p = ultralytics_sidecar_dir(str(tmp_path), "a", "b")
    assert p == os.path.join(str(tmp_path), "a", "b")
    assert os.path.isdir(p)
