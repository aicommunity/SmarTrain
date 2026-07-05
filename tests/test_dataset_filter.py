from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from smartrain.core.runtime.workspace_paths import DATASETS_INFO_FILE, WORKSPACE_ENV_VAR, deploy_workspace
from smartrain.services.datasets.bbox_edge_filter import (
    BboxEdgeFilterConfig,
    ContentBounds,
    bbox_geom_from_label,
    collect_baseline_stats,
    collect_empirical_bounds,
    is_baseline_inset,
    merge_content_bounds,
    should_drop_bbox,
    union_geoms_content_bounds,
)
from smartrain.services.datasets.yolo_labels import YoloBBox
from smartrain.workflows.datasets.dataset_filter import main as filter_main


def _write_jpg(path: Path, size: tuple[int, int] = (1000, 800), color: tuple[int, int, int] = (10, 10, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="JPEG", quality=85)


def _setup_split_dataset(tmp_path: Path, name: str = "src_ds") -> Path:
    ds = tmp_path / "datasets" / name
    for split in ("train", "val", "test"):
        (ds / split / "images").mkdir(parents=True, exist_ok=True)
        (ds / split / "labels").mkdir(parents=True, exist_ok=True)
    (tmp_path / "datasets" / DATASETS_INFO_FILE).write_text(
        json.dumps({name: {"classes": {"cat": 0, "dog": 1}, "structure": "split"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (ds / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\n\nnc: 2\nnames: ['cat', 'dog']\n",
        encoding="utf-8",
    )
    return ds


def test_baseline_inset_excludes_near_edge() -> None:
    geom = bbox_geom_from_label(YoloBBox(0, 0.02, 0.5, 0.04, 0.2), img_w=1000, img_h=800)
    assert geom is not None
    cfg = BboxEdgeFilterConfig(baseline_inset_margin=0.01)
    assert not is_baseline_inset(geom, config=cfg)


def test_should_drop_small_bbox_at_right_edge() -> None:
    geom = bbox_geom_from_label(YoloBBox(0, 0.9975, 0.5, 0.005, 0.1), img_w=1000, img_h=800)
    assert geom is not None
    cfg = BboxEdgeFilterConfig()
    samples = [(0, bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.2, 0.2), img_w=1000, img_h=800))]
    assert samples[0][1] is not None
    stats = collect_baseline_stats([(0, samples[0][1])], config=cfg, id_to_name={0: "cat"})
    drop, reason = should_drop_bbox(geom, config=cfg, class_stats=stats)
    assert drop
    assert reason is not None
    assert reason.value in {"abs_width", "rel_width"}


def test_edge_sides_left_ignores_right_edge_bbox() -> None:
    geom = bbox_geom_from_label(YoloBBox(0, 0.9975, 0.5, 0.005, 0.1), img_w=1000, img_h=800)
    assert geom is not None
    cfg = BboxEdgeFilterConfig(edge_sides="left")
    stats = collect_baseline_stats([], config=cfg, id_to_name={0: "cat"})
    drop, reason = should_drop_bbox(geom, config=cfg, class_stats=stats)
    assert not drop
    assert reason is None


def test_edge_sides_right_drops_right_edge_bbox() -> None:
    geom = bbox_geom_from_label(YoloBBox(0, 0.9975, 0.5, 0.005, 0.1), img_w=1000, img_h=800)
    assert geom is not None
    cfg = BboxEdgeFilterConfig(edge_sides="right")
    center = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.2, 0.2), img_w=1000, img_h=800)
    assert center is not None
    stats = collect_baseline_stats([(0, center)], config=cfg, id_to_name={0: "cat"})
    drop, reason = should_drop_bbox(geom, config=cfg, class_stats=stats)
    assert drop
    assert reason is not None
    assert reason.value in {"abs_width", "rel_width"}


def test_empirical_bounds_measures_from_content_hull() -> None:
    centers = [
        bbox_geom_from_label(YoloBBox(0, 0.4, 0.5, 0.2, 0.2), img_w=1000, img_h=800),
        bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.2, 0.2), img_w=1000, img_h=800),
        bbox_geom_from_label(YoloBBox(0, 0.6, 0.5, 0.2, 0.2), img_w=1000, img_h=800),
    ]
    assert all(c is not None for c in centers)
    content_bounds = merge_content_bounds([union_geoms_content_bounds(centers)])  # type: ignore[arg-type]
    assert content_bounds is not None
    near_content_left = bbox_geom_from_label(YoloBBox(0, 0.295, 0.5, 0.005, 0.1), img_w=1000, img_h=800)
    assert near_content_left is not None

    cfg_default = BboxEdgeFilterConfig()
    stats = collect_baseline_stats(
        [(0, c) for c in centers if c is not None],
        config=cfg_default,
        id_to_name={0: "cat"},
    )
    drop_default, _ = should_drop_bbox(near_content_left, config=cfg_default, class_stats=stats)
    assert not drop_default

    cfg_empirical = BboxEdgeFilterConfig(empirical_bounds=True)
    stats_e = collect_baseline_stats(
        [(0, c) for c in centers if c is not None],
        config=cfg_empirical,
        id_to_name={0: "cat"},
        content_bounds=content_bounds,
    )
    drop_empirical, reason = should_drop_bbox(
        near_content_left,
        config=cfg_empirical,
        class_stats=stats_e,
        content_bounds=content_bounds,
    )
    assert drop_empirical
    assert reason is not None
    assert reason.value in {"abs_width", "rel_width"}


def test_empirical_bounds_per_class_dataset_hull() -> None:
    center = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.2, 0.2), img_w=1000, img_h=800)
    edge = bbox_geom_from_label(YoloBBox(1, 0.98, 0.5, 0.02, 0.1), img_w=1000, img_h=800)
    assert center is not None and edge is not None
    cfg = BboxEdgeFilterConfig(empirical_bounds=True, edge_sides="horizontal", empirical_inset_only=True)
    samples = [(0, center), (1, edge)]
    bounds_map = collect_empirical_bounds(samples, config=cfg)
    class_bounds = bounds_map.by_class
    assert 0 in class_bounds and 1 in class_bounds
    assert class_bounds[0].x2 < 0.9


def test_filter_empirical_bounds_per_class_multi_class_frame(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "mixed.jpg")
    (ds / "train" / "labels" / "mixed.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.98 0.5 0.02 0.1\n",
        encoding="utf-8",
    )

    filter_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "src_ds",
            "--empirical-bounds",
            "--edge-sides",
            "horizontal",
            "-y",
        ]
    )

    manifest = json.loads((tmp_path / "datasets" / "src_ds_fltd" / "filter_manifest.json").read_text())
    summary = manifest["stats_after"]["empirical_content_bounds"]
    assert summary is not None
    assert summary["mode"] == "per_class_percentile"
    assert "cat" in summary["classes"]
    assert "dog" in summary["classes"]


def test_should_keep_large_bbox_at_edge() -> None:
    cfg = BboxEdgeFilterConfig()
    centers = [
        bbox_geom_from_label(YoloBBox(0, 0.3, 0.5, 0.3, 0.3), img_w=1000, img_h=800),
        bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.3, 0.3), img_w=1000, img_h=800),
        bbox_geom_from_label(YoloBBox(0, 0.7, 0.5, 0.3, 0.3), img_w=1000, img_h=800),
    ]
    edge = bbox_geom_from_label(YoloBBox(0, 0.86, 0.5, 0.28, 0.3), img_w=1000, img_h=800)
    assert all(c is not None for c in centers) and edge is not None
    stats = collect_baseline_stats([(0, c) for c in centers if c is not None], config=cfg, id_to_name={0: "cat"})
    drop, _ = should_drop_bbox(edge, config=cfg, class_stats=stats)
    assert not drop


def test_filter_creates_fltd_dataset(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "good.jpg")
    (ds / "train" / "labels" / "good.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_jpg(ds / "train" / "images" / "edge.jpg")
    (ds / "train" / "labels" / "edge.txt").write_text("0 0.9975 0.5 0.005 0.1\n", encoding="utf-8")

    filter_main(["--workspace", str(tmp_path), "--dataset", "src_ds", "-y"])

    out = tmp_path / "datasets" / "src_ds_fltd"
    assert (out / "train" / "images" / "good.jpg").is_file()
    assert not (out / "train" / "images" / "edge.jpg").exists()
    audit_img = out / "_filter_audit" / "dropped_images" / "train" / "images" / "edge.jpg"
    audit_lbl = out / "_filter_audit" / "dropped_images" / "train" / "labels" / "edge.txt"
    assert audit_img.is_file()
    assert audit_lbl.is_file()
    assert "0.9975" in audit_lbl.read_text(encoding="utf-8")
    assert (out / "dataset_passport.json").is_file()
    manifest = json.loads((out / "filter_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats_after"]["audit"]["dropped_image_pairs"] == 1


def test_filter_drop_images(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "bad.jpg")
    (ds / "train" / "labels" / "bad.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n0 0.9975 0.5 0.005 0.1\n",
        encoding="utf-8",
    )

    filter_main(["--workspace", str(tmp_path), "--dataset", "src_ds", "--drop-images", "-y"])

    out = tmp_path / "datasets" / "src_ds_fltd"
    assert not (out / "train" / "images" / "bad.jpg").exists()
    audit_img = out / "_filter_audit" / "dropped_images" / "train" / "images" / "bad.jpg"
    audit_lbl = out / "_filter_audit" / "dropped_images" / "train" / "labels" / "bad.txt"
    assert audit_img.is_file()
    assert audit_lbl.is_file()
    assert "0.9975" in audit_lbl.read_text(encoding="utf-8")
    assert "0.5 0.5 0.2 0.2" in audit_lbl.read_text(encoding="utf-8")


def test_filter_prune_empty(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "only_edge.jpg")
    (ds / "train" / "labels" / "only_edge.txt").write_text("0 0.9975 0.5 0.005 0.1\n", encoding="utf-8")

    filter_main(["--workspace", str(tmp_path), "--dataset", "src_ds", "-y"])

    out = tmp_path / "datasets" / "src_ds_fltd"
    assert not (out / "train" / "images" / "only_edge.jpg").exists()
    assert (out / "_filter_audit" / "dropped_images" / "train" / "images" / "only_edge.jpg").is_file()


def test_filter_archives_removed_labels_only(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "mixed.jpg")
    (ds / "train" / "labels" / "mixed.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n0 0.9975 0.5 0.005 0.1\n",
        encoding="utf-8",
    )

    filter_main(["--workspace", str(tmp_path), "--dataset", "src_ds", "-y"])

    out = tmp_path / "datasets" / "src_ds_fltd"
    assert (out / "train" / "images" / "mixed.jpg").is_file()
    kept = (out / "train" / "labels" / "mixed.txt").read_text(encoding="utf-8")
    assert "0.50000000" in kept
    assert "0.9975" not in kept
    removed = out / "_filter_audit" / "removed_labels" / "train" / "labels" / "mixed.txt"
    assert removed.is_file()
    assert "0.9975" in removed.read_text(encoding="utf-8")
    assert "0.5 0.5 0.2 0.2" not in removed.read_text(encoding="utf-8")


def test_filter_keeps_background_without_label(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "bg.jpg")

    filter_main(["--workspace", str(tmp_path), "--dataset", "src_ds", "-y"])

    out = tmp_path / "datasets" / "src_ds_fltd"
    assert (out / "train" / "images" / "bg.jpg").is_file()
    assert not (out / "_filter_audit" / "dropped_images" / "train" / "images" / "bg.jpg").exists()


def test_filter_keeps_empty_label_background_by_default(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "bg_empty.txt".replace(".txt", ".jpg"))
    (ds / "train" / "labels" / "bg_empty.txt").write_text("", encoding="utf-8")

    filter_main(["--workspace", str(tmp_path), "--dataset", "src_ds", "-y"])

    out = tmp_path / "datasets" / "src_ds_fltd"
    assert (out / "train" / "images" / "bg_empty.jpg").is_file()
    assert not (out / "train" / "labels" / "bg_empty.txt").exists()
    assert not (out / "_filter_audit" / "dropped_images" / "train" / "images" / "bg_empty.jpg").exists()


def test_filter_drop_background_removes_unlabeled(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "bg.jpg")
    (ds / "train" / "labels" / "bg.txt").write_text("", encoding="utf-8")

    filter_main(["--workspace", str(tmp_path), "--dataset", "src_ds", "--drop-background", "-y"])

    out = tmp_path / "datasets" / "src_ds_fltd"
    assert not (out / "train" / "images" / "bg.jpg").exists()
    assert (out / "_filter_audit" / "dropped_images" / "train" / "images" / "bg.jpg").is_file()


def test_filter_dry_run_and_stats_only(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "a.jpg")
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    filter_main(["--workspace", str(tmp_path), "--dataset", "src_ds", "--dry-run"])
    assert not (tmp_path / "datasets" / "src_ds_fltd").exists()

    filter_main(["--workspace", str(tmp_path), "--dataset", "src_ds", "--stats-only"])
    assert not (tmp_path / "datasets" / "src_ds_fltd").exists()
    assert (tmp_path / "tmp" / "filter_manifest.json").is_file()


def test_filter_interactive_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "a.jpg")
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(
        [
            "src_ds",
            "",
            "any",
            False,
            "0.01",
            False,
            True,
            "0.002",
            "8",
            "8",
            "0.10",
            "0.85",
            "0.85",
            False,
            False,
            False,
            False,
            False,
            True,
            "execute",
            True,
        ]
    )

    def _yes_no(*_a, default=False, **_k):
        return next(answers)

    def _text(*_a, default="", **_k):
        val = next(answers)
        return str(val) if val is not None else default

    def _choice(*_a, **_k):
        return next(answers)

    monkeypatch.setattr("smartrain.services.datasets.dataset_filter.prompt_yes_no", _yes_no)
    monkeypatch.setattr("smartrain.services.datasets.dataset_filter.prompt_text", _text)
    monkeypatch.setattr("smartrain.services.datasets.dataset_filter.prompt_choice", _choice)
    monkeypatch.setattr("smartrain.services.datasets.dataset_filter.prompt_multi_choice_csv", lambda *a, **k: [])

    filter_main([])
    assert (tmp_path / "datasets" / "src_ds_fltd").is_dir()


def test_filter_interactive_cancel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "a.jpg")
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(
        [
            "src_ds",
            "",
            "any",
            False,
            "0.01",
            False,
            True,
            "0.002",
            "8",
            "8",
            "0.10",
            "0.85",
            "0.85",
            False,
            False,
            False,
            False,
            False,
            True,
            "execute",
            False,
        ]
    )

    def _yes_no(*_a, default=False, **_k):
        return next(answers)

    def _text(*_a, default="", **_k):
        val = next(answers)
        return str(val) if val is not None else default

    def _choice(*_a, **_k):
        return next(answers)

    monkeypatch.setattr("smartrain.services.datasets.dataset_filter.prompt_yes_no", _yes_no)
    monkeypatch.setattr("smartrain.services.datasets.dataset_filter.prompt_text", _text)
    monkeypatch.setattr("smartrain.services.datasets.dataset_filter.prompt_choice", _choice)
    monkeypatch.setattr("smartrain.services.datasets.dataset_filter.prompt_multi_choice_csv", lambda *a, **k: [])

    filter_main([])
    assert not (tmp_path / "datasets" / "src_ds_fltd").exists()
