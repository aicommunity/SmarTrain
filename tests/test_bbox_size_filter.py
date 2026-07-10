from __future__ import annotations

from smartrain.services.datasets.bbox_edge_filter import (
    BboxEdgeFilterConfig,
    _stable_typical_threshold,
    allowed_size_dims,
    bbox_geom_from_label,
    collect_baseline_stats,
    collect_size_baseline_stats,
    should_drop_small_bbox,
)
from smartrain.services.datasets.yolo_labels import YoloBBox


def _baseline_stats(*, img_w: int = 1000, img_h: int = 800) -> dict:
    center = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.2, 0.2), img_w=img_w, img_h=img_h)
    assert center is not None
    cfg = BboxEdgeFilterConfig()
    return collect_baseline_stats([(0, center)], config=cfg, id_to_name={0: "cat"})


def test_should_drop_small_bbox_at_image_center() -> None:
    small = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.005, 0.005), img_w=1000, img_h=800)
    assert small is not None
    cfg = BboxEdgeFilterConfig(edge_filter=False, size_filter=True)
    stats = _baseline_stats()
    drop, reason = should_drop_small_bbox(small, config=cfg, class_stats=stats)
    assert drop
    assert reason is not None
    assert reason.value in {"abs_width", "rel_width", "abs_height", "rel_height"}


def test_should_drop_small_bbox_at_edge_when_size_filter_on() -> None:
    small_edge = bbox_geom_from_label(YoloBBox(0, 0.9975, 0.5, 0.005, 0.005), img_w=1000, img_h=800)
    assert small_edge is not None
    cfg = BboxEdgeFilterConfig(edge_filter=False, size_filter=True)
    stats = _baseline_stats()
    drop, reason = should_drop_small_bbox(small_edge, config=cfg, class_stats=stats)
    assert drop
    assert reason is not None


def test_size_dims_width_keeps_tall_narrow_bbox() -> None:
    tall_narrow = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.005, 0.2), img_w=1000, img_h=800)
    assert tall_narrow is not None
    cfg = BboxEdgeFilterConfig(edge_filter=False, size_filter=True, size_dims="width", abs_min_width_px=8.0)
    stats = _baseline_stats()
    drop, reason = should_drop_small_bbox(tall_narrow, config=cfg, class_stats=stats)
    assert drop
    assert reason is not None
    assert reason.value in {"abs_width", "rel_width"}


def test_size_dims_width_keeps_when_only_height_small() -> None:
    short_wide = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.2, 0.005), img_w=1000, img_h=800)
    assert short_wide is not None
    cfg = BboxEdgeFilterConfig(
        edge_filter=False,
        size_filter=True,
        size_dims="width",
        abs_min_width_px=8.0,
        abs_min_height_px=8.0,
    )
    stats = _baseline_stats()
    drop, reason = should_drop_small_bbox(short_wide, config=cfg, class_stats=stats)
    assert not drop
    assert reason is None


def test_size_dims_height_keeps_when_only_width_small() -> None:
    narrow_tall = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.005, 0.2), img_w=1000, img_h=800)
    assert narrow_tall is not None
    cfg = BboxEdgeFilterConfig(
        edge_filter=False,
        size_filter=True,
        size_dims="height",
        abs_min_width_px=8.0,
        abs_min_height_px=8.0,
    )
    stats = _baseline_stats()
    drop, reason = should_drop_small_bbox(narrow_tall, config=cfg, class_stats=stats)
    assert not drop
    assert reason is None


def test_size_dims_height_drops_short_bbox() -> None:
    short = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.2, 0.005), img_w=1000, img_h=800)
    assert short is not None
    cfg = BboxEdgeFilterConfig(edge_filter=False, size_filter=True, size_dims="height", abs_min_height_px=8.0)
    stats = _baseline_stats()
    drop, reason = should_drop_small_bbox(short, config=cfg, class_stats=stats)
    assert drop
    assert reason is not None
    assert reason.value in {"abs_height", "rel_height"}


def test_allowed_size_dims_any() -> None:
    assert allowed_size_dims("any") == frozenset({"width", "height"})


def test_size_filter_off_never_drops() -> None:
    small = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.001, 0.001), img_w=1000, img_h=800)
    assert small is not None
    cfg = BboxEdgeFilterConfig(size_filter=False)
    stats = _baseline_stats()
    drop, reason = should_drop_small_bbox(small, config=cfg, class_stats=stats)
    assert not drop
    assert reason is None


def test_stable_typical_threshold_bulk_trim() -> None:
    values = [500.0] * 20 + [40.0, 50.0, 60.0]
    typical, threshold, bulk_n = _stable_typical_threshold(
        values,
        split_ratio=0.5,
        typical_quantile=0.25,
        rel_factor=0.85,
        min_baseline_samples=5,
    )
    assert bulk_n == 20
    assert typical == 500.0
    assert threshold == 425.0


def test_stable_baseline_higher_threshold_than_inset_bimodal() -> None:
    samples: list[tuple[int, object]] = []
    for _ in range(18):
        g = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.5, 0.2), img_w=1000, img_h=800)
        assert g is not None
        samples.append((0, g))
    for _ in range(3):
        g = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.04, 0.2), img_w=1000, img_h=800)
        assert g is not None
        samples.append((0, g))
    for _ in range(2):
        g = bbox_geom_from_label(YoloBBox(0, 0.02, 0.5, 0.074, 0.2), img_w=1000, img_h=800)
        assert g is not None
        samples.append((0, g))

    inset_cfg = BboxEdgeFilterConfig(
        edge_filter=False,
        size_filter=True,
        size_dims="width",
        size_baseline_mode="inset",
    )
    stable_cfg = BboxEdgeFilterConfig(
        edge_filter=False,
        size_filter=True,
        size_dims="width",
        size_baseline_mode="stable",
    )
    class_stats = collect_baseline_stats(samples, config=inset_cfg, id_to_name={0: "cat"})
    size_map = collect_size_baseline_stats(samples, config=stable_cfg, id_to_name={0: "cat"})
    assert size_map is not None

    target = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.074, 0.2), img_w=1000, img_h=800)
    assert target is not None

    drop_inset, _ = should_drop_small_bbox(
        target, config=inset_cfg, class_stats=class_stats, size_stats_map=None
    )
    drop_stable, _ = should_drop_small_bbox(
        target, config=stable_cfg, class_stats=class_stats, size_stats_map=size_map
    )
    assert not drop_inset
    assert drop_stable


def test_stable_size_by_format_different_thresholds() -> None:
    samples: list[tuple[int, object]] = []
    for _ in range(10):
        g = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.3, 0.2), img_w=1000, img_h=800)
        assert g is not None
        samples.append((0, g))
    for _ in range(10):
        g = bbox_geom_from_label(YoloBBox(0, 0.5, 0.5, 0.5, 0.2), img_w=2000, img_h=1600)
        assert g is not None
        samples.append((0, g))

    cfg = BboxEdgeFilterConfig(
        edge_filter=False,
        size_filter=True,
        size_dims="width",
        size_baseline_mode="stable",
        size_by_format=True,
    )
    size_map = collect_size_baseline_stats(samples, config=cfg, id_to_name={0: "cat"})
    assert size_map is not None
    row_1k = size_map.by_class_format[(0, 1000, 800)]
    row_2k = size_map.by_class_format[(0, 2000, 1600)]
    assert row_2k.width_threshold > row_1k.width_threshold
