from __future__ import annotations

from smartrain.services.datasets.bbox_edge_filter import (
    BboxEdgeFilterConfig,
    BboxGeom,
    ClassBboxStats,
    bbox_geom_from_label,
    collect_baseline_stats,
    collect_empirical_bounds,
    should_drop_bbox,
    touches_physical_image_edge,
)
from smartrain.services.datasets.yolo_labels import YoloBBox


def _geom(cx: float, cy: float, w: float, h: float, *, img_w: int = 1000, img_h: int = 800) -> BboxGeom:
    g = bbox_geom_from_label(YoloBBox(0, cx, cy, w, h), img_w=img_w, img_h=img_h)
    assert g is not None
    return g


def test_dual_path_edge_touching_uses_image_borders() -> None:
    cfg = BboxEdgeFilterConfig(empirical_bounds=True, edge_sides="horizontal")
    center = _geom(0.5, 0.5, 0.2, 0.2)
    samples = [(0, center)]
    bounds = collect_empirical_bounds(samples, config=cfg)
    stats = collect_baseline_stats(samples, config=cfg, id_to_name={0: "cat"})
    edge = _geom(0.9975, 0.5, 0.005, 0.1)
    assert touches_physical_image_edge(edge, cfg.edge_eps)
    resolved = bounds.resolve(edge, config=cfg)
    assert resolved is None
    drop, reason = should_drop_bbox(edge, config=cfg, class_stats=stats, content_bounds=resolved)
    assert drop
    assert reason is not None


def test_percentile_bounds_exclude_edge_outliers_from_hull() -> None:
    cfg = BboxEdgeFilterConfig(empirical_bounds=True, empirical_percentile=0.10, empirical_inset_only=True)
    samples = [
        (0, _geom(0.45, 0.5, 0.2, 0.2)),
        (0, _geom(0.5, 0.5, 0.2, 0.2)),
        (0, _geom(0.55, 0.5, 0.2, 0.2)),
        (0, _geom(0.6, 0.5, 0.2, 0.2)),
        (0, _geom(0.65, 0.5, 0.2, 0.2)),
    ]
    bounds = collect_empirical_bounds(samples, config=cfg)
    assert bounds.by_class[0].x1 >= 0.35


def test_empirical_by_format_separate_hulls() -> None:
    cfg = BboxEdgeFilterConfig(empirical_bounds=True, empirical_by_format=True, empirical_inset_only=True)
    samples = [
        (1, _geom(0.5, 0.5, 0.2, 0.2, img_w=1000, img_h=800)),
        (1, _geom(0.5, 0.5, 0.2, 0.2, img_w=1920, img_h=1080)),
    ]
    bounds = collect_empirical_bounds(samples, config=cfg)
    assert (1, 1000, 800) in bounds.by_class_format
    assert (1, 1920, 1080) in bounds.by_class_format


def test_inset_bbox_uses_class_bounds() -> None:
    cfg = BboxEdgeFilterConfig(empirical_bounds=True, edge_sides="horizontal", empirical_inset_only=True)
    samples = [(0, _geom(0.5, 0.5, 0.2, 0.2))]
    bounds = collect_empirical_bounds(samples, config=cfg)
    stats = collect_baseline_stats(samples, config=cfg, id_to_name={0: "cat"})
    near_left = _geom(0.405, 0.5, 0.005, 0.1)
    resolved = bounds.resolve(near_left, config=cfg)
    assert resolved is not None
    drop, reason = should_drop_bbox(near_left, config=cfg, class_stats=stats, content_bounds=resolved)
    assert drop
    assert reason is not None
