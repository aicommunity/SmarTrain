from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from smartrain.services.datasets.yolo_labels import YoloBBox, YoloLabel, YoloSegment


class DropReason(str, Enum):
    ABS_WIDTH = "abs_width"
    REL_WIDTH = "rel_width"
    ABS_HEIGHT = "abs_height"
    REL_HEIGHT = "rel_height"
    MIN_VISIBILITY = "min_visibility"
    MIN_AREA = "min_area"
    ASPECT_RATIO = "aspect_ratio"


@dataclass(frozen=True)
class BboxEdgeFilterConfig:
    edge_filter: bool = True
    baseline_inset_margin: float = 0.01
    baseline_inset_margin_px: float | None = None
    edge_eps: float = 0.002
    filter_proximity_margin: float | None = None
    abs_min_width_px: float = 8.0
    abs_min_height_px: float = 8.0
    rel_quantile: float = 0.10
    rel_width_factor: float = 0.85
    rel_height_factor: float = 0.85
    min_visibility: float = 0.0
    min_area_px: float = 0.0
    max_aspect_ratio: float | None = None
    min_baseline_samples: int = 5

    def resolved_filter_proximity_margin(self) -> float:
        if self.filter_proximity_margin is not None:
            return float(self.filter_proximity_margin)
        if self.baseline_inset_margin_px is not None:
            return float(self.baseline_inset_margin)
        return float(self.baseline_inset_margin)


@dataclass
class ClassBboxStats:
    cls_id: int
    cls_name: str = ""
    baseline_eligible_count: int = 0
    baseline_excluded_near_edge_count: int = 0
    width_px_samples: list[float] = field(default_factory=list)
    height_px_samples: list[float] = field(default_factory=list)
    width_p10: float = 0.0
    width_p25: float = 0.0
    width_p50: float = 0.0
    width_p90: float = 0.0
    height_p10: float = 0.0
    height_p25: float = 0.0
    height_p50: float = 0.0
    height_p90: float = 0.0
    baseline_fallback: str | None = None

    def finalize(self, *, rel_quantile: float) -> None:
        if self.width_px_samples:
            self.width_p10 = _percentile(self.width_px_samples, 0.10)
            self.width_p25 = _percentile(self.width_px_samples, 0.25)
            self.width_p50 = _percentile(self.width_px_samples, 0.50)
            self.width_p90 = _percentile(self.width_px_samples, 0.90)
        if self.height_px_samples:
            self.height_p10 = _percentile(self.height_px_samples, 0.10)
            self.height_p25 = _percentile(self.height_px_samples, 0.25)
            self.height_p50 = _percentile(self.height_px_samples, 0.50)
            self.height_p90 = _percentile(self.height_px_samples, 0.90)

    def rel_width_threshold(self, rel_quantile: float, rel_factor: float) -> float:
        if self.baseline_fallback == "absolute_only" or self.baseline_eligible_count < 1:
            return 0.0
        q = _quantile_value(self.width_px_samples, rel_quantile, fallback_p50=self.width_p50)
        return float(rel_factor * q) if q > 0 else 0.0

    def rel_height_threshold(self, rel_quantile: float, rel_factor: float) -> float:
        if self.baseline_fallback == "absolute_only" or self.baseline_eligible_count < 1:
            return 0.0
        q = _quantile_value(self.height_px_samples, rel_quantile, fallback_p50=self.height_p50)
        return float(rel_factor * q) if q > 0 else 0.0

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "cls_id": self.cls_id,
            "cls_name": self.cls_name,
            "baseline_eligible_count": self.baseline_eligible_count,
            "baseline_excluded_near_edge_count": self.baseline_excluded_near_edge_count,
            "width_px": {
                "p10": self.width_p10,
                "p25": self.width_p25,
                "p50": self.width_p50,
                "p90": self.width_p90,
            },
            "height_px": {
                "p10": self.height_p10,
                "p25": self.height_p25,
                "p50": self.height_p50,
                "p90": self.height_p90,
            },
            "baseline_fallback": self.baseline_fallback,
        }


@dataclass(frozen=True)
class BboxGeom:
    cls_id: int
    cx: float
    cy: float
    w: float
    h: float
    x1: float
    y1: float
    x2: float
    y2: float
    w_px: float
    h_px: float
    img_w: int
    img_h: int
    is_segment_proxy: bool = False


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(float(v) for v in values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _quantile_value(values: list[float], q: float, *, fallback_p50: float) -> float:
    if values:
        return _percentile(values, q)
    if fallback_p50 > 0:
        return fallback_p50
    return 0.0


def _margin_norm(*, margin_norm: float, margin_px: float | None, img_w: int, img_h: int) -> float:
    if margin_px is not None and margin_px > 0:
        side = max(int(img_w), int(img_h), 1)
        return max(float(margin_norm), float(margin_px) / float(side))
    return float(margin_norm)


def bbox_geom_from_label(lb: YoloLabel, *, img_w: int, img_h: int) -> BboxGeom | None:
    if isinstance(lb, YoloBBox):
        cx, cy, w, h = float(lb.cx), float(lb.cy), float(lb.w), float(lb.h)
        is_seg = False
        cls_id = int(lb.cls_id)
    elif isinstance(lb, YoloSegment):
        xs = [p[0] for p in lb.points]
        ys = [p[1] for p in lb.points]
        if not xs or not ys:
            return None
        x1n, x2n = min(xs), max(xs)
        y1n, y2n = min(ys), max(ys)
        w, h = x2n - x1n, y2n - y1n
        if w <= 0 or h <= 0:
            return None
        cx, cy = (x1n + x2n) / 2.0, (y1n + y2n) / 2.0
        cls_id = int(lb.cls_id)
        is_seg = True
    else:
        return None
    x1, y1, x2, y2 = cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
    return BboxGeom(
        cls_id=cls_id,
        cx=cx,
        cy=cy,
        w=w,
        h=h,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        w_px=w * img_w,
        h_px=h * img_h,
        img_w=int(img_w),
        img_h=int(img_h),
        is_segment_proxy=is_seg,
    )


def is_baseline_inset(geom: BboxGeom, *, config: BboxEdgeFilterConfig) -> bool:
    margin = _margin_norm(
        margin_norm=config.baseline_inset_margin,
        margin_px=config.baseline_inset_margin_px,
        img_w=geom.img_w,
        img_h=geom.img_h,
    )
    return (
        geom.x1 >= margin
        and geom.y1 >= margin
        and geom.x2 <= 1.0 - margin
        and geom.y2 <= 1.0 - margin
    )


def _touches_edge_eps(geom: BboxGeom, eps: float) -> bool:
    return geom.x1 <= eps or geom.y1 <= eps or geom.x2 >= 1.0 - eps or geom.y2 >= 1.0 - eps


def _extends_outside(geom: BboxGeom) -> bool:
    return geom.x1 < 0.0 or geom.y1 < 0.0 or geom.x2 > 1.0 or geom.y2 > 1.0


def in_filter_zone(geom: BboxGeom, *, config: BboxEdgeFilterConfig) -> bool:
    if _extends_outside(geom):
        return True
    if _touches_edge_eps(geom, config.edge_eps):
        return True
    margin = _margin_norm(
        margin_norm=config.resolved_filter_proximity_margin(),
        margin_px=config.baseline_inset_margin_px,
        img_w=geom.img_w,
        img_h=geom.img_h,
    )
    return (
        geom.x1 <= margin
        or geom.y1 <= margin
        or geom.x2 >= 1.0 - margin
        or geom.y2 >= 1.0 - margin
    )


def _clip_xyxy(geom: BboxGeom) -> tuple[float, float, float, float]:
    x1 = min(1.0, max(0.0, geom.x1))
    y1 = min(1.0, max(0.0, geom.y1))
    x2 = min(1.0, max(0.0, geom.x2))
    y2 = min(1.0, max(0.0, geom.y2))
    return x1, y1, x2, y2


def _edge_sides(geom: BboxGeom, *, proximity_margin: float, edge_eps: float) -> set[str]:
    sides: set[str] = set()
    if geom.x1 < 0.0 or geom.x1 <= edge_eps or geom.x1 <= proximity_margin:
        sides.add("left")
    if geom.x2 > 1.0 or geom.x2 >= 1.0 - edge_eps or geom.x2 >= 1.0 - proximity_margin:
        sides.add("right")
    if geom.y1 < 0.0 or geom.y1 <= edge_eps or geom.y1 <= proximity_margin:
        sides.add("top")
    if geom.y2 > 1.0 or geom.y2 >= 1.0 - edge_eps or geom.y2 >= 1.0 - proximity_margin:
        sides.add("bottom")
    return sides


def collect_baseline_stats(
    samples: list[tuple[int, BboxGeom]],
    *,
    config: BboxEdgeFilterConfig,
    id_to_name: dict[int, str] | None = None,
) -> dict[int, ClassBboxStats]:
    stats: dict[int, ClassBboxStats] = {}
    for cls_id, geom in samples:
        row = stats.setdefault(
            cls_id,
            ClassBboxStats(cls_id=cls_id, cls_name=(id_to_name or {}).get(cls_id, str(cls_id))),
        )
        if is_baseline_inset(geom, config=config):
            row.baseline_eligible_count += 1
            row.width_px_samples.append(geom.w_px)
            row.height_px_samples.append(geom.h_px)
        else:
            row.baseline_excluded_near_edge_count += 1

    for cls_id, row in stats.items():
        if row.baseline_eligible_count < config.min_baseline_samples:
            fallback_w: list[float] = []
            fallback_h: list[float] = []
            for cid, geom in samples:
                if cid != cls_id:
                    continue
                if in_filter_zone(geom, config=config):
                    continue
                fallback_w.append(geom.w_px)
                fallback_h.append(geom.h_px)
            if len(fallback_w) >= config.min_baseline_samples:
                row.width_px_samples = fallback_w
                row.height_px_samples = fallback_h
                row.baseline_fallback = "non_edge_samples"
            else:
                row.width_px_samples = list(row.width_px_samples)
                row.height_px_samples = list(row.height_px_samples)
                row.baseline_fallback = "absolute_only"
        row.finalize(rel_quantile=config.rel_quantile)
    return stats


def should_drop_bbox(
    geom: BboxGeom,
    *,
    config: BboxEdgeFilterConfig,
    class_stats: dict[int, ClassBboxStats],
) -> tuple[bool, DropReason | None]:
    if not config.edge_filter:
        return _global_filters(geom, config)

    if not in_filter_zone(geom, config=config):
        return _global_filters(geom, config)

    proximity = _margin_norm(
        margin_norm=config.resolved_filter_proximity_margin(),
        margin_px=config.baseline_inset_margin_px,
        img_w=geom.img_w,
        img_h=geom.img_h,
    )
    sides = _edge_sides(geom, proximity_margin=proximity, edge_eps=config.edge_eps)
    x1c, y1c, x2c, y2c = _clip_xyxy(geom)
    w_clip_px = max(0.0, (x2c - x1c) * geom.img_w)
    h_clip_px = max(0.0, (y2c - y1c) * geom.img_h)
    orig_area = max(geom.w_px * geom.h_px, 1e-9)
    clip_area = w_clip_px * h_clip_px
    visibility = clip_area / orig_area

    if config.min_visibility > 0 and visibility < config.min_visibility:
        return True, DropReason.MIN_VISIBILITY

    cls_row = class_stats.get(geom.cls_id)
    rel_w_thr = cls_row.rel_width_threshold(config.rel_quantile, config.rel_width_factor) if cls_row else 0.0
    rel_h_thr = cls_row.rel_height_threshold(config.rel_quantile, config.rel_height_factor) if cls_row else 0.0

    if ("left" in sides or "right" in sides) and w_clip_px > 0:
        if w_clip_px < config.abs_min_width_px:
            return True, DropReason.ABS_WIDTH
        if rel_w_thr > 0 and w_clip_px < rel_w_thr:
            return True, DropReason.REL_WIDTH

    if ("top" in sides or "bottom" in sides) and h_clip_px > 0:
        if h_clip_px < config.abs_min_height_px:
            return True, DropReason.ABS_HEIGHT
        if rel_h_thr > 0 and h_clip_px < rel_h_thr:
            return True, DropReason.REL_HEIGHT

    return _global_filters(geom, config, w_px=w_clip_px, h_px=h_clip_px)


def _global_filters(
    geom: BboxGeom,
    config: BboxEdgeFilterConfig,
    *,
    w_px: float | None = None,
    h_px: float | None = None,
) -> tuple[bool, DropReason | None]:
    w = w_px if w_px is not None else geom.w_px
    h = h_px if h_px is not None else geom.h_px
    area = w * h
    if config.min_area_px > 0 and area < config.min_area_px:
        return True, DropReason.MIN_AREA
    if config.max_aspect_ratio is not None and w > 0 and h > 0:
        ratio = max(w / h, h / w)
        if ratio > config.max_aspect_ratio:
            return True, DropReason.ASPECT_RATIO
    return False, None
