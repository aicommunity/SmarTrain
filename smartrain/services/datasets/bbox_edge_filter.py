from __future__ import annotations

from collections import defaultdict
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


EDGE_SIDES_CHOICES = ("any", "horizontal", "vertical", "up", "down", "left", "right")
EDGE_SIDES_ALIASES = {"horisontal": "horizontal", "top": "up", "bottom": "down"}

SIZE_DIMS_CHOICES = ("any", "width", "height")
SIZE_BASELINE_MODE_CHOICES = ("inset", "stable")


def normalize_edge_sides(mode: str) -> str:
    value = str(mode or "any").strip().lower()
    return EDGE_SIDES_ALIASES.get(value, value)


def allowed_filter_sides(mode: str) -> frozenset[str]:
    normalized = normalize_edge_sides(mode)
    mapping: dict[str, frozenset[str]] = {
        "any": frozenset({"left", "right", "top", "bottom"}),
        "horizontal": frozenset({"left", "right"}),
        "vertical": frozenset({"top", "bottom"}),
        "up": frozenset({"top"}),
        "down": frozenset({"bottom"}),
        "left": frozenset({"left"}),
        "right": frozenset({"right"}),
    }
    if normalized not in mapping:
        allowed = ", ".join(EDGE_SIDES_CHOICES)
        raise ValueError(f"Unknown --edge-sides {mode!r}; expected one of: {allowed}")
    return mapping[normalized]


def normalize_size_dims(mode: str) -> str:
    return str(mode or "any").strip().lower()


def allowed_size_dims(mode: str) -> frozenset[str]:
    normalized = normalize_size_dims(mode)
    mapping: dict[str, frozenset[str]] = {
        "any": frozenset({"width", "height"}),
        "width": frozenset({"width"}),
        "height": frozenset({"height"}),
    }
    if normalized not in mapping:
        allowed = ", ".join(SIZE_DIMS_CHOICES)
        raise ValueError(f"Unknown --size-dims {mode!r}; expected one of: {allowed}")
    return mapping[normalized]


def normalize_size_baseline_mode(mode: str) -> str:
    return str(mode or "inset").strip().lower()


def allowed_size_baseline_mode(mode: str) -> str:
    normalized = normalize_size_baseline_mode(mode)
    if normalized not in SIZE_BASELINE_MODE_CHOICES:
        allowed = ", ".join(SIZE_BASELINE_MODE_CHOICES)
        raise ValueError(f"Unknown --size-baseline-mode {mode!r}; expected one of: {allowed}")
    return normalized


@dataclass(frozen=True)
class ContentBounds:
    """Normalized axis-aligned region where objects of a class appear in the dataset."""

    x1: float
    y1: float
    x2: float
    y2: float
    source_frames: int = 0  # per-class mode: number of bbox instances aggregated

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "source_frames": self.source_frames,
        }


@dataclass(frozen=True)
class EmpiricalBoundsMap:
    by_class: dict[int, ContentBounds]
    by_class_format: dict[tuple[int, int, int], ContentBounds]

    def resolve(self, geom: BboxGeom, *, config: BboxEdgeFilterConfig) -> ContentBounds | None:
        if not config.empirical_bounds:
            return None
        if touches_physical_image_edge(geom, config.edge_eps):
            return None
        if config.empirical_by_format:
            key = (int(geom.cls_id), int(geom.img_w), int(geom.img_h))
            found = self.by_class_format.get(key)
            if found is not None:
                return found
        return self.by_class.get(int(geom.cls_id))


def union_geoms_content_bounds(geoms: list[BboxGeom]) -> ContentBounds | None:
    if not geoms:
        return None
    return ContentBounds(
        x1=min(g.x1 for g in geoms),
        y1=min(g.y1 for g in geoms),
        x2=max(g.x2 for g in geoms),
        y2=max(g.y2 for g in geoms),
        source_frames=1,
    )


def merge_content_bounds(bounds: list[ContentBounds]) -> ContentBounds | None:
    if not bounds:
        return None
    return ContentBounds(
        x1=min(b.x1 for b in bounds),
        y1=min(b.y1 for b in bounds),
        x2=max(b.x2 for b in bounds),
        y2=max(b.y2 for b in bounds),
        source_frames=len(bounds),
    )


def percentile_content_bounds(
    geoms: list[BboxGeom],
    *,
    q_low: float,
    q_high: float,
) -> ContentBounds | None:
    if not geoms:
        return None
    x1s = [g.x1 for g in geoms]
    y1s = [g.y1 for g in geoms]
    x2s = [g.x2 for g in geoms]
    y2s = [g.y2 for g in geoms]
    return ContentBounds(
        x1=_percentile(x1s, q_low),
        y1=_percentile(y1s, q_low),
        x2=_percentile(x2s, q_high),
        y2=_percentile(y2s, q_high),
        source_frames=len(geoms),
    )


def touches_physical_image_edge(geom: BboxGeom, edge_eps: float) -> bool:
    if _extends_outside(geom):
        return True
    eps = float(edge_eps)
    return (
        geom.x1 <= eps
        or geom.y1 <= eps
        or geom.x2 >= 1.0 - eps
        or geom.y2 >= 1.0 - eps
    )


def is_empirical_inset_sample(geom: BboxGeom, *, config: BboxEdgeFilterConfig) -> bool:
    return is_baseline_inset(geom, config=config, content_bounds=None)


def collect_empirical_bounds(
    samples: list[tuple[int, BboxGeom]],
    *,
    config: BboxEdgeFilterConfig,
) -> EmpiricalBoundsMap:
    eligible: list[tuple[int, BboxGeom]] = []
    for cls_id, geom in samples:
        if config.empirical_inset_only and not is_empirical_inset_sample(geom, config=config):
            continue
        eligible.append((int(cls_id), geom))

    q = float(config.empirical_percentile)
    q_low = max(0.0, min(q, 0.49))
    q_high = 1.0 - q_low

    by_class_geoms: dict[int, list[BboxGeom]] = defaultdict(list)
    by_format_geoms: dict[tuple[int, int, int], list[BboxGeom]] = defaultdict(list)
    for cls_id, geom in eligible:
        by_class_geoms[cls_id].append(geom)
        if config.empirical_by_format:
            by_format_geoms[(cls_id, int(geom.img_w), int(geom.img_h))].append(geom)

    by_class: dict[int, ContentBounds] = {}
    for cls_id, geoms in by_class_geoms.items():
        bounds = percentile_content_bounds(geoms, q_low=q_low, q_high=q_high)
        if bounds is not None:
            by_class[cls_id] = bounds

    by_class_format: dict[tuple[int, int, int], ContentBounds] = {}
    for key, geoms in by_format_geoms.items():
        bounds = percentile_content_bounds(geoms, q_low=q_low, q_high=q_high)
        if bounds is not None:
            by_class_format[key] = bounds

    return EmpiricalBoundsMap(by_class=by_class, by_class_format=by_class_format)


def summarize_empirical_bounds(
    bounds: EmpiricalBoundsMap,
    *,
    config: BboxEdgeFilterConfig,
    id_to_name: dict[int, str] | None = None,
) -> dict[str, Any]:
    classes: dict[str, Any] = {}
    for cls_id in sorted(bounds.by_class):
        name = (id_to_name or {}).get(cls_id, str(cls_id))
        classes[name] = bounds.by_class[cls_id].to_manifest_dict()

    formats: dict[str, dict[str, Any]] = {}
    for (cls_id, img_w, img_h), row in sorted(bounds.by_class_format.items()):
        fmt = f"{img_w}x{img_h}"
        name = (id_to_name or {}).get(cls_id, str(cls_id))
        formats.setdefault(fmt, {})[name] = row.to_manifest_dict()

    return {
        "mode": "per_class_percentile",
        "percentile": config.empirical_percentile,
        "inset_only": config.empirical_inset_only,
        "by_format": config.empirical_by_format,
        "dual_path_image_edge": True,
        "classes": classes,
        "formats": formats,
    }


def collect_class_content_bounds(samples: list[tuple[int, BboxGeom]]) -> dict[int, ContentBounds]:
    by_class: dict[int, list[BboxGeom]] = defaultdict(list)
    for cls_id, geom in samples:
        by_class[int(cls_id)].append(geom)
    result: dict[int, ContentBounds] = {}
    for cls_id, geoms in by_class.items():
        merged = union_geoms_content_bounds(geoms)
        if merged is not None:
            result[cls_id] = ContentBounds(
                x1=merged.x1,
                y1=merged.y1,
                x2=merged.x2,
                y2=merged.y2,
                source_frames=len(geoms),
            )
    return result


def summarize_class_content_bounds(
    bounds_map: dict[int, ContentBounds],
    *,
    id_to_name: dict[int, str] | None = None,
) -> dict[str, Any] | None:
    if not bounds_map:
        return None
    classes: dict[str, Any] = {}
    for cls_id in sorted(bounds_map):
        row = bounds_map[cls_id]
        name = (id_to_name or {}).get(cls_id, str(cls_id))
        classes[name] = row.to_manifest_dict()
    return {
        "mode": "per_class",
        "classes": classes,
    }


@dataclass(frozen=True)
class BboxEdgeFilterConfig:
    edge_filter: bool = True
    edge_sides: str = "any"
    size_filter: bool = False
    size_dims: str = "any"
    size_baseline_mode: str = "inset"
    size_bulk_split_ratio: float = 0.5
    size_typical_quantile: float = 0.25
    size_by_format: bool = False
    empirical_bounds: bool = False
    empirical_percentile: float = 0.10
    empirical_inset_only: bool = True
    empirical_by_format: bool = True
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


@dataclass
class ClassSizeStats:
    cls_id: int
    cls_name: str = ""
    sample_count: int = 0
    bulk_width_count: int = 0
    bulk_height_count: int = 0
    width_typical: float = 0.0
    height_typical: float = 0.0
    width_threshold: float = 0.0
    height_threshold: float = 0.0
    baseline_mode: str = "stable"
    img_w: int | None = None
    img_h: int | None = None

    def to_manifest_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cls_id": self.cls_id,
            "cls_name": self.cls_name,
            "sample_count": self.sample_count,
            "baseline_mode": self.baseline_mode,
            "bulk_width_count": self.bulk_width_count,
            "bulk_height_count": self.bulk_height_count,
            "width_typical_px": self.width_typical,
            "height_typical_px": self.height_typical,
            "width_threshold_px": self.width_threshold,
            "height_threshold_px": self.height_threshold,
        }
        if self.img_w is not None and self.img_h is not None:
            out["img_w"] = self.img_w
            out["img_h"] = self.img_h
        return out


@dataclass(frozen=True)
class SizeStatsMap:
    by_class: dict[int, ClassSizeStats]
    by_class_format: dict[tuple[int, int, int], ClassSizeStats]

    def resolve(self, geom: BboxGeom, *, config: BboxEdgeFilterConfig) -> ClassSizeStats | None:
        if config.size_by_format:
            key = (int(geom.cls_id), int(geom.img_w), int(geom.img_h))
            found = self.by_class_format.get(key)
            if found is not None:
                return found
        return self.by_class.get(int(geom.cls_id))


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


def _stable_typical_threshold(
    values: list[float],
    *,
    split_ratio: float,
    typical_quantile: float,
    rel_factor: float,
    min_baseline_samples: int,
) -> tuple[float, float, int]:
    if not values:
        return 0.0, 0.0, 0
    med = _percentile(values, 0.50)
    split = float(split_ratio) * med
    bulk = [float(v) for v in values if float(v) >= split]
    if len(bulk) < min_baseline_samples:
        bulk = [float(v) for v in values]
    typical = _percentile(bulk, typical_quantile)
    threshold = float(rel_factor * typical) if typical > 0 else 0.0
    return typical, threshold, len(bulk)


def _build_class_size_stats(
    cls_id: int,
    widths: list[float],
    heights: list[float],
    *,
    config: BboxEdgeFilterConfig,
    cls_name: str = "",
    img_w: int | None = None,
    img_h: int | None = None,
) -> ClassSizeStats:
    w_typ, w_thr, w_bulk = _stable_typical_threshold(
        widths,
        split_ratio=config.size_bulk_split_ratio,
        typical_quantile=config.size_typical_quantile,
        rel_factor=config.rel_width_factor,
        min_baseline_samples=config.min_baseline_samples,
    )
    h_typ, h_thr, h_bulk = _stable_typical_threshold(
        heights,
        split_ratio=config.size_bulk_split_ratio,
        typical_quantile=config.size_typical_quantile,
        rel_factor=config.rel_height_factor,
        min_baseline_samples=config.min_baseline_samples,
    )
    return ClassSizeStats(
        cls_id=cls_id,
        cls_name=cls_name,
        sample_count=len(widths),
        bulk_width_count=w_bulk,
        bulk_height_count=h_bulk,
        width_typical=w_typ,
        height_typical=h_typ,
        width_threshold=w_thr,
        height_threshold=h_thr,
        baseline_mode="stable",
        img_w=img_w,
        img_h=img_h,
    )


def collect_size_baseline_stats(
    samples: list[tuple[int, BboxGeom]],
    *,
    config: BboxEdgeFilterConfig,
    id_to_name: dict[int, str] | None = None,
) -> SizeStatsMap | None:
    if not config.size_filter or normalize_size_baseline_mode(config.size_baseline_mode) != "stable":
        return None

    by_class_w: dict[int, list[float]] = defaultdict(list)
    by_class_h: dict[int, list[float]] = defaultdict(list)
    by_format_w: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    by_format_h: dict[tuple[int, int, int], list[float]] = defaultdict(list)

    for cls_id, geom in samples:
        cid = int(cls_id)
        by_class_w[cid].append(geom.w_px)
        by_class_h[cid].append(geom.h_px)
        if config.size_by_format:
            key = (cid, int(geom.img_w), int(geom.img_h))
            by_format_w[key].append(geom.w_px)
            by_format_h[key].append(geom.h_px)

    by_class: dict[int, ClassSizeStats] = {}
    for cls_id in sorted(set(by_class_w) | set(by_class_h)):
        name = (id_to_name or {}).get(cls_id, str(cls_id))
        by_class[cls_id] = _build_class_size_stats(
            cls_id,
            by_class_w.get(cls_id, []),
            by_class_h.get(cls_id, []),
            config=config,
            cls_name=name,
        )

    by_class_format: dict[tuple[int, int, int], ClassSizeStats] = {}
    if config.size_by_format:
        for key in sorted(set(by_format_w) | set(by_format_h)):
            cls_id, img_w, img_h = key
            name = (id_to_name or {}).get(cls_id, str(cls_id))
            by_class_format[key] = _build_class_size_stats(
                cls_id,
                by_format_w.get(key, []),
                by_format_h.get(key, []),
                config=config,
                cls_name=name,
                img_w=img_w,
                img_h=img_h,
            )

    return SizeStatsMap(by_class=by_class, by_class_format=by_class_format)


def summarize_size_baseline_stats(
    size_stats_map: SizeStatsMap | None,
    *,
    config: BboxEdgeFilterConfig,
    id_to_name: dict[int, str] | None = None,
) -> dict[str, Any] | None:
    if size_stats_map is None:
        return None
    classes: dict[str, Any] = {}
    for cls_id in sorted(size_stats_map.by_class):
        row = size_stats_map.by_class[cls_id]
        name = (id_to_name or {}).get(cls_id, str(cls_id))
        classes[name] = row.to_manifest_dict()
    result: dict[str, Any] = {
        "mode": "stable",
        "bulk_split_ratio": config.size_bulk_split_ratio,
        "typical_quantile": config.size_typical_quantile,
        "by_format": config.size_by_format,
        "classes": classes,
    }
    if config.size_by_format and size_stats_map.by_class_format:
        formats: dict[str, Any] = {}
        for (cls_id, img_w, img_h), row in sorted(size_stats_map.by_class_format.items()):
            name = (id_to_name or {}).get(cls_id, str(cls_id))
            formats[f"{name}@{img_w}x{img_h}"] = row.to_manifest_dict()
        result["formats"] = formats
    return result


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


def _reference_rect(content_bounds: ContentBounds | None) -> tuple[float, float, float, float]:
    if content_bounds is None:
        return 0.0, 0.0, 1.0, 1.0
    return content_bounds.x1, content_bounds.y1, content_bounds.x2, content_bounds.y2


def is_baseline_inset(
    geom: BboxGeom,
    *,
    config: BboxEdgeFilterConfig,
    content_bounds: ContentBounds | None = None,
) -> bool:
    margin = _margin_norm(
        margin_norm=config.baseline_inset_margin,
        margin_px=config.baseline_inset_margin_px,
        img_w=geom.img_w,
        img_h=geom.img_h,
    )
    rx1, ry1, rx2, ry2 = _reference_rect(content_bounds if config.empirical_bounds else None)
    return (
        geom.x1 >= rx1 + margin
        and geom.y1 >= ry1 + margin
        and geom.x2 <= rx2 - margin
        and geom.y2 <= ry2 - margin
    )



def _extends_outside(geom: BboxGeom) -> bool:
    return geom.x1 < 0.0 or geom.y1 < 0.0 or geom.x2 > 1.0 or geom.y2 > 1.0


def _extends_outside_selected(geom: BboxGeom, allowed: frozenset[str]) -> bool:
    if "left" in allowed and geom.x1 < 0.0:
        return True
    if "right" in allowed and geom.x2 > 1.0:
        return True
    if "top" in allowed and geom.y1 < 0.0:
        return True
    if "bottom" in allowed and geom.y2 > 1.0:
        return True
    return False


def in_filter_zone(
    geom: BboxGeom,
    *,
    config: BboxEdgeFilterConfig,
    content_bounds: ContentBounds | None = None,
) -> bool:
    allowed = allowed_filter_sides(config.edge_sides)
    proximity = _margin_norm(
        margin_norm=config.resolved_filter_proximity_margin(),
        margin_px=config.baseline_inset_margin_px,
        img_w=geom.img_w,
        img_h=geom.img_h,
    )
    bounds = content_bounds
    sides = _edge_sides(geom, proximity_margin=proximity, edge_eps=config.edge_eps, content_bounds=bounds)
    if sides & allowed:
        return True
    return _extends_outside_selected(geom, allowed)


def _clip_xyxy(geom: BboxGeom) -> tuple[float, float, float, float]:
    x1 = min(1.0, max(0.0, geom.x1))
    y1 = min(1.0, max(0.0, geom.y1))
    x2 = min(1.0, max(0.0, geom.x2))
    y2 = min(1.0, max(0.0, geom.y2))
    return x1, y1, x2, y2


def _edge_sides(
    geom: BboxGeom,
    *,
    proximity_margin: float,
    edge_eps: float,
    content_bounds: ContentBounds | None = None,
) -> set[str]:
    if content_bounds is None:
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

    rx1, ry1, rx2, ry2 = _reference_rect(content_bounds)
    sides = set()
    if _near_ref_edge(geom.x1, geom.x2, rx1, proximity_margin, edge_eps):
        sides.add("left")
    if _near_ref_edge(geom.x2, geom.x1, rx2, proximity_margin, edge_eps):
        sides.add("right")
    if _near_ref_edge(geom.y1, geom.y2, ry1, proximity_margin, edge_eps):
        sides.add("top")
    if _near_ref_edge(geom.y2, geom.y1, ry2, proximity_margin, edge_eps):
        sides.add("bottom")
    return sides


def _near_ref_edge(edge_coord: float, other_coord: float, ref: float, proximity_margin: float, edge_eps: float) -> bool:
    return edge_coord <= ref + max(proximity_margin, edge_eps) and other_coord >= ref - max(proximity_margin, edge_eps)


def collect_baseline_stats(
    samples: list[tuple[int, BboxGeom]],
    *,
    config: BboxEdgeFilterConfig,
    id_to_name: dict[int, str] | None = None,
    content_bounds: ContentBounds | None = None,
) -> dict[int, ClassBboxStats]:
    stats: dict[int, ClassBboxStats] = {}
    for cls_id, geom in samples:
        row = stats.setdefault(
            cls_id,
            ClassBboxStats(cls_id=cls_id, cls_name=(id_to_name or {}).get(cls_id, str(cls_id))),
        )
        if is_baseline_inset(geom, config=config, content_bounds=content_bounds):
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
                if in_filter_zone(geom, config=config, content_bounds=content_bounds):
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
    content_bounds: ContentBounds | None = None,
) -> tuple[bool, DropReason | None]:
    if not config.edge_filter:
        return _global_filters(geom, config)

    bounds = content_bounds
    if not in_filter_zone(geom, config=config, content_bounds=bounds):
        return _global_filters(geom, config)

    proximity = _margin_norm(
        margin_norm=config.resolved_filter_proximity_margin(),
        margin_px=config.baseline_inset_margin_px,
        img_w=geom.img_w,
        img_h=geom.img_h,
    )
    allowed = allowed_filter_sides(config.edge_sides)
    sides = _edge_sides(geom, proximity_margin=proximity, edge_eps=config.edge_eps, content_bounds=bounds) & allowed
    if not sides:
        return _global_filters(geom, config)
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


def should_drop_small_bbox(
    geom: BboxGeom,
    *,
    config: BboxEdgeFilterConfig,
    class_stats: dict[int, ClassBboxStats],
    size_stats_map: SizeStatsMap | None = None,
) -> tuple[bool, DropReason | None]:
    if not config.size_filter:
        return False, None

    dims = allowed_size_dims(config.size_dims)
    cls_row = class_stats.get(geom.cls_id)
    size_row = size_stats_map.resolve(geom, config=config) if size_stats_map is not None else None

    if normalize_size_baseline_mode(config.size_baseline_mode) == "stable" and size_row is not None:
        rel_w_thr = size_row.width_threshold
        rel_h_thr = size_row.height_threshold
    else:
        rel_w_thr = cls_row.rel_width_threshold(config.rel_quantile, config.rel_width_factor) if cls_row else 0.0
        rel_h_thr = cls_row.rel_height_threshold(config.rel_quantile, config.rel_height_factor) if cls_row else 0.0

    w_px = geom.w_px
    h_px = geom.h_px

    if "width" in dims and w_px > 0:
        if w_px < config.abs_min_width_px:
            return True, DropReason.ABS_WIDTH
        if rel_w_thr > 0 and w_px < rel_w_thr:
            return True, DropReason.REL_WIDTH

    if "height" in dims and h_px > 0:
        if h_px < config.abs_min_height_px:
            return True, DropReason.ABS_HEIGHT
        if rel_h_thr > 0 and h_px < rel_h_thr:
            return True, DropReason.REL_HEIGHT

    return False, None


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
