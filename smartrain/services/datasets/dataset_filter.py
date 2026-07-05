from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import (
    prompt_choice,
    prompt_multi_choice_csv,
    prompt_text,
    prompt_yes_no,
)
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.services.datasets.bbox_edge_filter import (
    BboxEdgeFilterConfig,
    BboxGeom,
    ClassBboxStats,
    ContentBounds,
    EmpiricalBoundsMap,
    EDGE_SIDES_CHOICES,
    allowed_filter_sides,
    collect_baseline_stats,
    collect_empirical_bounds,
    normalize_edge_sides,
    should_drop_bbox,
    bbox_geom_from_label,
    summarize_empirical_bounds,
)
from smartrain.services.datasets.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.services.datasets.dataset_cli_catalog import (
    EMPTY_DATASETS_INFO_MESSAGE,
    load_datasets_catalog,
    sorted_class_names_for_dataset,
    try_prompt_dataset_interactive,
)
from smartrain.services.datasets.dataset_cli_common import update_datasets_sidecar
from smartrain.services.datasets.dataset_hash import calculate_dataset_hash
from smartrain.services.datasets.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.services.datasets.yolo_labels import YoloLabel, read_yolo_labels, serialize_yolo_labels

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FILTER_AUDIT_ROOT = "_filter_audit"
FILTER_AUDIT_DROPPED_IMAGES = "dropped_images"
FILTER_AUDIT_REMOVED_LABELS = "removed_labels"


@dataclass
class LabelPairItem:
    split: str
    image_path: str
    label_path: str
    rel_image: str
    rel_label: str


@dataclass
class FilterForecast:
    baseline_stats: dict[int, ClassBboxStats]
    empirical_bounds_map: EmpiricalBoundsMap | None = None
    empirical_bounds_summary: dict[str, Any] | None = None
    removed_by_class: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    removed_by_reason: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    would_drop: int = 0
    would_keep: int = 0
    instances_before: int = 0
    images_with_drop: int = 0
    segment_proxy_count: int = 0


def build_filter_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Filter edge-truncated YOLO annotations into a new dataset")
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (aka {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Source dataset key from datasets_info.json")
    p.add_argument("--output-name", type=str, default=None, help="Output dataset name (default <dataset>_fltd)")
    p.add_argument("--classes", type=str, default=None, help="Comma-separated class names to filter (default all)")
    p.add_argument(
        "--edge-sides",
        type=str,
        default="any",
        choices=EDGE_SIDES_CHOICES,
        help="Which image edges to filter bbox against: any, horizontal, vertical, up, down, left, right",
    )
    p.add_argument("--edge-filter", dest="edge_filter", action="store_true", default=True)
    p.add_argument("--no-edge-filter", dest="edge_filter", action="store_false")
    p.add_argument("--baseline-inset-margin", type=float, default=0.01)
    p.add_argument("--baseline-inset-margin-px", type=float, default=None)
    p.add_argument("--edge-eps", type=float, default=0.002)
    p.add_argument("--filter-proximity-margin", type=float, default=None)
    p.add_argument(
        "--empirical-bounds",
        action="store_true",
        help=(
            "Dual-path edge filter: image borders for edge-touching bbox, "
            "per-class percentile hull for inset bbox (see --empirical-percentile)"
        ),
    )
    p.add_argument(
        "--empirical-percentile",
        type=float,
        default=0.10,
        help="Lower/upper percentile for empirical class hull (default 0.10 → p10–p90)",
    )
    p.add_argument("--empirical-inset-only", dest="empirical_inset_only", action="store_true", default=True)
    p.add_argument("--no-empirical-inset-only", dest="empirical_inset_only", action="store_false")
    p.add_argument("--empirical-by-format", dest="empirical_by_format", action="store_true", default=True)
    p.add_argument("--no-empirical-by-format", dest="empirical_by_format", action="store_false")
    p.add_argument("--abs-min-width-px", type=float, default=8.0)
    p.add_argument("--abs-min-height-px", type=float, default=8.0)
    p.add_argument("--rel-quantile", type=float, default=0.10)
    p.add_argument("--rel-width-factor", type=float, default=0.85)
    p.add_argument("--rel-height-factor", type=float, default=0.85)
    p.add_argument("--min-visibility", type=float, default=0.0)
    p.add_argument("--min-area-px", type=float, default=0.0)
    p.add_argument("--max-aspect-ratio", type=float, default=None)
    p.add_argument("--drop-images", action="store_true", help="Remove entire image if any bbox matches filter")
    p.add_argument(
        "--drop-background",
        action="store_true",
        help="Remove images that had no annotations in the source dataset (background frames)",
    )
    p.add_argument("--prune-empty", dest="prune_empty", action="store_true", default=True)
    p.add_argument("--no-prune-empty", dest="prune_empty", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stats-only", action="store_true")
    p.add_argument("--yes", "-y", action="store_true", help="Skip interactive confirmation after preview")
    return p


def _config_from_args(args: argparse.Namespace) -> BboxEdgeFilterConfig:
    edge_sides = normalize_edge_sides(args.edge_sides)
    allowed_filter_sides(edge_sides)
    return BboxEdgeFilterConfig(
        edge_filter=bool(args.edge_filter),
        edge_sides=edge_sides,
        empirical_bounds=bool(args.empirical_bounds),
        empirical_percentile=float(args.empirical_percentile),
        empirical_inset_only=bool(args.empirical_inset_only),
        empirical_by_format=bool(args.empirical_by_format),
        baseline_inset_margin=float(args.baseline_inset_margin),
        baseline_inset_margin_px=args.baseline_inset_margin_px,
        edge_eps=float(args.edge_eps),
        filter_proximity_margin=args.filter_proximity_margin,
        abs_min_width_px=float(args.abs_min_width_px),
        abs_min_height_px=float(args.abs_min_height_px),
        rel_quantile=float(args.rel_quantile),
        rel_width_factor=float(args.rel_width_factor),
        rel_height_factor=float(args.rel_height_factor),
        min_visibility=float(args.min_visibility),
        min_area_px=float(args.min_area_px),
        max_aspect_ratio=args.max_aspect_ratio,
    )


def _resolve_image_for_label(label_path: Path, img_dir: Path) -> Path | None:
    rel = label_path.relative_to(label_path.parent)
    stem = label_path.stem
    for ext in IMAGE_EXTS:
        candidate = img_dir / rel.parent / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
        candidate2 = img_dir / f"{stem}{ext}"
        if candidate2.is_file():
            return candidate2
    return None


def _list_items(src_root: str, structure: str, entry: dict[str, Any], dataset_name: str, tmp_root: str) -> list[LabelPairItem]:
    items: list[LabelPairItem] = []
    buckets = iter_image_label_buckets(
        src_root,
        structure,
        entry,
        dataset_name=dataset_name,
        temp_root=tmp_root,
        exclude_test=False,
    )
    root = Path(src_root)
    seen_images: set[str] = set()
    for img_dir, lbl_dir in buckets:
        img_path = Path(img_dir)
        lbl_path = Path(lbl_dir)
        split = img_path.parent.name if img_path.parent.name in {"train", "val", "valid", "test"} else "train"
        for lf in sorted(lbl_path.rglob("*.txt")):
            image = _resolve_image_for_label(lf, img_path)
            if image is None:
                continue
            rel_img = str(image.relative_to(root))
            rel_lbl = str(lf.relative_to(root))
            seen_images.add(rel_img)
            items.append(
                LabelPairItem(
                    split=split,
                    image_path=str(image),
                    label_path=str(lf),
                    rel_image=rel_img,
                    rel_label=rel_lbl,
                )
            )
        for img_file in sorted(img_path.rglob("*")):
            if not img_file.is_file() or img_file.suffix.lower() not in IMAGE_EXTS:
                continue
            rel_img = str(img_file.relative_to(root))
            if rel_img in seen_images:
                continue
            rel = img_file.relative_to(img_path)
            lbl = lbl_path / rel.parent / f"{img_file.stem}.txt"
            rel_lbl = str(lbl.relative_to(root))
            items.append(
                LabelPairItem(
                    split=split,
                    image_path=str(img_file),
                    label_path=str(lbl),
                    rel_image=rel_img,
                    rel_label=rel_lbl,
                )
            )
    return items


def _image_size(path: str, cache: dict[str, tuple[int, int]]) -> tuple[int, int]:
    if path in cache:
        return cache[path]
    with Image.open(path) as im:
        size = (int(im.width), int(im.height))
    cache[path] = size
    return size


def _selected_class_ids(entry: dict[str, Any], classes_csv: str | None) -> set[int] | None:
    class_map = entry.get("classes", {})
    if not isinstance(class_map, dict) or not classes_csv:
        return None
    wanted = {x.strip() for x in classes_csv.split(",") if x.strip()}
    if not wanted:
        return None
    return {int(v) for k, v in class_map.items() if str(k) in wanted}


def _gather_samples(
    items: list[LabelPairItem],
    *,
    config: BboxEdgeFilterConfig,
    class_filter: set[int] | None,
    size_cache: dict[str, tuple[int, int]],
) -> tuple[list[tuple[int, BboxGeom]], int]:
    samples: list[tuple[int, BboxGeom]] = []
    segment_proxy = 0
    for item in items:
        if not os.path.isfile(item.label_path):
            continue
        iw, ih = _image_size(item.image_path, size_cache)
        for lb in read_yolo_labels(item.label_path):
            if class_filter is not None and int(getattr(lb, "cls_id", -1)) not in class_filter:
                continue
            geom = bbox_geom_from_label(lb, img_w=iw, img_h=ih)
            if geom is None:
                continue
            if geom.is_segment_proxy:
                segment_proxy += 1
            samples.append((geom.cls_id, geom))
    return samples, segment_proxy


def _resolve_empirical_bounds(
    geom: BboxGeom,
    *,
    config: BboxEdgeFilterConfig,
    empirical_bounds_map: EmpiricalBoundsMap | None,
) -> ContentBounds | None:
    if empirical_bounds_map is None:
        return None
    return empirical_bounds_map.resolve(geom, config=config)


def _forecast_filter(
    items: list[LabelPairItem],
    *,
    config: BboxEdgeFilterConfig,
    class_filter: set[int] | None,
    id_to_name: dict[int, str],
    size_cache: dict[str, tuple[int, int]],
) -> FilterForecast:
    samples, segment_proxy = _gather_samples(items, config=config, class_filter=class_filter, size_cache=size_cache)
    empirical_bounds_map = collect_empirical_bounds(samples, config=config) if config.empirical_bounds else None
    bounds_summary = (
        summarize_empirical_bounds(empirical_bounds_map, config=config, id_to_name=id_to_name)
        if empirical_bounds_map is not None
        else None
    )
    baseline = collect_baseline_stats(
        samples,
        config=config,
        id_to_name=id_to_name,
        content_bounds=None,
    )
    forecast = FilterForecast(
        baseline_stats=baseline,
        empirical_bounds_map=empirical_bounds_map,
        empirical_bounds_summary=bounds_summary,
        segment_proxy_count=segment_proxy,
    )
    forecast.instances_before = len(samples)

    for item in items:
        if not os.path.isfile(item.label_path):
            continue
        iw, ih = _image_size(item.image_path, size_cache)
        image_has_drop = False
        for lb in read_yolo_labels(item.label_path):
            if class_filter is not None and int(getattr(lb, "cls_id", -1)) not in class_filter:
                forecast.would_keep += 1
                continue
            geom = bbox_geom_from_label(lb, img_w=iw, img_h=ih)
            if geom is None:
                continue
            drop, reason = should_drop_bbox(
                geom,
                config=config,
                class_stats=baseline,
                content_bounds=_resolve_empirical_bounds(
                    geom, config=config, empirical_bounds_map=empirical_bounds_map
                ),
            )
            if drop:
                forecast.would_drop += 1
                forecast.removed_by_class[geom.cls_id] += 1
                if reason is not None:
                    forecast.removed_by_reason[reason.value] += 1
                image_has_drop = True
            else:
                forecast.would_keep += 1
        if image_has_drop:
            forecast.images_with_drop += 1
    return forecast


def _print_forecast_table(forecast: FilterForecast, *, id_to_name: dict[int, str]) -> None:
    print("\n[INFO] Filter preview")
    print(f"  instances_before={forecast.instances_before} would_drop={forecast.would_drop} would_keep={forecast.would_keep}")
    if forecast.segment_proxy_count:
        print(f"  segment_proxy_count={forecast.segment_proxy_count} (bbox envelope used)")
    if forecast.empirical_bounds_summary is not None:
        s = forecast.empirical_bounds_summary
        print(
            f"  empirical_bounds=per_class_percentile p={s['percentile']:.2f} "
            f"inset_only={s['inset_only']} by_format={s['by_format']}"
        )
        for name, row in s.get("classes", {}).items():
            print(
                f"    {name}: x1={row['x1']:.4f} x2={row['x2']:.4f} "
                f"y1={row['y1']:.4f} y2={row['y2']:.4f} instances={row['source_frames']}"
            )
    print("  per-class:")
    cls_ids = sorted(set(forecast.baseline_stats.keys()) | set(forecast.removed_by_class.keys()))
    for cid in cls_ids:
        row = forecast.baseline_stats.get(cid)
        name = id_to_name.get(cid, str(cid))
        if row is None:
            print(f"    {name}: would_drop={forecast.removed_by_class.get(cid, 0)}")
            continue
        fb = f" fallback={row.baseline_fallback}" if row.baseline_fallback else ""
        print(
            f"    {name}: baseline_eligible={row.baseline_eligible_count} "
            f"excluded_near_edge={row.baseline_excluded_near_edge_count} "
            f"would_drop={forecast.removed_by_class.get(cid, 0)} "
            f"w_p10={row.width_p10:.1f}px h_p10={row.height_p10:.1f}px{fb}"
        )
    if forecast.removed_by_reason:
        print("  removed_by_reason:", dict(forecast.removed_by_reason))


def _filter_labels_for_item(
    item: LabelPairItem,
    *,
    config: BboxEdgeFilterConfig,
    class_stats: dict[int, ClassBboxStats],
    class_filter: set[int] | None,
    size_cache: dict[str, tuple[int, int]],
    empirical_bounds_map: EmpiricalBoundsMap | None = None,
) -> tuple[list[YoloLabel], list[YoloLabel], bool, dict[str, int], dict[int, int]]:
    if not os.path.isfile(item.label_path):
        return [], [], False, {}, {}
    iw, ih = _image_size(item.image_path, size_cache)
    kept: list[YoloLabel] = []
    removed: list[YoloLabel] = []
    had_drop = False
    reason_counts: dict[str, int] = defaultdict(int)
    class_counts: dict[int, int] = defaultdict(int)
    for lb in read_yolo_labels(item.label_path):
        if class_filter is not None and int(getattr(lb, "cls_id", -1)) not in class_filter:
            kept.append(lb)
            continue
        geom = bbox_geom_from_label(lb, img_w=iw, img_h=ih)
        if geom is None:
            kept.append(lb)
            continue
        drop, reason = should_drop_bbox(
            geom,
            config=config,
            class_stats=class_stats,
            content_bounds=_resolve_empirical_bounds(
                geom, config=config, empirical_bounds_map=empirical_bounds_map
            ),
        )
        if drop:
            had_drop = True
            removed.append(lb)
            class_counts[geom.cls_id] += 1
            if reason is not None:
                reason_counts[reason.value] += 1
            continue
        kept.append(lb)
    return kept, removed, had_drop, dict(reason_counts), dict(class_counts)


def _originally_unlabeled(item: LabelPairItem) -> bool:
    if not os.path.isfile(item.label_path):
        return True
    return len(read_yolo_labels(item.label_path)) == 0


def _copy_background_image(out_dir: str, item: LabelPairItem, *, copied_images: int) -> int:
    out_img = os.path.join(out_dir, item.rel_image)
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    shutil.copy2(item.image_path, out_img)
    return copied_images + 1


def _audit_dropped_image_paths(out_dir: str, rel_image: str, rel_label: str) -> tuple[str, str]:
    base = os.path.join(out_dir, FILTER_AUDIT_ROOT, FILTER_AUDIT_DROPPED_IMAGES)
    return os.path.join(base, rel_image), os.path.join(base, rel_label)


def _audit_removed_label_path(out_dir: str, rel_label: str) -> str:
    return os.path.join(out_dir, FILTER_AUDIT_ROOT, FILTER_AUDIT_REMOVED_LABELS, rel_label)


def _archive_dropped_image_pair(
    out_dir: str,
    item: LabelPairItem,
    *,
    archive_reason: str,
    archive_reasons: dict[str, int],
) -> None:
    audit_img, audit_lbl = _audit_dropped_image_paths(out_dir, item.rel_image, item.rel_label)
    os.makedirs(os.path.dirname(audit_img), exist_ok=True)
    os.makedirs(os.path.dirname(audit_lbl), exist_ok=True)
    shutil.copy2(item.image_path, audit_img)
    if os.path.isfile(item.label_path):
        shutil.copy2(item.label_path, audit_lbl)
    archive_reasons[archive_reason] = archive_reasons.get(archive_reason, 0) + 1


def _archive_removed_labels(out_dir: str, item: LabelPairItem, removed: list[YoloLabel]) -> bool:
    if not removed:
        return False
    audit_lbl = _audit_removed_label_path(out_dir, item.rel_label)
    os.makedirs(os.path.dirname(audit_lbl), exist_ok=True)
    Path(audit_lbl).write_text(serialize_yolo_labels(removed), encoding="utf-8")
    return True


def _copy_and_filter(
    out_dir: str,
    items: list[LabelPairItem],
    *,
    config: BboxEdgeFilterConfig,
    class_stats: dict[int, ClassBboxStats],
    class_filter: set[int] | None,
    drop_images: bool,
    prune_empty: bool,
    drop_background: bool,
    size_cache: dict[str, tuple[int, int]],
    empirical_bounds_map: EmpiricalBoundsMap | None = None,
    empirical_bounds_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    removed_instances = 0
    kept_instances = 0
    images_dropped = 0
    images_pruned_empty = 0
    background_images_kept = 0
    background_images_dropped = 0
    removed_by_class: dict[int, int] = defaultdict(int)
    removed_by_reason: dict[str, int] = defaultdict(int)
    copied_images = 0
    audit_dropped_image_pairs = 0
    audit_removed_label_files = 0
    audit_removed_label_instances = 0
    audit_archive_reasons: dict[str, int] = {}

    for item in items:
        out_img = os.path.join(out_dir, item.rel_image)
        out_lbl = os.path.join(out_dir, item.rel_label)
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        os.makedirs(os.path.dirname(out_lbl), exist_ok=True)

        if not os.path.isfile(item.label_path):
            if drop_background:
                _archive_dropped_image_pair(
                    out_dir,
                    item,
                    archive_reason="background",
                    archive_reasons=audit_archive_reasons,
                )
                background_images_dropped += 1
                audit_dropped_image_pairs += 1
            else:
                copied_images = _copy_background_image(out_dir, item, copied_images=copied_images)
                background_images_kept += 1
            continue

        originally_bg = _originally_unlabeled(item)

        kept_labels, removed_labels, had_drop, item_reasons, item_class_counts = _filter_labels_for_item(
            item,
            config=config,
            class_stats=class_stats,
            class_filter=class_filter,
            size_cache=size_cache,
            empirical_bounds_map=empirical_bounds_map,
        )
        for k, v in item_reasons.items():
            removed_by_reason[k] += v
        for k, v in item_class_counts.items():
            removed_by_class[k] += v

        orig_count = len(read_yolo_labels(item.label_path))
        kept_count = len(kept_labels)
        removed_instances += max(0, orig_count - kept_count)
        kept_instances += kept_count

        if drop_images and had_drop:
            _archive_dropped_image_pair(
                out_dir,
                item,
                archive_reason="drop_images",
                archive_reasons=audit_archive_reasons,
            )
            images_dropped += 1
            audit_dropped_image_pairs += 1
            continue

        if prune_empty and kept_count == 0:
            if originally_bg:
                if drop_background:
                    _archive_dropped_image_pair(
                        out_dir,
                        item,
                        archive_reason="background",
                        archive_reasons=audit_archive_reasons,
                    )
                    background_images_dropped += 1
                    audit_dropped_image_pairs += 1
                else:
                    copied_images = _copy_background_image(out_dir, item, copied_images=copied_images)
                    background_images_kept += 1
                continue
            _archive_dropped_image_pair(
                out_dir,
                item,
                archive_reason="pruned_empty",
                archive_reasons=audit_archive_reasons,
            )
            images_pruned_empty += 1
            audit_dropped_image_pairs += 1
            continue

        if _archive_removed_labels(out_dir, item, removed_labels):
            audit_removed_label_files += 1
            audit_removed_label_instances += len(removed_labels)

        shutil.copy2(item.image_path, out_img)
        copied_images += 1
        Path(out_lbl).write_text(serialize_yolo_labels(kept_labels), encoding="utf-8")

    return {
        "copied_images": copied_images,
        "removed_instances": removed_instances,
        "kept_instances": kept_instances,
        "images_dropped": images_dropped,
        "images_pruned_empty": images_pruned_empty,
        "background_images_kept": background_images_kept,
        "background_images_dropped": background_images_dropped,
        "empirical_content_bounds": empirical_bounds_summary,
        "removed_by_class": {str(k): int(v) for k, v in removed_by_class.items()},
        "removed_by_reason": dict(removed_by_reason),
        "audit": {
            "root": FILTER_AUDIT_ROOT,
            "dropped_images_dir": f"{FILTER_AUDIT_ROOT}/{FILTER_AUDIT_DROPPED_IMAGES}",
            "removed_labels_dir": f"{FILTER_AUDIT_ROOT}/{FILTER_AUDIT_REMOVED_LABELS}",
            "excluded_from_training": True,
            "dropped_image_pairs": audit_dropped_image_pairs,
            "removed_label_files": audit_removed_label_files,
            "removed_label_instances": audit_removed_label_instances,
            "archive_reasons": audit_archive_reasons,
        },
    }


def _write_data_yaml(out_dir: str, class_map: dict[str, Any]) -> None:
    names = [k for k, _ in sorted(((str(k), int(v)) for k, v in class_map.items()), key=lambda kv: kv[1])]
    val_rel = "valid/images" if (Path(out_dir) / "valid" / "images").is_dir() else "val/images"
    Path(out_dir, "data.yaml").write_text(
        f"train: train/images\nval: {val_rel}\ntest: test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )


def _write_filter_manifest(out_dir: str, *, parameters: dict[str, Any], forecast: FilterForecast, stats_after: dict[str, Any]) -> str:
    payload = {
        "parameters": parameters,
        "class_baseline_stats": {str(k): v.to_manifest_dict() for k, v in forecast.baseline_stats.items()},
        "forecast": {
            "instances_before": forecast.instances_before,
            "would_drop": forecast.would_drop,
            "would_keep": forecast.would_keep,
            "removed_by_class": {str(k): int(v) for k, v in forecast.removed_by_class.items()},
            "removed_by_reason": dict(forecast.removed_by_reason),
            "images_with_drop": forecast.images_with_drop,
            "segment_proxy_count": forecast.segment_proxy_count,
        },
        "stats_after": stats_after,
    }
    out_path = os.path.join(out_dir, "filter_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _interactive_fill(args: argparse.Namespace, dataset_names: list[str], catalog: dict[str, Any]) -> None:
    print("[INFO] Interactive filter mode")
    args.dataset = prompt_choice("Dataset", dataset_names, default=(args.dataset or dataset_names[0]))
    class_names = sorted_class_names_for_dataset(catalog, str(args.dataset))
    if class_names:
        picked = prompt_multi_choice_csv("Classes (--classes; empty=all)", class_names, default_values=[])
        args.classes = ",".join(picked) if picked else None
    args.output_name = prompt_text("Output dataset name (empty=auto)", default=(args.output_name or "")).strip() or None
    args.edge_sides = prompt_choice(
        "Edge sides to filter (--edge-sides)",
        list(EDGE_SIDES_CHOICES),
        default=str(getattr(args, "edge_sides", "any")),
    )

    print("[INFO] Block: baseline margins")
    args.empirical_bounds = prompt_yes_no(
        "--empirical-bounds (dual-path percentile class hull)?",
        default=bool(args.empirical_bounds),
    )
    if args.empirical_bounds:
        args.empirical_percentile = float(
            prompt_text("--empirical-percentile", default=str(args.empirical_percentile)).strip()
            or str(args.empirical_percentile)
        )
        args.empirical_inset_only = prompt_yes_no(
            "--empirical-inset-only (aggregate only inset samples)?",
            default=bool(args.empirical_inset_only),
        )
        args.empirical_by_format = prompt_yes_no(
            "--empirical-by-format (separate hull per image resolution)?",
            default=bool(args.empirical_by_format),
        )
    args.baseline_inset_margin = float(
        prompt_text("--baseline-inset-margin", default=str(args.baseline_inset_margin)).strip()
        or str(args.baseline_inset_margin)
    )
    use_px = prompt_yes_no("Set baseline inset margin in pixels?", default=args.baseline_inset_margin_px is not None)
    if use_px:
        args.baseline_inset_margin_px = float(
            prompt_text("--baseline-inset-margin-px", default=str(args.baseline_inset_margin_px or 10)).strip() or "10"
        )
    else:
        args.baseline_inset_margin_px = None
    same_prox = prompt_yes_no("Use same value for --filter-proximity-margin?", default=args.filter_proximity_margin is None)
    if same_prox:
        args.filter_proximity_margin = None
    else:
        args.filter_proximity_margin = float(
            prompt_text("--filter-proximity-margin", default=str(args.baseline_inset_margin)).strip()
            or str(args.baseline_inset_margin)
        )

    print("[INFO] Block: edge eps and thresholds")
    args.edge_eps = float(prompt_text("--edge-eps", default=str(args.edge_eps)).strip() or str(args.edge_eps))
    args.abs_min_width_px = float(
        prompt_text("--abs-min-width-px", default=str(args.abs_min_width_px)).strip() or str(args.abs_min_width_px)
    )
    args.abs_min_height_px = float(
        prompt_text("--abs-min-height-px", default=str(args.abs_min_height_px)).strip() or str(args.abs_min_height_px)
    )
    args.rel_quantile = float(prompt_text("--rel-quantile", default=str(args.rel_quantile)).strip() or str(args.rel_quantile))
    args.rel_width_factor = float(
        prompt_text("--rel-width-factor", default=str(args.rel_width_factor)).strip() or str(args.rel_width_factor)
    )
    args.rel_height_factor = float(
        prompt_text("--rel-height-factor", default=str(args.rel_height_factor)).strip() or str(args.rel_height_factor)
    )

    print("[INFO] Block: extra filters")
    if prompt_yes_no("Enable --min-visibility?", default=args.min_visibility > 0):
        args.min_visibility = float(
            prompt_text("--min-visibility", default=str(args.min_visibility or 0.5)).strip() or "0.5"
        )
    else:
        args.min_visibility = 0.0
    if prompt_yes_no("Enable --min-area-px?", default=args.min_area_px > 0):
        args.min_area_px = float(prompt_text("--min-area-px", default=str(args.min_area_px or 16)).strip() or "16")
    else:
        args.min_area_px = 0.0
    if prompt_yes_no("Enable --max-aspect-ratio?", default=args.max_aspect_ratio is not None):
        args.max_aspect_ratio = float(
            prompt_text("--max-aspect-ratio", default=str(args.max_aspect_ratio or 20)).strip() or "20"
        )
    else:
        args.max_aspect_ratio = None

    print("[INFO] Block: image removal")
    args.drop_images = prompt_yes_no("--drop-images (remove whole image on any match)?", default=bool(args.drop_images))
    args.drop_background = prompt_yes_no(
        "--drop-background (remove source images without annotations)?",
        default=bool(args.drop_background),
    )
    args.prune_empty = prompt_yes_no("--prune-empty (default on)?", default=bool(args.prune_empty))

    mode = prompt_choice("Run mode", ["execute", "dry-run", "stats-only"], default="execute")
    args.dry_run = mode == "dry-run"
    args.stats_only = mode == "stats-only"


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_filter_arg_parser()
    args = parser.parse_args(argv)
    try:
        args.edge_sides = normalize_edge_sides(args.edge_sides)
        allowed_filter_sides(args.edge_sides)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return
    interactive_allowed = is_interactive_allowed(argv)

    if args.dataset is None and not interactive_allowed:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return

    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    catalog = load_datasets_catalog(layout)
    if not catalog:
        print(EMPTY_DATASETS_INFO_MESSAGE)
        return

    interactive_used = try_prompt_dataset_interactive(
        args=args,
        argv=argv,
        fill=lambda: _interactive_fill(args, sorted(catalog.keys()), catalog),
    )
    if not args.dataset:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return
    if args.dataset not in catalog:
        print(f"[ERROR] Unknown dataset: {args.dataset}")
        return

    replay_cmd = None
    if interactive_used:
        replay_cmd = build_non_interactive_command("filter", parser, args)
        print_replay_command("before launch", replay_cmd)

    entry = catalog[args.dataset]
    if not isinstance(entry, dict):
        print(f"[ERROR] Invalid catalog entry for {args.dataset}")
        return
    class_map = entry.get("classes", {})
    if not isinstance(class_map, dict):
        class_map = {}
    id_to_name = {int(v): str(k) for k, v in class_map.items()}
    class_filter = _selected_class_ids(entry, args.classes)
    if args.classes and class_filter is not None and not class_filter:
        print(f"[ERROR] Unknown classes in filter: {args.classes}")
        return

    src_root = resolve_dataset_root_for_entry(
        args.dataset,
        entry,
        workspace_root=layout.root,
        source_catalog_dir=layout.datasets,
        legacy_source_parent=layout.datasets,
    )
    structure = str(entry.get("structure", "split"))
    tmp_root = os.path.join(layout.root, "tmp")
    items = _list_items(src_root, structure, entry, args.dataset, tmp_root)
    config = _config_from_args(args)
    size_cache: dict[str, tuple[int, int]] = {}

    forecast = _forecast_filter(items, config=config, class_filter=class_filter, id_to_name=id_to_name, size_cache=size_cache)
    _print_forecast_table(forecast, id_to_name=id_to_name)

    if interactive_used and not args.yes and not args.dry_run and not args.stats_only:
        if not prompt_yes_no("Proceed with filter?", default=True):
            print("[INFO] Cancelled by user.")
            if replay_cmd:
                print_replay_command("after execution", replay_cmd)
            return

    out_base = args.output_name or f"{args.dataset}_fltd"
    out_name = next_dataset_name(layout.datasets, out_base)
    out_dir = os.path.join(layout.datasets, out_name)

    parameters = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    parameters["config"] = asdict(config)

    if args.stats_only:
        preview_dir = os.path.join(layout.root, "tmp")
        os.makedirs(preview_dir, exist_ok=True)
        manifest_path = _write_filter_manifest(
            preview_dir,
            parameters=parameters,
            forecast=forecast,
            stats_after={"mode": "stats-only"},
        )
        print(f"[OK] stats-only preview written: {manifest_path}")
        if replay_cmd:
            print_replay_command("after execution", replay_cmd)
        return

    if args.dry_run:
        print(f"[OK] dry-run: dataset={args.dataset}, output={out_name}")
        print(
            f"[INFO] Audit layout (not written in dry-run): "
            f"{FILTER_AUDIT_ROOT}/{FILTER_AUDIT_DROPPED_IMAGES}/<split>/images|labels, "
            f"{FILTER_AUDIT_ROOT}/{FILTER_AUDIT_REMOVED_LABELS}/<split>/labels"
        )
        if replay_cmd:
            print_replay_command("after execution", replay_cmd)
        return

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    stats_after = _copy_and_filter(
        out_dir,
        items,
        config=config,
        class_stats=forecast.baseline_stats,
        class_filter=class_filter,
        drop_images=bool(args.drop_images),
        prune_empty=bool(args.prune_empty),
        drop_background=bool(args.drop_background),
        size_cache=size_cache,
        empirical_bounds_map=forecast.empirical_bounds_map,
        empirical_bounds_summary=forecast.empirical_bounds_summary,
    )
    _write_data_yaml(out_dir, class_map)
    manifest_path = _write_filter_manifest(out_dir, parameters=parameters, forecast=forecast, stats_after=stats_after)
    out_hash = calculate_dataset_hash(out_dir)
    update_datasets_sidecar(
        layout=layout,
        output_key=out_name,
        class_map={str(k): int(v) for k, v in class_map.items()},
        target_dir=out_dir,
        output_hash=out_hash,
        structure=structure,
    )
    passport_path = write_dataset_passport(
        output_dataset_dir=out_dir,
        command="filter",
        source_datasets=[{"name": args.dataset, "path": src_root, "dataset_hash": entry.get("dataset_hash")}],
        parameters=parameters,
        workspace_root=layout.root,
        transformations=[{"edge_filter": config.edge_filter, "drop_images": bool(args.drop_images)}],
        stats_before={
            "instances_before": forecast.instances_before,
            "would_drop": forecast.would_drop,
        },
        stats_after=stats_after | {"output_hash": out_hash},
        random_seed=None,
    )
    print(f"[OK] Dataset created: {out_dir}")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Passport: {passport_path}")
    if replay_cmd:
        print_replay_command("after execution", replay_cmd)
