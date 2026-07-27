from __future__ import annotations

import argparse
from typing import Any

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR

AUGMENT_PRESETS: dict[str, dict[str, object]] = {
    "augment-tail-safe": {
        "aug_class_aware_geo": True,
        "aug_total_bbox_cap_mult": 1.10,
        "aug_budget_tail_first": True,
        "aug_budget_tail_gamma": 1.0,
    },
}



def build_augment_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Offline augmentation of a dataset into a new datasets/<name>")
    # --- source ---
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (aka {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Name of source dataset from datasets_info.json")
    p.add_argument(
        "--label-type",
        choices=("auto", "bbox", "segment"),
        default="auto",
        help="Label interpretation for augment: auto-detect, force bbox, or force segment (polygons).",
    )
    p.add_argument("--output-name", type=str, default=None, help="Name of output dataset (default <dataset>_aug)")
    p.add_argument("--splits", type=str, default="train", help="CSV: train,val,test")
    p.add_argument("--classes", type=str, default=None, help="Limit augmentation to CSV classes")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--preset",
        choices=tuple(AUGMENT_PRESETS.keys()),
        default=None,
        help=(
            "Preset with tuned augment balancing. "
            "augment-tail-safe: class-aware geo, bbox cap 1.10× baseline, tail-first budget "
            "(aligned with hybrid-aug-tail-budget defaults; standalone augment defaults stay unchanged)."
        ),
    )
    # --- flip ---
    p.add_argument("--enable-flip", action="store_true", help="Enable flip augmentation")
    p.add_argument(
        "--flip",
        choices=("horizontal", "vertical", "both", "h-and-v", "none"),
        default="horizontal",
        help="Flip mode: horizontal|vertical|both (combined H+V)|h-and-v (separate files)|none",
    )
    p.add_argument(
        "--flip-sampling",
        choices=("probabilistic", "exhaustive"),
        default="probabilistic",
        help=(
            "probabilistic: at most one flip per frame via --flip-prob; exhaustive: always all variants for --flip. "
            "With --aug-class-aware-geo, exhaustive bypasses per-frame probability scaling (see startup WARN)."
        ),
    )
    p.add_argument(
        "--flip-prob",
        type=float,
        default=0.5,
        help="Probability of creating a flip variant per frame [0..1] (ignored when --flip-sampling exhaustive)",
    )
    # --- orthogonal ±90° ---
    p.add_argument("--enable-orthogonal-rotate", action="store_true", help="Enable ±90° orthogonal rotation augmentation")
    p.add_argument(
        "--orthogonal-rotate-sampling",
        choices=("probabilistic", "exhaustive"),
        default="probabilistic",
        help=(
            "probabilistic: at most one ±90° variant; exhaustive: always both +90° and -90°. "
            "With --aug-class-aware-geo, exhaustive bypasses per-frame probability scaling (see startup WARN)."
        ),
    )
    p.add_argument(
        "--orthogonal-rotate-prob",
        type=float,
        default=0.5,
        help="Probability of orthogonal rotate variant [0..1] (ignored when exhaustive)",
    )
    p.add_argument(
        "--orthogonal-rotate-direction",
        choices=("random", "cw", "ccw"),
        default="random",
        help="Direction for probabilistic orthogonal rotate: random|cw (+90°)|ccw (-90°)",
    )
    # --- center-rotate ---
    p.add_argument("--enable-center-rotate", action="store_true", dest="enable_center_rotate", help="Enable frame rotation around the center")
    p.add_argument("--disable-center-rotate", action="store_false", dest="enable_center_rotate", help="Disable frame rotation around center")
    p.add_argument("--center-rotate-deg", type=float, default=5.0, help="Maximum rotation angle in both directions")
    p.add_argument("--rotate-copies", type=int, default=1, help="Number of rotate options per frame")
    p.add_argument(
        "--center-rotate-anchor",
        choices=("center", "bbox", "detector"),
        default="center",
        help="Rotation center source: frame center, bbox markup or ROI detector",
    )
    # --- photometric ---
    p.add_argument("--enable-photometric", action="store_true", help="Enable brightness/contrast")
    p.add_argument("--brightness-limit", type=float, default=0.1, help="For policy=basic: brightness range")
    p.add_argument("--contrast-limit", type=float, default=0.1, help="For policy=basic: range contrast")
    # --- conveyor ---
    p.add_argument(
        "--enable-conveyor",
        action="store_true",
        help="Enable all conveyor effects (rotate, scale, blur, shift, noise); alias for all --enable-conveyor-* flags",
    )
    p.add_argument("--enable-conveyor-rotate", action="store_true", help="Conveyor Affine rotate (±5°)")
    p.add_argument("--enable-conveyor-scale", action="store_true", help="Conveyor Affine scale (0.95–1.05)")
    p.add_argument("--enable-conveyor-blur", action="store_true", help="Conveyor motion blur")
    p.add_argument("--enable-conveyor-shift", action="store_true", help="Conveyor Affine shift (±3%% translate)")
    p.add_argument(
        "--enable-conveyor-noise",
        action="store_true",
        help="Conveyor sensor noise (see --conveyor-noise-types)",
    )
    p.add_argument(
        "--conveyor-noise-types",
        type=str,
        default="iso,shot,gaussian",
        help="CSV noise types: gaussian,iso,shot,poisson-gaussian,multiplicative,impulse",
    )
    p.add_argument(
        "--conveyor-noise-intensity",
        type=float,
        default=0.35,
        help="Overall noise strength [0..1]",
    )
    p.add_argument(
        "--conveyor-noise-selection",
        choices=("random", "stack"),
        default="random",
        help="random: one noise type per frame; stack: apply all selected types",
    )
    # --- bbox copy ---
    p.add_argument("--enable-bbox-copy", action="store_true", help="Enable bbox-copy augmentation")
    p.add_argument("--bbox-copy-copies", type=int, default=1, help="Number of bbox_copy options per frame")
    p.add_argument("--copy-paste-rotation", type=float, default=0.0, help="For policy=bbox_copy: max |angle| to rotate the insert")
    p.add_argument("--copy-paste-scale-min", type=float, default=1.0, help="For policy=bbox_copy: min scale (inner)")
    p.add_argument("--copy-paste-scale-max", type=float, default=1.0, help="For policy=bbox_copy: max scale (inner)")
    p.add_argument("--copy-paste-max-iou", type=float, default=0.0, help="Max IoU insertions with existing bboxes")
    p.add_argument("--copy-paste-tries", type=int, default=25, help="Number of attempts to find a valid placement")
    p.add_argument(
        "--copy-paste-min-center-dist",
        type=float,
        default=0.15,
        help="Minimum distance between the centers of new inserts (fraction of the frame diagonal)",
    )
    p.add_argument(
        "--copy-paste-placement-style",
        choices=("random", "uniform-grid"),
        default="random",
        help="Insertion position selection style bbox_copy",
    )
    p.add_argument(
        "--class-balance",
        choices=("on", "off"),
        default="on",
        help="For bbox_copy: balance the selection of donors by class",
    )
    p.add_argument(
        "--color-match",
        choices=("meanstd", "off"),
        default="meanstd",
        help="For bbox_copy: align the brightness/contrast of the patch relative to the target area",
    )
    p.add_argument(
        "--blend-feather",
        type=float,
        default=0.16,
        help="For bbox_copy: seam smoothing strength [0..0.5], 0=no feather",
    )
    p.add_argument("--copy-paste-count", type=int, default=1, help="For policy=bbox_copy: number of inserts per image")
    # --- placement / ROI ---
    p.add_argument(
        "--placement-mode",
        choices=("none", "bbox", "detector"),
        default="none",
        help="ROI placement mode for bbox_copy and rotation pivot: none|bbox|detector (default none)",
    )
    p.add_argument("--placement-roi", action="store_true", help="Legacy: same as --placement-mode bbox")
    p.add_argument("--roi-model", type=str, default="yolo11n.pt", help="ROI detector model for --placement-mode detector")
    p.add_argument("--roi-conf", type=float, default=0.25, help="Confidence threshold for ROI detector")
    p.add_argument("--roi-class-ids", type=str, default=None, help="CSV class ids for ROI detector (empty=all)")
    p.add_argument("--side-tolerance-px", type=float, default=3.0, help="Tolerance in px for ROI side classification")
    # --- balancing / budget ---
    p.add_argument("--imbalance-mode", choices=("off", "soft"), default="soft", help="Balancing according to scarce classes")
    p.add_argument("--imbalance-strength", type=float, default=1.0, help="Balancing strength >=0")
    p.add_argument(
        "--aug-class-aware-geo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Scale flip / photometric / conveyor attempt rate by inverse class frequency on the frame "
            "(uses --imbalance-mode soft and _image_soft_weight). Off keeps prior per-image behavior."
        ),
    )
    p.add_argument(
        "--aug-total-bbox-cap-mult",
        type=float,
        default=0.0,
        help=(
            "If >0: after augment, sum bbox on train ≤ ceil(mult × baseline B₀); baseline copies stay; "
            "only extra augmented frames consume slack (slack = ceil(mult×B₀) − B₀). mult=1.0 forbids extra bbox."
        ),
    )
    p.add_argument(
        "--aug-budget-tail-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --aug-total-bbox-cap-mult >0: process train images in split by descending tail priority "
            "(max_c (n_max/n_c)^γ over classes on the frame) so scarce classes consume slack first. "
            "Disable with --no-aug-budget-tail-first (dataset iteration order)."
        ),
    )
    p.add_argument(
        "--aug-budget-tail-gamma",
        type=float,
        default=1.0,
        help="Exponent γ for tail priority when --aug-budget-tail-first (default 1.0).",
    )
    p.add_argument(
        "--aug-per-class-bbox-cap-mult",
        type=float,
        default=0.0,
        help=(
            "If >0: per-class train bbox cap = ceil(mult × baseline n_c); extra variants that would exceed "
            "class slack are skipped. Complements --aug-total-bbox-cap-mult and tail-first ordering."
        ),
    )
    p.add_argument("--min-diversity-iou", type=float, default=0.97, help="bbox similarity threshold (higher -> almost duplicate)")
    p.add_argument("--min-angle-delta", type=float, default=1.0, help="Minimum angle difference between rotate options")
    # --- service ---
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-legend", action="store_true")
    p.set_defaults(enable_center_rotate=True)
    return p



def _provided_augment_flags(argv: list[str]) -> set[str]:
    out: set[str] = set()
    for tok in argv:
        if tok.startswith("--"):
            out.add(tok.split("=", 1)[0])
    return out



def _apply_augment_preset_defaults(args: argparse.Namespace, provided_flags: set[str]) -> None:
    if not getattr(args, "preset", None):
        return
    preset_cfg = AUGMENT_PRESETS.get(str(args.preset), {})
    flag_for_attr = {
        "aug_class_aware_geo": "--aug-class-aware-geo",
        "aug_total_bbox_cap_mult": "--aug-total-bbox-cap-mult",
        "aug_budget_tail_first": "--aug-budget-tail-first",
        "aug_budget_tail_gamma": "--aug-budget-tail-gamma",
    }
    for attr, value in preset_cfg.items():
        flag = flag_for_attr.get(attr)
        if flag and flag in provided_flags:
            continue
        setattr(args, attr, value)



def _warn_exhaustive_class_aware(args) -> None:
    if not _class_aware_enabled(args):
        return
    flip_ex = bool(getattr(args, "enable_flip", False)) and str(getattr(args, "flip_sampling", "probabilistic")) == "exhaustive"
    orth_ex = bool(getattr(args, "enable_orthogonal_rotate", False)) and str(
        getattr(args, "orthogonal_rotate_sampling", "probabilistic")
    ) == "exhaustive"
    if flip_ex or orth_ex:
        print(
            "[WARN] augment: --aug-class-aware-geo is active but flip/orthogonal use exhaustive sampling; "
            "per-frame probability scaling is bypassed (all variants are emitted). "
            "Use probabilistic sampling or disable class-aware geo for head-tail control."
        )


