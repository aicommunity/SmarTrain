from __future__ import annotations

import argparse
from typing import Any

BALANCE_PRESETS: dict[str, dict[str, object]] = {
    # Conservative weighted balancing for most datasets.
    "weights-safe": {
        "strategy": "weights",
        "weight_mode": "effective",
        "beta": 0.9999,
        "image_weight_agg": "mean",
        "weight_clip_min": 0.5,
        "weight_clip_max": 2.0,
        "replacement": "auto",
        "target": 1.2,
        "max_repeat_per_image": 3,
    },
    # LVIS-like repeat-factor sampling, more aggressive on tail classes.
    "rfs-aggressive": {
        "strategy": "rfs",
        "rfs_thresh": 0.002,
        "rfs_power": 0.5,
        "target": 1.5,
        "max_repeat_per_image": 6,
    },
    # Instance-aware RFS: frequency from bbox counts; image factor is weighted mean.
    "irfs-default": {
        "strategy": "irfs",
        "rfs_thresh": 0.001,
        "rfs_power": 0.5,
        "target": 1.3,
        "max_repeat_per_image": 5,
    },
    # Recommended default: moderate RFS + weighted sampling.
    "hybrid-default": {
        "strategy": "hybrid",
        "weight_mode": "effective",
        "beta": 0.9999,
        "image_weight_agg": "max",
        "weight_clip_min": 0.4,
        "weight_clip_max": 3.0,
        "replacement": "auto",
        "rfs_thresh": 0.001,
        "rfs_power": 0.5,
        "target": 1.3,
        "max_repeat_per_image": 5,
    },
    # Default for hybrid-aug: constrained growth + tail-first budget + head trim.
    "hybrid-aug-tail-budget": {
        "strategy": "hybrid-aug",
        "aug_class_aware_geo": True,
        "aug_total_bbox_cap_mult": 1.10,
        "aug_budget_tail_first": True,
        "aug_budget_tail_gamma": 1.0,
        "train_head_bbox_undersample": "median-factor",
        "train_head_bbox_cap_mult": 5.0,
        "eval_head_bbox_undersample": "median-factor",
        "eval_head_bbox_cap_mult": 8.0,
        "eval_head_bbox_min_count": 30,
        "eval_head_bbox_max_remove_frac": 0.35,
    },
}


def _apply_hybrid_aug_default_mode(args: argparse.Namespace, provided_flags: set[str]) -> None:
    """Default mode for hybrid-aug unless explicitly overridden by CLI flags."""
    if str(getattr(args, "strategy", "")) != "hybrid-aug":
        return
    mode_defaults: dict[str, object] = BALANCE_PRESETS["hybrid-aug-tail-budget"]
    flags_by_attr: dict[str, set[str]] = {
        "aug_class_aware_geo": {"--aug-class-aware-geo", "--no-aug-class-aware-geo"},
        "aug_total_bbox_cap_mult": {"--aug-total-bbox-cap-mult"},
        "aug_budget_tail_first": {"--aug-budget-tail-first", "--no-aug-budget-tail-first"},
        "aug_budget_tail_gamma": {"--aug-budget-tail-gamma"},
        "train_head_bbox_undersample": {"--train-head-bbox-undersample"},
        "train_head_bbox_cap_mult": {"--train-head-bbox-cap-mult"},
        "eval_head_bbox_undersample": {"--eval-head-bbox-undersample"},
        "eval_head_bbox_cap_mult": {"--eval-head-bbox-cap-mult"},
        "eval_head_bbox_min_count": {"--eval-head-bbox-min-count"},
        "eval_head_bbox_max_remove_frac": {"--eval-head-bbox-max-remove-frac"},
    }
    for attr, value in mode_defaults.items():
        if attr == "strategy":
            continue
        flags = flags_by_attr.get(attr, set())
        if flags and any(f in provided_flags for f in flags):
            continue
        setattr(args, attr, value)



def _build_hybrid_aug_augment_argv(
    *,
    workspace: str,
    intermediate_dataset: str,
    final_base: str,
    seed: int,
    preset: str,
    aug_enable_bbox_copy: bool,
    aug_class_aware_geo: bool,
    aug_total_bbox_cap_mult: float,
    aug_budget_tail_first: bool,
    aug_budget_tail_gamma: float,
    aug_imbalance_mode: str = "soft",
    aug_imbalance_strength: float = 1.0,
    aug_flip_sampling: str = "probabilistic",
    aug_flip_prob: float = 0.5,
    aug_min_diversity_iou: float = 0.97,
) -> list[str]:
    argv = [
        "--workspace",
        workspace,
        "--dataset",
        intermediate_dataset,
        "--output-name",
        final_base,
        "--seed",
        str(seed),
        "--splits",
        "train",
        "--enable-flip",
        "--enable-photometric",
        "--enable-center-rotate",
        "--center-rotate-anchor",
        "center",
        "--center-rotate-deg",
        "5",
        "--rotate-copies",
        "1",
        "--imbalance-mode",
        str(aug_imbalance_mode),
        "--imbalance-strength",
        str(float(aug_imbalance_strength)),
        "--flip-sampling",
        str(aug_flip_sampling),
        "--flip-prob",
        str(float(aug_flip_prob)),
        "--min-diversity-iou",
        str(float(aug_min_diversity_iou)),
    ]
    if preset == "conveyor-lite":
        argv.append("--enable-conveyor")
    if aug_enable_bbox_copy:
        argv.append("--enable-bbox-copy")
    if aug_class_aware_geo:
        argv.append("--aug-class-aware-geo")
    else:
        argv.append("--no-aug-class-aware-geo")
    if float(aug_total_bbox_cap_mult) > 0:
        argv.extend(["--aug-total-bbox-cap-mult", str(float(aug_total_bbox_cap_mult))])
        if aug_budget_tail_first:
            argv.append("--aug-budget-tail-first")
        else:
            argv.append("--no-aug-budget-tail-first")
        argv.extend(["--aug-budget-tail-gamma", str(float(aug_budget_tail_gamma))])
    return argv


