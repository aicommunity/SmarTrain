from __future__ import annotations

import argparse
from unittest.mock import patch

from smartrain.services.datasets.dataset_augment import (
    _augment_roi_prompt_label,
    _interactive_fill,
    _iter_flip_variants,
    _iter_orthogonal_variants,
    _normalize_augment_args,
)
import random


def test_augment_roi_prompt_label_center_rotate_only() -> None:
    assert _augment_roi_prompt_label(enable_center_rotate=True, enable_bbox_copy=False) == (
        "Rotation pivot (--placement-mode)"
    )


def test_augment_roi_prompt_label_bbox_copy_only() -> None:
    assert _augment_roi_prompt_label(enable_center_rotate=False, enable_bbox_copy=True) == (
        "Paste area for bbox_copy (--placement-mode)"
    )


def test_iter_flip_variants_exhaustive_h_and_v() -> None:
    args = argparse.Namespace(
        enable_flip=True,
        flip="h-and-v",
        flip_sampling="exhaustive",
        flip_prob=0.0,
    )
    assert len(_iter_flip_variants(args, random.Random(0))) == 2


def test_iter_orthogonal_exhaustive_two() -> None:
    args = argparse.Namespace(
        enable_orthogonal_rotate=True,
        orthogonal_rotate_sampling="exhaustive",
        orthogonal_rotate_prob=0.0,
        orthogonal_rotate_direction="random",
    )
    specs = _iter_orthogonal_variants(args, random.Random(0))
    assert len(specs) == 2
    assert {s.direction for s in specs} == {"cw", "ccw"}


def test_normalize_flip_none_error() -> None:
    args = argparse.Namespace(
        enable_flip=True,
        flip="none",
        flip_prob=0.5,
        orthogonal_rotate_prob=0.5,
        conveyor_noise_types="iso",
        conveyor_noise_intensity=0.35,
        placement_roi=False,
        enable_center_rotate=False,
        enable_bbox_copy=False,
        placement_mode="detector",
        center_rotate_anchor="center",
    )
    assert _normalize_augment_args(args) is not None


def test_interactive_skips_bbox_copy_block_when_disabled(capsys) -> None:
    args = argparse.Namespace(
        dataset=None,
        classes=None,
        output_name=None,
        label_type="segment",
        enable_flip=False,
        flip="horizontal",
        flip_sampling="probabilistic",
        flip_prob=0.5,
        enable_orthogonal_rotate=False,
        orthogonal_rotate_sampling="probabilistic",
        orthogonal_rotate_prob=0.5,
        orthogonal_rotate_direction="random",
        enable_photometric=False,
        enable_conveyor=False,
        enable_conveyor_rotate=False,
        enable_conveyor_scale=False,
        enable_conveyor_blur=False,
        enable_conveyor_shift=False,
        enable_conveyor_noise=False,
        conveyor_noise_types="iso,shot,gaussian",
        conveyor_noise_intensity=0.35,
        conveyor_noise_selection="random",
        enable_center_rotate=True,
        enable_bbox_copy=False,
        center_rotate_deg=5.0,
        rotate_copies=1,
        placement_mode="none",
        roi_model="yolo11n.pt",
        roi_conf=0.25,
        roi_class_ids=None,
        imbalance_mode="soft",
        imbalance_strength=1.0,
        min_diversity_iou=0.97,
        min_angle_delta=1.0,
        class_balance="on",
        color_match="meanstd",
        blend_feather=0.16,
        copy_paste_count=1,
        copy_paste_min_center_dist=0.15,
        copy_paste_placement_style="random",
        bbox_copy_copies=1,
        aug_class_aware_geo=False,
        aug_total_bbox_cap_mult=0.0,
        splits="train,val,test",
        dry_run=False,
    )
    prompts = iter(
        [
            "ds_a",
            "",
            "",
            "n",
            "n",
            "y",
            "n",
            "5.0",
            "1",
            "none",
            "n",
            "n",
            "n",
            "n",
            "n",
            "n",
            "soft",
            "1.0",
            "0.97",
            "1.0",
            "n",
            "0",
            "train,val,test",
            "n",
        ]
    )

    def fake_prompt_text(label: str, default: str = "", **kwargs) -> str:
        return next(prompts)

    def fake_prompt_yes_no(label: str, default: bool = False) -> bool:
        raw = next(prompts)
        return raw.lower() in {"y", "yes", "1", "true"}

    def fake_prompt_choice(label: str, options, default=None, **kwargs) -> str:
        raw = next(prompts)
        if raw in options:
            return raw
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        return str(default or options[0])

    with patch("smartrain.services.datasets.dataset_augment.prompt_choice", side_effect=fake_prompt_choice), patch(
        "smartrain.services.datasets.dataset_augment.prompt_text", side_effect=fake_prompt_text
    ), patch("smartrain.services.datasets.dataset_augment.prompt_yes_no", side_effect=fake_prompt_yes_no), patch(
        "smartrain.services.datasets.dataset_augment.prompt", side_effect=lambda *a, **k: next(prompts)
    ):
        _interactive_fill(args, ["ds_a"], ["cat"], "/tmp/ws")

    assert args.enable_bbox_copy is False
    out = capsys.readouterr().out
    assert "Block: bbox_copy" not in out
    assert "Class balance (--class-balance)" not in out
    assert "Block: placement / ROI" in out
    assert "image center; simple" in out
    assert "Block: Balancing/Variety (center rotation)" in out
    assert args.placement_mode == "none"


def test_interactive_preserves_segment_label_type() -> None:
    args = argparse.Namespace(label_type="segment")
    assert args.label_type == "segment"
