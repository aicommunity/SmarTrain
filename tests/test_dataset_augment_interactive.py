from __future__ import annotations

import argparse
from unittest.mock import patch

from smartrain.services.datasets.dataset_augment import (
    _augment_roi_prompt_label,
    _interactive_fill,
)


def test_augment_roi_prompt_label_center_rotate_only() -> None:
    assert _augment_roi_prompt_label(enable_center_rotate=True, enable_bbox_copy=False) == (
        "Rotation pivot (--placement-mode)"
    )


def test_augment_roi_prompt_label_bbox_copy_only() -> None:
    assert _augment_roi_prompt_label(enable_center_rotate=False, enable_bbox_copy=True) == (
        "Paste area for bbox_copy (--placement-mode)"
    )


def test_interactive_skips_bbox_copy_block_when_disabled(capsys) -> None:
    args = argparse.Namespace(
        dataset=None,
        classes=None,
        output_name=None,
        enable_flip=False,
        flip="horizontal",
        flip_prob=0.5,
        enable_photometric=False,
        enable_conveyor=False,
        enable_conveyor_rotate=False,
        enable_conveyor_scale=False,
        enable_conveyor_blur=False,
        enable_conveyor_shift=False,
        enable_conveyor_noise=False,
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
            "n",
            "n",
            "n",
            "n",
            "n",
            "y",
            "n",
            "5.0",
            "1",
            "none",
            "soft",
            "1.0",
            "0.97",
            "1.0",
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
