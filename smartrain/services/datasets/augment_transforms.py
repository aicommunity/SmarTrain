"""Flip, orthogonal, and albumentations compose helpers for dataset augment."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import albumentations as A

from smartrain.services.datasets.noise_augment import build_conveyor_noise_transform, parse_noise_types


@dataclass(frozen=True)
class FlipSpec:
    mode: Literal["horizontal", "vertical", "both"]
    tag: str


@dataclass(frozen=True)
class OrthogonalSpec:
    direction: Literal["cw", "ccw"]
    tag: str


def flip_specs_for_mode(flip: str) -> list[FlipSpec]:
    if flip == "horizontal":
        return [FlipSpec("horizontal", "h")]
    if flip == "vertical":
        return [FlipSpec("vertical", "v")]
    if flip == "both":
        return [FlipSpec("both", "b")]
    if flip == "h-and-v":
        return [FlipSpec("horizontal", "h"), FlipSpec("vertical", "v")]
    return []


def iter_flip_variants(args, rng: random.Random, *, flip_prob: float | None = None) -> list[FlipSpec]:
    if not args.enable_flip or str(args.flip) == "none":
        return []
    specs = flip_specs_for_mode(str(args.flip))
    if not specs:
        return []
    if str(getattr(args, "flip_sampling", "probabilistic")) == "exhaustive":
        return specs
    p = float(flip_prob if flip_prob is not None else getattr(args, "flip_prob", 0.5))
    if rng.random() > p:
        return []
    if len(specs) == 1:
        return specs
    return [rng.choice(specs)]


def iter_orthogonal_variants(args, rng: random.Random, *, orth_prob: float | None = None) -> list[OrthogonalSpec]:
    if not bool(getattr(args, "enable_orthogonal_rotate", False)):
        return []
    if str(getattr(args, "orthogonal_rotate_sampling", "probabilistic")) == "exhaustive":
        return [OrthogonalSpec("cw", "c"), OrthogonalSpec("ccw", "a")]
    p = float(orth_prob if orth_prob is not None else getattr(args, "orthogonal_rotate_prob", 0.5))
    if rng.random() > p:
        return []
    d = str(getattr(args, "orthogonal_rotate_direction", "random"))
    if d == "cw":
        return [OrthogonalSpec("cw", "c")]
    if d == "ccw":
        return [OrthogonalSpec("ccw", "a")]
    direction = rng.choice(["cw", "ccw"])
    return [OrthogonalSpec(direction, "c" if direction == "cw" else "a")]


def normalize_augment_args(args, *, argv: list[str] | None = None) -> str | None:
    if bool(getattr(args, "placement_roi", False)):
        args.placement_mode = "bbox"

    if bool(getattr(args, "enable_flip", False)) and str(getattr(args, "flip", "horizontal")) == "none":
        return "[ERROR] --enable-flip conflicts with --flip none."
    if not 0.0 <= float(getattr(args, "flip_prob", 0.5)) <= 1.0:
        return "[ERROR] --flip-prob must be in [0, 1]."
    if not 0.0 <= float(getattr(args, "orthogonal_rotate_prob", 0.5)) <= 1.0:
        return "[ERROR] --orthogonal-rotate-prob must be in [0, 1]."
    try:
        parse_noise_types(getattr(args, "conveyor_noise_types", None))
    except ValueError as exc:
        return f"[ERROR] {exc}"
    if not 0.0 <= float(getattr(args, "conveyor_noise_intensity", 0.35)) <= 1.0:
        return "[ERROR] --conveyor-noise-intensity must be in [0, 1]."

    anchor_explicit = argv is not None and "--center-rotate-anchor" in argv
    placement_explicit = argv is not None and ("--placement-mode" in argv or "--placement-roi" in argv)
    rotate_on = bool(getattr(args, "enable_center_rotate", False))
    copy_on = bool(getattr(args, "enable_bbox_copy", False))

    if rotate_on and copy_on:
        if placement_explicit and not anchor_explicit:
            args.center_rotate_anchor = {
                "none": "center",
                "bbox": "bbox",
                "detector": "detector",
            }[str(getattr(args, "placement_mode", "none"))]
        elif anchor_explicit and not placement_explicit:
            args.placement_mode = {
                "center": "none",
                "bbox": "bbox",
                "detector": "detector",
            }[str(getattr(args, "center_rotate_anchor", "center"))]

    return None


def conveyor_any(args) -> bool:
    return any(
        bool(getattr(args, name, False))
        for name in (
            "enable_conveyor_rotate",
            "enable_conveyor_scale",
            "enable_conveyor_blur",
            "enable_conveyor_shift",
            "enable_conveyor_noise",
        )
    )


def sync_conveyor_flags(args, *, argv: list[str] | None = None) -> None:
    if argv is not None and "--enable-conveyor" in argv:
        args.enable_conveyor_rotate = True
        args.enable_conveyor_scale = True
        args.enable_conveyor_blur = True
        args.enable_conveyor_shift = True
        args.enable_conveyor_noise = True
    args.enable_conveyor = conveyor_any(args)


def set_conveyor_enabled(args, enabled: bool) -> None:
    args.enable_conveyor = bool(enabled)
    if not enabled:
        args.enable_conveyor_rotate = False
        args.enable_conveyor_scale = False
        args.enable_conveyor_blur = False
        args.enable_conveyor_shift = False
        args.enable_conveyor_noise = False


def append_conveyor_transforms(t: list[A.BasicTransform], args) -> None:
    geo = (
        bool(getattr(args, "enable_conveyor_rotate", False))
        or bool(getattr(args, "enable_conveyor_scale", False))
        or bool(getattr(args, "enable_conveyor_shift", False))
    )
    if geo:
        rotate = (-5, 5) if bool(getattr(args, "enable_conveyor_rotate", False)) else 0
        translate = (
            {"x": (-0.03, 0.03), "y": (-0.03, 0.03)}
            if bool(getattr(args, "enable_conveyor_shift", False))
            else {"x": (0, 0), "y": (0, 0)}
        )
        scale = (0.95, 1.05) if bool(getattr(args, "enable_conveyor_scale", False)) else (1.0, 1.0)
        t.append(
            A.Affine(
                translate_percent=translate,
                scale=scale,
                rotate=rotate,
                border_mode=0,
                p=0.8,
            )
        )
    if bool(getattr(args, "enable_conveyor_noise", False)):
        noise_tf = build_conveyor_noise_transform(args)
        if noise_tf is not None:
            t.append(noise_tf)
    if bool(getattr(args, "enable_conveyor_blur", False)):
        t.append(A.MotionBlur(blur_limit=3, p=0.15))


def compose_for_basic(
    args,
    *,
    flip_mode: str | None = None,
    with_bboxes: bool = True,
    with_keypoints: bool = False,
) -> A.Compose:
    t: list[A.BasicTransform] = []
    flip = str(flip_mode if flip_mode is not None else getattr(args, "flip", "horizontal"))
    apply_flip = bool(getattr(args, "enable_flip", False)) or flip_mode is not None
    if apply_flip:
        if flip == "horizontal":
            t.append(A.HorizontalFlip(p=1.0))
        elif flip == "vertical":
            t.append(A.VerticalFlip(p=1.0))
        elif flip == "both":
            t.append(A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]))
    if conveyor_any(args):
        append_conveyor_transforms(t, args)
    if args.enable_photometric:
        t.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(args.brightness_limit),
                contrast_limit=float(args.contrast_limit),
                p=1.0,
            )
        )
    if args.enable_center_rotate:
        t.append(A.Affine(rotate=(-float(args.center_rotate_deg), float(args.center_rotate_deg)), p=1.0))
    if not t:
        return A.Compose([])
    params: dict = {}
    if with_bboxes:
        params["bbox_params"] = A.BboxParams(format="yolo", label_fields=["class_labels"], clip=True)
    if with_keypoints:
        params["keypoint_params"] = A.KeypointParams(format="xy", remove_invisible=False)
    return A.Compose(t, **params)
