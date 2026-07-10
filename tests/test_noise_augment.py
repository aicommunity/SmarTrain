from __future__ import annotations

import argparse

import albumentations as A
import numpy as np
import pytest

from smartrain.services.datasets.noise_augment import (
    PoissonGaussianNoise,
    build_conveyor_noise_transform,
    flatten_compose_transforms,
    noise_params_for_type,
    parse_noise_types,
)


def test_parse_noise_types_default() -> None:
    assert parse_noise_types(None) == ["iso", "shot", "gaussian"]


def test_parse_noise_types_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown noise types"):
        parse_noise_types("foo")


def test_build_conveyor_noise_transform_disabled() -> None:
    args = argparse.Namespace(enable_conveyor_noise=False)
    assert build_conveyor_noise_transform(args) is None


def test_build_conveyor_noise_transform_shot_only() -> None:
    args = argparse.Namespace(
        enable_conveyor_noise=True,
        conveyor_noise_types="shot",
        conveyor_noise_intensity=0.5,
        conveyor_noise_selection="random",
    )
    tf = build_conveyor_noise_transform(args)
    names = {type(t).__name__ for t in flatten_compose_transforms(tf)}
    assert "ShotNoise" in names


def test_build_conveyor_noise_transform_oneof() -> None:
    args = argparse.Namespace(
        enable_conveyor_noise=True,
        conveyor_noise_types="iso,shot",
        conveyor_noise_intensity=0.5,
        conveyor_noise_selection="random",
    )
    tf = build_conveyor_noise_transform(args)
    assert isinstance(tf, A.OneOf)


def test_noise_params_intensity_monotonic() -> None:
    low = noise_params_for_type("gaussian", 0.0)
    high = noise_params_for_type("gaussian", 1.0)
    assert low["std_range"][0] < high["std_range"][0]


def test_poisson_gaussian_noise_uint8() -> None:
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    tf = PoissonGaussianNoise(pg_a=0.05, pg_b=0.005, p=1.0)
    out = tf(image=img)["image"]
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert float(np.mean(np.abs(out.astype(np.float32) - img.astype(np.float32)))) < 40.0
