from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np

NOISE_TYPES = ("gaussian", "iso", "shot", "poisson-gaussian", "multiplicative", "impulse")
NOISE_SELECTIONS = ("random", "stack")


def parse_noise_types(raw: str | None) -> list[str]:
    if not raw or not str(raw).strip():
        return ["iso", "shot", "gaussian"]
    out = [t.strip().lower() for t in str(raw).split(",") if t.strip()]
    unknown = [t for t in out if t not in NOISE_TYPES]
    if unknown:
        raise ValueError(f"Unknown noise types: {unknown}; allowed: {NOISE_TYPES}")
    return out


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp01(t)


def noise_params_for_type(kind: str, intensity: float) -> dict[str, Any]:
    t = _clamp01(intensity)
    if kind == "gaussian":
        lo, hi = 0.002, 0.06
        std = _lerp(lo, hi, t)
        return {"std_range": (std, std * 1.5), "per_channel": True}
    if kind == "iso":
        return {
            "intensity": (_lerp(0.01, 0.08, t), _lerp(0.05, 0.45, t)),
            "color_shift": (_lerp(0.001, 0.005, t), _lerp(0.005, 0.03, t)),
        }
    if kind == "shot":
        return {"scale_range": (_lerp(0.02, 0.05, t), _lerp(0.05, 0.35, t))}
    if kind == "poisson-gaussian":
        return {"pg_a": _lerp(0.08, 0.01, t), "pg_b": _lerp(0.001, 0.02, t)}
    if kind == "multiplicative":
        spread = _lerp(0.01, 0.08, t)
        return {"multiplier": (1.0 - spread, 1.0 + spread)}
    if kind == "impulse":
        return {"amount": _lerp(0.0002, 0.004, t)}
    raise ValueError(kind)


class PoissonGaussianNoise(A.ImageOnlyTransform):
    def __init__(self, pg_a: float, pg_b: float, p: float = 1.0):
        super().__init__(p=p)
        self.pg_a = float(pg_a)
        self.pg_b = float(pg_b)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        x = img.astype(np.float32) / 255.0
        a = max(float(self.pg_a), 1e-6)
        b = float(self.pg_b)
        shot = np.random.poisson(np.clip(x / a, 0, None)) * a
        read = np.random.normal(0.0, b, x.shape).astype(np.float32)
        out = np.clip(shot + read, 0.0, 1.0)
        if img.dtype == np.uint8:
            return (out * 255.0).astype(np.uint8)
        return out.astype(img.dtype, copy=False)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("pg_a", "pg_b")


class ImpulseNoise(A.ImageOnlyTransform):
    def __init__(self, amount: float, p: float = 1.0):
        super().__init__(p=p)
        self.amount = float(amount)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        out = img.copy()
        amount = max(0.0, min(1.0, float(self.amount)))
        if amount <= 0.0:
            return out
        mask = np.random.random(out.shape[:2]) < amount
        salt = np.random.random(out.shape[:2]) < 0.5
        for c in range(out.shape[2] if out.ndim == 3 else 1):
            channel = out[..., c] if out.ndim == 3 else out
            channel[mask & salt] = 255
            channel[mask & ~salt] = 0
        return out

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("amount",)


def _transform_for_type(kind: str, intensity: float, *, p: float = 1.0) -> A.BasicTransform:
    params = noise_params_for_type(kind, intensity)
    if kind == "gaussian":
        return A.GaussNoise(**params, p=p)
    if kind == "iso":
        return A.ISONoise(color_shift=params["color_shift"], intensity=params["intensity"], p=p)
    if kind == "shot":
        return A.ShotNoise(scale_range=params["scale_range"], p=p)
    if kind == "poisson-gaussian":
        return PoissonGaussianNoise(pg_a=params["pg_a"], pg_b=params["pg_b"], p=p)
    if kind == "multiplicative":
        return A.MultiplicativeNoise(multiplier=params["multiplier"], p=p)
    if kind == "impulse":
        return ImpulseNoise(amount=params["amount"], p=p)
    raise ValueError(kind)


def build_conveyor_noise_transform(args) -> A.BasicTransform | None:
    if not bool(getattr(args, "enable_conveyor_noise", False)):
        return None
    types = parse_noise_types(getattr(args, "conveyor_noise_types", None))
    intensity = float(getattr(args, "conveyor_noise_intensity", 0.35))
    selection = str(getattr(args, "conveyor_noise_selection", "random"))
    transforms = [_transform_for_type(t, intensity, p=1.0) for t in types]
    if not transforms:
        return None
    if len(transforms) == 1:
        return transforms[0]
    if selection == "stack":
        return A.Compose(transforms, p=1.0)
    return A.OneOf(transforms, p=1.0)


def flatten_compose_transforms(transform: A.BasicTransform | None) -> list[A.BasicTransform]:
    if transform is None:
        return []
    if isinstance(transform, A.Compose):
        out: list[A.BasicTransform] = []
        for child in transform.transforms:
            out.extend(flatten_compose_transforms(child))
        return out
    if isinstance(transform, A.OneOf):
        out = []
        for child in transform.transforms:
            out.extend(flatten_compose_transforms(child))
        return out
    return [transform]
