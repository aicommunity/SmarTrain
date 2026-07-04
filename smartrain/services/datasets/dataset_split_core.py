from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _random import Random

TRAIN_PART = 0.8
VAL_PART = 0.1
TEST_PART = 0.1
DEFAULT_RANDOM_SEED = 12345

_SPLIT_SUM_EPS = 1e-5


def parse_split_ratio_arg(value: str | None) -> tuple[float, float, float]:
    """
    Three parts train, val, test for repartitioning frames within each bucket.
    The sum must be 1.0 (with tolerance). If value is None - module constants.
    """
    if value is None or not str(value).strip():
        return TRAIN_PART, VAL_PART, TEST_PART
    raw = [x.strip() for x in str(value).split(",")]
    if len(raw) != 3:
        raise ValueError(
            "Exactly three numbers separated by commas are expected: train,val,test (for example 0.8,0.1,0.1)."
        )
    try:
        tr, va, te = (float(x) for x in raw)
    except ValueError as e:
        raise ValueError(f"Invalid numbers in split ratio: {value!r}") from e
    if tr < 0 or va < 0 or te < 0:
        raise ValueError("Split ratio shares cannot be negative.")
    s = tr + va + te
    if abs(s - 1.0) > _SPLIT_SUM_EPS:
        raise ValueError(f"Sum of split ratio should be 1.0 (currently {s:.6f}): {value!r}")
    return tr, va, te


def split_pairs_by_ratio(
    pairs: list[tuple[str, str]],
    train: float,
    val: float,
    test: float,
    *,
    rng: Random | None = None,
) -> dict[str, list[tuple[str, str]]]:
    shuffled = list(pairs)
    r = rng if rng is not None else random
    r.shuffle(shuffled)
    n = len(shuffled)
    return {
        "train": shuffled[: int(n * train)],
        "valid": shuffled[int(n * train) : int(n * (train + val))],
        "test": shuffled[int(n * (train + val)) :],
    }


def unique_output_stem(src_image_path: str, used_stems: set[str]) -> str:
    base = os.path.splitext(os.path.basename(src_image_path))[0]
    safe_base = base.replace(os.sep, "_").replace("/", "_")
    stem = safe_base
    if stem not in used_stems:
        used_stems.add(stem)
        return stem
    n = 2
    while True:
        candidate = f"{safe_base}__{n}"
        if candidate not in used_stems:
            used_stems.add(candidate)
            return candidate
        n += 1
