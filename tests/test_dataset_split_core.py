from __future__ import annotations

import random

from smartrain.services.datasets.dataset_split_core import (
    DEFAULT_RANDOM_SEED,
    split_pairs_by_ratio,
)


def test_split_pairs_by_ratio_empty() -> None:
    result = split_pairs_by_ratio([], 0.8, 0.1, 0.1, rng=random.Random(DEFAULT_RANDOM_SEED))
    assert result == {"train": [], "valid": [], "test": []}


def test_split_pairs_by_ratio_remainder_goes_to_test() -> None:
    pairs = [(f"img{i}.jpg", f"lbl{i}.txt") for i in range(10)]
    result = split_pairs_by_ratio(pairs, 0.8, 0.1, 0.1, rng=random.Random(42))
    assert len(result["train"]) == 8
    assert len(result["valid"]) == 1
    assert len(result["test"]) == 1
    assert sum(len(v) for v in result.values()) == 10


def test_split_pairs_by_ratio_deterministic_with_seed() -> None:
    pairs = [(f"img{i}.jpg", f"lbl{i}.txt") for i in range(20)]
    a = split_pairs_by_ratio(pairs, 0.7, 0.2, 0.1, rng=random.Random(DEFAULT_RANDOM_SEED))
    b = split_pairs_by_ratio(pairs, 0.7, 0.2, 0.1, rng=random.Random(DEFAULT_RANDOM_SEED))
    assert a == b


def test_split_pairs_by_ratio_all_train() -> None:
    pairs = [(f"img{i}.jpg", f"lbl{i}.txt") for i in range(5)]
    result = split_pairs_by_ratio(pairs, 1.0, 0.0, 0.0, rng=random.Random(1))
    assert len(result["train"]) == 5
    assert result["valid"] == []
    assert result["test"] == []
