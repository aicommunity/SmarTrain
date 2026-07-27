"""Unit tests for instance-aware RFS (IRFS) vs image-level RFS."""

from __future__ import annotations

import random

from smartrain.services.datasets.balance_strategies import (
    _irfs_expand_pool,
    _rfs_expand_pool,
)


def _synthetic_long_tail_pool() -> tuple[
    list[tuple[str, str, str, list[str]]],
    dict[str, set[str]],
    dict[str, int],
    dict[str, int],
]:
    """Head dominates image presence; rare tail co-occurs with many head boxes.

    Image-level RFS repeats co-occurrence frames via ``max(r_c)``, inflating
    head bbox mass. IRFS uses instance-weighted mean, so head growth is milder.
    """
    pool: list[tuple[str, str, str, list[str]]] = []
    for i in range(95):
        img = f"head_{i}.jpg"
        pool.append(("train", img, f"{img}.txt", ["h"]))
    for i in range(5):
        img = f"co_{i}.jpg"
        # Many head instances + one rare: image RFS max(r_c) ≈ r_t; IRFS mean ≪ r_t.
        pool.append(("train", img, f"{img}.txt", ["h"] * 20 + ["t"]))

    img_to_classes: dict[str, set[str]] = {}
    image_presence: dict[str, int] = {"h": 0, "t": 0}
    bbox_count: dict[str, int] = {"h": 0, "t": 0}
    for _s, img, _l, cls_names in pool:
        classes = set(cls_names)
        img_to_classes[img] = classes
        for c in classes:
            image_presence[c] += 1
        for c in cls_names:
            bbox_count[c] += 1
    return pool, img_to_classes, image_presence, bbox_count


def _bbox_count(expanded: list[tuple[str, str, str, list[str]]], cls: str) -> int:
    return sum(1 for item in expanded for c in item[3] if c == cls)


def test_irfs_inflates_head_less_than_image_rfs_on_cooccurrence() -> None:
    pool, img_to_classes, image_presence, bbox_count = _synthetic_long_tail_pool()
    # power=1 keeps floors distinct: image RFS ≈ r_t; IRFS mean ≪ r_t.
    thresh = 0.15
    power = 1.0
    base_head = _bbox_count(pool, "h")

    rfs = _rfs_expand_pool(
        pool,
        img_to_classes,
        image_presence,
        rfs_thresh=thresh,
        rfs_power=power,
        max_repeat_per_image=20,
        rng=random.Random(0),
    )
    irfs = _irfs_expand_pool(
        pool,
        bbox_count,
        rfs_thresh=thresh,
        rfs_power=power,
        max_repeat_per_image=20,
        rng=random.Random(0),
    )

    rfs_head_delta = _bbox_count(rfs, "h") - base_head
    irfs_head_delta = _bbox_count(irfs, "h") - base_head
    assert rfs_head_delta > irfs_head_delta
    assert len(irfs) >= len(pool)
    assert _bbox_count(irfs, "t") >= _bbox_count(pool, "t")


def test_irfs_preset_registered() -> None:
    from smartrain.services.datasets.balance_presets import BALANCE_PRESETS

    cfg = BALANCE_PRESETS["irfs-default"]
    assert cfg["strategy"] == "irfs"
    assert float(cfg["rfs_thresh"]) == 0.001
    assert float(cfg["rfs_power"]) == 0.5
