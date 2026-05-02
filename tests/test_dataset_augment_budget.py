from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
from PIL import Image

from smartrain.dataset_augment import (
    _effective_flip_prob_geo,
    _geo_photo_trigger,
    _per_class_extra_bbox_allowances,
    main as augment_main,
    sum_train_bbox_disk,
)
from smartrain.datasets_json_former import main as scan_main
from smartrain.workspace_paths import deploy_workspace


def test_effective_flip_prob_geo_prefers_higher_image_weight() -> None:
    args = SimpleNamespace(
        flip_prob=1.0,
        aug_class_aware_geo=True,
        imbalance_mode="soft",
        imbalance_strength=1.0,
    )
    assert _effective_flip_prob_geo(args, 3.0) >= _effective_flip_prob_geo(args, 0.2)


def test_effective_flip_prob_geo_disabled_returns_flip_prob() -> None:
    args = SimpleNamespace(
        flip_prob=0.41,
        aug_class_aware_geo=False,
        imbalance_mode="soft",
        imbalance_strength=9.0,
    )
    assert abs(_effective_flip_prob_geo(args, 50.0) - 0.41) < 1e-9


def test_geo_photo_trigger_tail_higher_than_head_empirical() -> None:
    args = SimpleNamespace(aug_class_aware_geo=True, imbalance_mode="soft", imbalance_strength=1.0)
    rate_tail = sum(_geo_photo_trigger(args, 4.0, random.Random(i)) for i in range(800)) / 800.0
    rate_head = sum(_geo_photo_trigger(args, 0.08, random.Random(i)) for i in range(800)) / 800.0
    assert rate_tail >= rate_head


@pytest.fixture()
def workspace_two_train_images(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("aug_budget_ds")
    deploy_workspace(str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_cap"
    (raw / "train" / "images").mkdir(parents=True, exist_ok=True)
    (raw / "train" / "labels").mkdir(parents=True, exist_ok=True)
    for name in ("a", "b"):
        Image.new("RGB", (32, 24), color=(10, 20, 30)).save(raw / "train" / "images" / f"{name}.jpg")
        (raw / "train" / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 1\nnames: ['cat']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])
    return tmp_path


def test_per_class_extra_allowances_formula() -> None:
    assert _per_class_extra_bbox_allowances({0: 2}, 1.5) == {0: 1}
    assert _per_class_extra_bbox_allowances({0: 10}, 1.0) == {0: 0}
    assert _per_class_extra_bbox_allowances({1: 3, 2: 1}, 1.2) == {1: 1, 2: 1}  # ceil(1.2*1)=2 → +1


def test_augment_per_class_cap_allows_only_one_extra_copy(workspace_two_train_images) -> None:
    """Baseline 2 bbox class 0; mult 1.5 -> extra slack 1 for that class -> at most one flip variant."""
    augment_main(
        [
            "--workspace",
            str(workspace_two_train_images),
            "--dataset",
            "ds_cap",
            "--enable-flip",
            "--flip-prob",
            "1.0",
            "--disable-center-rotate",
            "--aug-per-class-bbox-cap-mult",
            "1.5",
            "--no-aug-class-aware-geo",
        ]
    )
    out = workspace_two_train_images / "datasets" / "ds_cap_aug"
    assert sum_train_bbox_disk(str(out)) == 3
    flip_like = list((out / "train" / "labels").glob("*__a-f*.txt"))
    assert len(flip_like) == 1


def test_augment_bbox_cap_mult_one_blocks_extra_flip_variants(workspace_two_train_images) -> None:
    augment_main(
        [
            "--workspace",
            str(workspace_two_train_images),
            "--dataset",
            "ds_cap",
            "--enable-flip",
            "--flip-prob",
            "1.0",
            "--disable-center-rotate",
            "--aug-total-bbox-cap-mult",
            "1.0",
            "--no-aug-class-aware-geo",
        ]
    )
    out = workspace_two_train_images / "datasets" / "ds_cap_aug"
    assert sum_train_bbox_disk(str(out)) == 2
    lbl_dir = out / "train" / "labels"
    flip_like = list(lbl_dir.glob("*__a-f*.txt"))
    assert not flip_like


def test_augment_bbox_cap_disabled_allows_flip(workspace_two_train_images) -> None:
    augment_main(
        [
            "--workspace",
            str(workspace_two_train_images),
            "--dataset",
            "ds_cap",
            "--enable-flip",
            "--flip-prob",
            "1.0",
            "--disable-center-rotate",
            "--no-aug-class-aware-geo",
        ]
    )
    out = workspace_two_train_images / "datasets" / "ds_cap_aug"
    flip_like = list((out / "train" / "labels").glob("*__a-f*.txt"))
    assert flip_like
