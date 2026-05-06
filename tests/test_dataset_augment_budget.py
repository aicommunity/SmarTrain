from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
from PIL import Image

from smartrain.workflows.datasets.dataset_augment import (
    _effective_flip_prob_geo,
    _geo_photo_trigger,
    _image_tail_priority_score,
    main as augment_main,
    sum_train_bbox_disk,
)
from smartrain.datasets_json_former import main as scan_main
from smartrain.core.runtime.workspace_paths import deploy_workspace


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


def test_image_tail_priority_score_prefers_rare_class_on_frame(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("tail_score")
    lbl_head = tmp_path / "h.txt"
    lbl_tail = tmp_path / "t.txt"
    lbl_head.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    lbl_tail.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    cls_counts = {0: 10, 1: 1}
    assert _image_tail_priority_score(str(lbl_head), cls_counts, 1.0) < _image_tail_priority_score(
        str(lbl_tail), cls_counts, 1.0
    )


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


@pytest.fixture()
def workspace_tail_budget_order(tmp_path_factory: pytest.TempPathFactory):
    """Lex order a_head, b_bulk, z_tail — only one extra bbox allowed; tail-first should augment z_tail."""
    tmp_path = tmp_path_factory.mktemp("aug_tail_order")
    deploy_workspace(str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_tail"
    (raw / "train" / "images").mkdir(parents=True, exist_ok=True)
    (raw / "train" / "labels").mkdir(parents=True, exist_ok=True)
    line0 = "0 0.5 0.5 0.2 0.2\n"
    line1 = "1 0.5 0.5 0.2 0.2\n"
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(raw / "train" / "images" / "a_head.jpg")
    (raw / "train" / "labels" / "a_head.txt").write_text(line0, encoding="utf-8")
    Image.new("RGB", (16, 16), color=(4, 5, 6)).save(raw / "train" / "images" / "b_bulk.jpg")
    (raw / "train" / "labels" / "b_bulk.txt").write_text(line0 * 10, encoding="utf-8")
    Image.new("RGB", (16, 16), color=(7, 8, 9)).save(raw / "train" / "images" / "z_tail.jpg")
    (raw / "train" / "labels" / "z_tail.txt").write_text(line1, encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 2\nnames: ['c0', 'c1']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])
    return tmp_path


def test_augment_bbox_cap_tail_first_prefers_rare_stem(workspace_tail_budget_order) -> None:
    # B0 = 12, one extra line => mult = 13/12
    augment_main(
        [
            "--workspace",
            str(workspace_tail_budget_order),
            "--dataset",
            "ds_tail",
            "--enable-flip",
            "--flip-prob",
            "1.0",
            "--disable-center-rotate",
            "--aug-total-bbox-cap-mult",
            str(13.0 / 12.0),
            "--aug-budget-tail-first",
            "--no-aug-class-aware-geo",
        ]
    )
    out = workspace_tail_budget_order / "datasets" / "ds_tail_aug"
    lbl_dir = out / "train" / "labels"
    assert list(lbl_dir.glob("z_tail__a-f*.txt"))
    assert not list(lbl_dir.glob("a_head__a-f*.txt"))


def test_augment_bbox_cap_without_tail_first_uses_lex_order(workspace_tail_budget_order) -> None:
    augment_main(
        [
            "--workspace",
            str(workspace_tail_budget_order),
            "--dataset",
            "ds_tail",
            "--enable-flip",
            "--flip-prob",
            "1.0",
            "--disable-center-rotate",
            "--aug-total-bbox-cap-mult",
            str(13.0 / 12.0),
            "--no-aug-budget-tail-first",
            "--no-aug-class-aware-geo",
        ]
    )
    out = workspace_tail_budget_order / "datasets" / "ds_tail_aug"
    lbl_dir = out / "train" / "labels"
    assert list(lbl_dir.glob("a_head__a-f*.txt"))
    assert not list(lbl_dir.glob("z_tail__a-f*.txt"))


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
