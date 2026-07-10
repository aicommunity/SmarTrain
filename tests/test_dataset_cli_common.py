from __future__ import annotations

from smartrain.services.datasets.dataset_cli_common import (
    sorted_class_names_for_dataset,
    sorted_class_names_union_from_catalog,
)


def test_sorted_class_names_for_dataset_returns_only_entry_classes() -> None:
    catalog = {
        "ds_a": {"classes": {"cat": 0, "dog": 1}},
        "ds_b": {"classes": {"bee": 0, "ant": 1, "worm": 2, "fly": 3}},
    }
    assert sorted_class_names_for_dataset(catalog, "ds_a") == ["cat", "dog"]
    assert sorted_class_names_for_dataset(catalog, "ds_b") == ["ant", "bee", "fly", "worm"]
    assert sorted_class_names_for_dataset(catalog, "missing") == []


def test_sorted_class_names_union_from_catalog_collects_all() -> None:
    catalog = {
        "ds_a": {"classes": {"cat": 0, "dog": 1}},
        "ds_b": {"classes": {"bee": 0, "ant": 1}},
    }
    assert sorted_class_names_union_from_catalog(catalog) == ["ant", "bee", "cat", "dog"]
