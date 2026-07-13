from __future__ import annotations

from smartrain.services.datasets.dataset_convert_service import (
    list_available_targets,
    pick_structure_for_target,
    structures_display_name,
    TARGET_CVAT11,
    TARGET_YOLO,
)


def test_structures_display_name_joins_labels() -> None:
    label = structures_display_name(["split", "cvat11"])
    assert "YOLO split directories layout" in label
    assert "CVAT for images 1.1 (folder)" in label
    assert "; " in label


def test_list_available_targets_union() -> None:
    targets = {t.target_id for t in list_available_targets(["split", "cvat11"])}
    assert TARGET_YOLO in targets
    assert TARGET_CVAT11 in targets


def test_pick_structure_for_target_prefers_cvat_for_yolo() -> None:
    assert pick_structure_for_target(["split", "cvat11"], TARGET_YOLO) == "cvat11"
    assert pick_structure_for_target(["split", "cvat11"], TARGET_CVAT11) == "split"
