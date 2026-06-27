from __future__ import annotations

from smartrain.core.training.train_profile import task_to_metadata_task_type
from smartrain.external_providers.task_alias import ultralytics_task_alias


def test_task_to_metadata_segment() -> None:
    assert task_to_metadata_task_type("segment") == "segmentation"


def test_ultralytics_task_alias_segment() -> None:
    assert ultralytics_task_alias("segmentation") == "segment"


def test_ultralytics_task_alias_detect_default() -> None:
    assert ultralytics_task_alias(None) == "detect"
