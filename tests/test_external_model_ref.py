from __future__ import annotations

import pytest

from smartrain.external_model_ref import parse_external_model_ref, validate_external_model_ref


def test_parse_external_model_ref_valid() -> None:
    ref = parse_external_model_ref("dr-yolo:yolov8n")
    assert ref.is_external is True
    assert ref.provider_id == "dr-yolo"
    assert ref.model_ref == "yolov8n"
    assert ref.raw_value == "dr-yolo:yolov8n"


def test_parse_external_model_ref_invalid() -> None:
    ref = parse_external_model_ref("yolov8n.pt")
    assert ref.is_external is False
    assert ref.provider_id is None
    assert ref.model_ref == "yolov8n.pt"


def test_validate_external_model_ref_unknown_provider_raises() -> None:
    ref = parse_external_model_ref("unknown-provider:yolov8n")
    with pytest.raises(ValueError):
        validate_external_model_ref(ref, known_provider_ids={"dr-yolo"})
