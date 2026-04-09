import pytest

from smartrain.dataset_former import TRAIN_PART, VAL_PART, TEST_PART, parse_fusion_split_arg


def test_parse_fusion_split_default() -> None:
    assert parse_fusion_split_arg(None) == (TRAIN_PART, VAL_PART, TEST_PART)
    assert parse_fusion_split_arg("") == (TRAIN_PART, VAL_PART, TEST_PART)
    assert parse_fusion_split_arg("  ") == (TRAIN_PART, VAL_PART, TEST_PART)


def test_parse_fusion_split_custom() -> None:
    assert parse_fusion_split_arg("0.7,0.2,0.1") == (0.7, 0.2, 0.1)
    assert parse_fusion_split_arg("1,0,0") == (1.0, 0.0, 0.0)


def test_parse_fusion_split_errors() -> None:
    with pytest.raises(ValueError, match="Exactly three|exactly three"):
        parse_fusion_split_arg("0.5,0.5")
    with pytest.raises(ValueError, match="Sum"):
        parse_fusion_split_arg("0.5,0.5,0.5")
    with pytest.raises(ValueError, match="negative"):
        parse_fusion_split_arg("1.0,-0.1,0.1")
