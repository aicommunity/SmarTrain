from __future__ import annotations

from pathlib import Path

import pytest

from smartrain.services.datasets.dataset_roi_yolo import _transform_segment_line


def test_transform_segment_line_full_inside_crop() -> None:
  # square in center of 100x80 image, crop to center 50x40
    parts = ["0", "0.30", "0.25", "0.70", "0.25", "0.70", "0.75", "0.30", "0.75"]
    line = _transform_segment_line(parts, crop=(25, 20, 75, 60), iw=100, ih=80)
    assert line is not None
    out_parts = line.split()
    assert int(out_parts[0]) == 0
    assert len(out_parts) == 9  # class + 8 coords


def test_transform_segment_line_outside_crop_returns_none() -> None:
    parts = ["0", "0.05", "0.05", "0.15", "0.05", "0.15", "0.15", "0.05", "0.15"]
    line = _transform_segment_line(parts, crop=(50, 40, 90, 70), iw=100, ih=80)
    assert line is None


def test_transform_segment_line_partial_overlap_clips() -> None:
    parts = ["1", "0.40", "0.40", "0.90", "0.40", "0.90", "0.90", "0.40", "0.90"]
    line = _transform_segment_line(parts, crop=(50, 30, 100, 80), iw=100, ih=80)
    assert line is not None
    coords = [float(x) for x in line.split()[1:]]
    for i in range(0, len(coords), 2):
        assert 0.0 <= coords[i] <= 1.0
        assert 0.0 <= coords[i + 1] <= 1.0


def test_transform_segment_line_invalid_coords() -> None:
    assert _transform_segment_line(["0", "0.1", "0.2", "0.3"], crop=(0, 0, 50, 50), iw=100, ih=80) is None
