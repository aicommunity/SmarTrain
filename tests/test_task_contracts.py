from __future__ import annotations

import pytest

from smartrain.tasks.contracts import TaskTypeLabel, normalize_task_type


def test_normalize_task_type_accepts_known_aliases() -> None:
    assert normalize_task_type("Detection") == "detection"


def test_task_type_label_normalized() -> None:
    ctx = TaskTypeLabel(task_type="Detection")
    assert ctx.normalized() == "detection"


def test_normalize_task_type_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported task_type"):
        normalize_task_type("pose")
