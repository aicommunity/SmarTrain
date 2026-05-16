"""Regression: Ultralytics val() must use canonical tests/test-ultralytics kwargs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from smartrain.core.runtime.run_artifacts import run_tests_dir


@pytest.fixture()
def minimal_detect_run(tmp_path: Path) -> tuple[str, str, str]:
    """Run dir + weights + data.yaml with test/images and one jpg."""
    ds = tmp_path / "ds"
    (ds / "test" / "images").mkdir(parents=True)
    (ds / "test" / "labels").mkdir(parents=True)
    (ds / "test" / "images" / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (ds / "test" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml = ds / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: test/images\nval: test/images\ntest: test/images\nnames: ['obj']\n",
        encoding="utf-8",
    )

    run = tmp_path / "run1"
    (run / "train" / "weights").mkdir(parents=True)
    weights = run / "train" / "weights" / "best.pt"
    weights.write_bytes(b"stub")

    return str(run), str(weights), str(data_yaml)


def test_run_ultralytics_backend_passes_canonical_val_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, minimal_detect_run: tuple[str, str, str]
) -> None:
    root_dir, weights_path, data_yaml = minimal_detect_run
    captured: dict[str, object] = {}

    class _FakeModel:
        def __init__(self, *_a, **_k) -> None:
            pass

        def val(self, **kwargs):
            captured["val_kwargs"] = kwargs
            return SimpleNamespace()

    monkeypatch.setattr("smartrain.services.testing.backends.format_runners.YOLO", _FakeModel)
    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners._save_metrics_csv_for_format",
        lambda *_a, **_k: str(tmp_path / "ignored.csv"),
    )
    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners._finalize_ultralytics_pt_test_dir",
        lambda **_k: None,
    )
    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners._ensure_confidence_recommendations_for_explicit_artifact",
        lambda **_k: None,
    )
    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners.persist_target_test_artifacts_state",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "smartrain.services.testing.backends.format_runners._collect_test_system_profile",
        lambda **_k: {},
    )

    from smartrain.workflows.testing.model_test_backends import run_ultralytics_backend

    res = run_ultralytics_backend(
        root_dir=root_dir,
        weights_path=weights_path,
        dataset_yaml_path=data_yaml,
        format_name="pt",
        imgsz=640,
        val_conf=0.25,
        val_iou=0.7,
        val_batch=1,
        conf_rec_disable=True,
        task_type="detection",
    )
    assert res.success is True
    kw = captured.get("val_kwargs")
    assert isinstance(kw, dict)
    assert kw.get("name") == "test-ultralytics"
    assert kw.get("plots") is True
    assert kw.get("save") is True
    assert kw.get("exist_ok") is True
    project = str(kw.get("project") or "")
    assert Path(project) == run_tests_dir(root_dir)
