from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from smartrain import model_training_module as mtm


def test_load_batch_from_training_metadata_reads_batch_size(tmp_path: Path) -> None:
    md = tmp_path / "run1"
    md.mkdir()
    (md / "training_metadata.json").write_text(
        json.dumps(
            {"training_info": {"hyperparameters": {"batch_size": 8}}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    assert mtm._load_batch_from_training_metadata(str(md)) == 8


def test_load_batch_from_training_metadata_returns_none_when_missing(tmp_path: Path) -> None:
    md = tmp_path / "run2"
    md.mkdir()
    assert mtm._load_batch_from_training_metadata(str(md)) is None


def test_test_only_default_val_batch_uses_metadata_then_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    main() in --test-only should select val_batch:
    - args.val_batch if specified
    - otherwise training_metadata.json (batch_size)
    - otherwise batch from u_cfg (after merge_cli_into_ultralytics_cfg)
    """
    md = tmp_path / "run3"
    md.mkdir()
    (md / "training_metadata.json").write_text(
        json.dumps({"training_info": {"hyperparameters": {"batch_size": 3}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    # Minimal stubs so that main gets to the test_yolo call.
    args = argparse.Namespace(
        workspace=str(tmp_path),
        config=None,
        ultralytics_yaml=None,
        data="dummy_ds",
        task=None,
        model=None,
        epochs=None,
        batch=7,  # will end up in u_cfg as fallback
        img_size=None,
        target_path=None,
        model_dir=str(md),
        test_only=True,
        non_interactive=True,
        val_imgsz=1280,
        val_conf=None,
        val_iou=None,
        val_batch=None,
        weighted_sampling=False,
        export_onnx=False,
        export_onnx_fp32=False,
        clearml=False,
        clearml_project=None,
    )

    monkeypatch.setattr(mtm, "parse_args", lambda _argv: args)
    monkeypatch.setattr(mtm, "_resolve_cli_paths_with_profile", lambda *_a, **_k: (str(tmp_path), "/ds", "/runs"))

    called: dict[str, int | None] = {"val_batch": None}

    def _fake_test_yolo(*_a, **kw):
        called["val_batch"] = kw.get("val_batch")
        return None, None, {}

    monkeypatch.setattr(mtm, "test_yolo", _fake_test_yolo)
    monkeypatch.setattr(mtm, "save_training_metadata", lambda **_kw: None)

    mtm.main(["--test-only"])
    assert called["val_batch"] == 3


def test_maybe_free_cuda_memory_calls_torch_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCuda:
        def __init__(self) -> None:
            self.empty_cache_called = 0
            self.ipc_collect_called = 0

        def is_available(self) -> bool:
            return True

        def empty_cache(self) -> None:
            self.empty_cache_called += 1

        def ipc_collect(self) -> None:
            self.ipc_collect_called += 1

    class _FakeTorch:
        def __init__(self) -> None:
            self.cuda = _FakeCuda()

    fake = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake)

    mtm._maybe_free_cuda_memory()
    assert fake.cuda.empty_cache_called == 1
    assert fake.cuda.ipc_collect_called == 1

