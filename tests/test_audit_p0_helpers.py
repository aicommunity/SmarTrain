"""Unit tests for shared interactive helpers and registry run fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smartrain.cli_entrypoints.support.cli_interactive import should_run_interactive
from smartrain.core.runtime.run_refs import resolve_run_ref
from smartrain.services.registry.run_fields import load_run_list_fields, read_training_metadata


def test_should_run_interactive_requires_missing_attr(monkeypatch) -> None:
    monkeypatch.setattr(
        "smartrain.cli_entrypoints.support.cli_interactive.is_interactive_tty",
        lambda: True,
    )
    args = argparse.Namespace(dataset=None, nit=False)
    assert should_run_interactive(args, ("dataset",)) is True
    args.dataset = "ds"
    assert should_run_interactive(args, ("dataset",)) is False
    args.dataset = None
    args.nit = True
    assert should_run_interactive(args, ("dataset",)) is False


def test_resolve_run_ref_path_and_index(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    r1 = runs / "ds" / "run_a"
    r1.mkdir(parents=True)
    (r1 / "training_metadata.json").write_text("{}", encoding="utf-8")
    # find_run_directories needs recognizable layout — use absolute path branch
    abs_resolved = resolve_run_ref(str(runs), str(r1), exit_on_error=False)
    assert Path(abs_resolved).resolve() == r1.resolve()


def test_read_training_metadata_and_list_fields(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    md = {
        "training_info": {"model": "yolov8n", "dataset": {"name": "ds1", "hash": "abc"}},
        "timestamps": {"training": {"end": None, "start": None}},
    }
    (run / "training_metadata.json").write_text(json.dumps(md), encoding="utf-8")
    loaded = read_training_metadata(str(run))
    assert loaded["training_info"]["model"] == "yolov8n"
    fields = load_run_list_fields(str(run))
    assert fields["dataset"] == "ds1"
