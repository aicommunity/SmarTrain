"""Tests for workspace sync service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.services.workspace import workspace_sync_service as sync


def test_datasets_related_allows_empty_target_bootstrap() -> None:
    src = {"ds_a": {"classes": {"cat": 0}}}
    dst: dict = {}
    related, common = sync._datasets_related(dst, src, threshold=0.3)
    assert related is True
    assert common == set()


def test_datasets_related_requires_threshold_when_both_nonempty() -> None:
    dst = {"ds_a": {"classes": {"cat": 0}}}
    src = {"ds_b": {"classes": {"dog": 0}}}
    related, _ = sync._datasets_related(dst, src, threshold=0.3)
    assert related is False


def test_datasets_related_with_common_keys() -> None:
    dst = {"ds_a": {"classes": {"cat": 0}}}
    src = {"ds_a": {"classes": {"cat": 0}}, "ds_b": {"classes": {"dog": 0}}}
    related, common = sync._datasets_related(dst, src, threshold=0.3)
    assert related is True
    assert common == {"ds_a"}


def test_dataset_hash_reads_sidecar_field() -> None:
    assert sync._dataset_hash({"dataset_hash": " abc "}) == "abc"
    assert sync._dataset_hash({}) is None


def test_copy_missing_dir_children_skips_existing(tmp_path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "new.txt").write_text("x", encoding="utf-8")
    (dst / "old.txt").write_text("y", encoding="utf-8")
    stats = sync.SyncStats()
    sync._copy_missing_dir_children(str(src), str(dst), stats, dry_run=False)
    assert stats.copied == 1
    assert (dst / "new.txt").is_file()


def test_copy_missing_dir_children_logs_and_counts_errors(tmp_path, caplog) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "broken").mkdir()
    stats = sync.SyncStats()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sync, "_copy_tree_or_file", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom")))
        sync._copy_missing_dir_children(str(src), str(dst), stats, dry_run=False)
    assert stats.errors == 1
    assert stats.copied == 0


def test_read_json_returns_empty_on_invalid(tmp_path, caplog) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert sync._read_json(str(bad)) == {}


def test_write_json_uses_lock_and_writes_payload(tmp_path) -> None:
    target = tmp_path / "datasets_info.json"
    sync._write_json(str(target), {"ds": {"classes": {"cat": 0}}}, dry_run=False)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert "ds" in payload


def test_merge_datasets_list_appends_unique_lines(tmp_path) -> None:
    src_layout = WorkspaceLayout(str(tmp_path / "src"))
    dst_layout = WorkspaceLayout(str(tmp_path / "dst"))
    src_list = Path(src_layout.raw_data) / "datasets_list.txt"
    dst_list = Path(dst_layout.raw_data) / "datasets_list.txt"
    src_list.parent.mkdir(parents=True)
    dst_list.parent.mkdir(parents=True)
    src_list.write_text("alpha\nbeta\n", encoding="utf-8")
    dst_list.write_text("beta\ngamma\n", encoding="utf-8")
    sync._merge_datasets_list(src_layout, dst_layout, dry_run=False)
    lines = dst_list.read_text(encoding="utf-8").splitlines()
    assert lines == ["beta", "gamma", "alpha"]


def test_write_data_yaml_from_class_map_roundtrip(tmp_path) -> None:
    from smartrain.services.datasets.data_yaml_writer import write_data_yaml_from_class_map

    out = tmp_path / "dataset"
    out.mkdir()
    write_data_yaml_from_class_map(str(out), {"cat": 1, "dog": 0})
    text = (out / "data.yaml").read_text(encoding="utf-8")
    assert "names: ['dog', 'cat']" in text
    assert "train: train/images" in text


def test_print_summary_includes_error_count(capsys) -> None:
    stats = sync.SyncStats(errors=2, copied=1)
    sync._print_summary(stats, dry_run=False)
    out = capsys.readouterr().out
    assert "errors: 2" in out
    assert "copied: 1" in out
