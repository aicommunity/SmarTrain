from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from smartrain.workflows.datasets.dataset_prune import main as prune_main
from smartrain.core.runtime.workspace_paths import DATASETS_INFO_FILE, WORKSPACE_ENV_VAR, deploy_workspace


def _write_jpg(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path, format="JPEG", quality=85)


def _setup_split_dataset(tmp_path: Path, name: str = "src_ds") -> Path:
    ds = tmp_path / "datasets" / name
    for split in ("train", "val", "test"):
        (ds / split / "images").mkdir(parents=True, exist_ok=True)
        (ds / split / "labels").mkdir(parents=True, exist_ok=True)
    (tmp_path / "datasets" / DATASETS_INFO_FILE).write_text(
        json.dumps({name: {"classes": {"cat": 0}, "structure": "split"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ds


def test_prune_empty_creates_pruned_dataset(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "good.jpg", (10, 10, 10))
    (ds / "train" / "labels" / "good.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_jpg(ds / "val" / "images" / "empty.jpg", (20, 20, 20))
    (ds / "val" / "labels" / "empty.txt").write_text("", encoding="utf-8")

    prune_main(["empty", "--workspace", str(tmp_path), "--dataset", "src_ds"])

    out = tmp_path / "datasets" / "src_ds_pruned"
    assert (out / "train" / "images" / "good.jpg").is_file()
    assert not (out / "val" / "images" / "empty.jpg").exists()
    assert (out / "dataset_passport.json").is_file()


def test_prune_dedup_removes_cross_split_duplicates_with_priority(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "a.jpg", (33, 33, 33))
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    # exact duplicate image in lower-priority split
    _write_jpg(ds / "test" / "images" / "dup.jpg", (33, 33, 33))
    (ds / "test" / "labels" / "dup.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    prune_main(["dedup", "--workspace", str(tmp_path), "--dataset", "src_ds"])
    out = tmp_path / "datasets" / "src_ds_deduped"
    assert (out / "train" / "images" / "a.jpg").is_file()
    assert not (out / "test" / "images" / "dup.jpg").exists()


def test_prune_dedup_refuses_balanced_source_without_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "a.jpg", (44, 44, 44))
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (ds / "dataset_passport.json").write_text(json.dumps({"command": "balance"}), encoding="utf-8")

    prune_main(["dedup", "--workspace", str(tmp_path), "--dataset", "src_ds"])
    out = capsys.readouterr().out.lower()
    assert "refusing dedup" in out
    assert not (tmp_path / "datasets" / "src_ds_deduped").exists()


def test_prune_dedup_allows_balanced_source_with_flag(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "a.jpg", (55, 55, 55))
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (ds / "dataset_passport.json").write_text(json.dumps({"command": "balance"}), encoding="utf-8")

    prune_main(
        [
            "dedup",
            "--workspace",
            str(tmp_path),
            "--dataset",
            "src_ds",
            "--allow-balanced-dedup",
        ]
    )
    assert (tmp_path / "datasets" / "src_ds_deduped").is_dir()


def test_prune_interactive_empty_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_split_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "a.jpg", (66, 66, 66))
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["empty", "src_ds", ""])
    monkeypatch.setattr("smartrain.services.datasets.dataset_prune.prompt_choice", lambda *a, **k: next(answers))
    monkeypatch.setattr("smartrain.services.datasets.dataset_prune.prompt_text", lambda *a, **k: "")
    prune_main([])
    assert (tmp_path / "datasets" / "src_ds_pruned").is_dir()


def _setup_classes_dataset(tmp_path: Path, name: str = "src_ds") -> Path:
    ds = tmp_path / "datasets" / name
    (ds / "train" / "images").mkdir(parents=True, exist_ok=True)
    (ds / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (tmp_path / "datasets" / DATASETS_INFO_FILE).write_text(
        json.dumps(
            {
                name: {
                    "classes": {"A": 0, "B": 1, "C": 2},
                    "structure": "split",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (ds / "data.yaml").write_text(
        "train: train/images\nval: train/images\ntest: train/images\n\n"
        "nc: 3\nnames: ['A', 'B', 'C']\n",
        encoding="utf-8",
    )
    return ds


def test_prune_classes_removes_unused_and_remaps_ids(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_classes_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "a.jpg", (10, 10, 10))
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.2 0.2\n", encoding="utf-8")

    prune_main(["classes", "--workspace", str(tmp_path), "--dataset", "src_ds"])

    out = tmp_path / "datasets" / "src_ds_classes_pruned"
    cfg = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["names"] == ["A", "C"]
    label_text = (out / "train" / "labels" / "a.txt").read_text(encoding="utf-8")
    assert "0 0.5" in label_text
    assert "1 0.3" in label_text
    info = json.loads((tmp_path / "datasets" / DATASETS_INFO_FILE).read_text(encoding="utf-8"))
    assert info["src_ds_classes_pruned"]["classes"] == {"A": 0, "C": 1}


def test_prune_classes_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    deploy_workspace(str(tmp_path))
    ds = _setup_classes_dataset(tmp_path)
    _write_jpg(ds / "train" / "images" / "a.jpg", (10, 10, 10))
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    prune_main(["classes", "--workspace", str(tmp_path), "--dataset", "src_ds", "--dry-run"])
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert not (tmp_path / "datasets" / "src_ds_classes_pruned").exists()

