from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from PIL import Image

from smartrain.workflows.datasets.dataset_former import _collect_label_image_pairs, main as fusion_main, prune_output_empty_label_pairs
from smartrain.core.runtime.workspace_paths import DATASETS_INFO_FILE, CLASS_NAMES_FILE, WORKSPACE_ENV_VAR, deploy_workspace


def _write_jpg(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path, format="JPEG", quality=85)


def _write_split_dataset(
    root: Path,
    name: str,
    class_name: str,
    class_idx: int,
    stem: str,
) -> None:
    base = root / "datasets" / name
    img = base / "train" / "images" / f"{stem}.jpg"
    lbl = base / "train" / "labels" / f"{stem}.txt"
    seed = sum(ord(ch) for ch in f"{name}-{stem}") % 255
    _write_jpg(img, color=(seed, (seed * 3) % 255, (seed * 7) % 255))
    lbl.parent.mkdir(parents=True, exist_ok=True)
    lbl.write_text(f"{class_idx} 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def test_fusion_without_partial_requires_all_classes_in_each_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    _write_split_dataset(tmp_path, "ds_a", "cat", 0, "a1")
    _write_split_dataset(tmp_path, "ds_b", "dog", 0, "b1")

    datasets_info = {
        "ds_a": {"classes": {"cat": 0}, "structure": "split"},
        "ds_b": {"classes": {"dog": 0}, "structure": "split"},
    }
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps(datasets_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (sd / CLASS_NAMES_FILE).write_text(
        json.dumps({"cat": "cat", "dog": "dog"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--output-name",
            "merged",
            "--dataset",
            "ds_a",
            "--dataset",
            "ds_b",
            "--classes",
            "cat,dog",
            "--no-include-partial-datasets",
        ]
    )
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "No dataset contains all selected classes" in out
    assert not (tmp_path / "datasets" / "merged" / "data.yaml").is_file()


def test_collect_label_image_pairs_finds_nested_labels(tmp_path: Path) -> None:
    """Regression: fusion must not only os.listdir(labels) — nested YOLO layout must be included."""
    root = tmp_path / "d"
    (root / "images" / "s").mkdir(parents=True)
    (root / "labels" / "s").mkdir(parents=True)
    (root / "images" / "s" / "z.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (root / "labels" / "s" / "z.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    pairs = _collect_label_image_pairs(str(root / "images"), str(root / "labels"))
    assert len(pairs) == 1
    assert pairs[0][0].endswith("z.jpg")
    assert pairs[0][1].endswith("z.txt")


def test_fusion_include_partial_merges_disjoint_class_datasets(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    _write_split_dataset(tmp_path, "ds_a", "cat", 0, "a1")
    _write_split_dataset(tmp_path, "ds_b", "dog", 0, "b1")

    datasets_info = {
        "ds_a": {"classes": {"cat": 0}, "structure": "split"},
        "ds_b": {"classes": {"dog": 0}, "structure": "split"},
    }
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps(datasets_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (sd / CLASS_NAMES_FILE).write_text(
        json.dumps({"cat": "cat", "dog": "dog"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--output-name",
            "merged",
            "--dataset",
            "ds_a",
            "--dataset",
            "ds_b",
            "--classes",
            "cat,dog",
            "--include-partial-datasets",
        ]
    )

    out_root = tmp_path / "datasets" / "merged"
    assert (out_root / "data.yaml").is_file()
    yaml_text = (out_root / "data.yaml").read_text(encoding="utf-8")
    assert "cat" in yaml_text and "dog" in yaml_text
    imgs = list(out_root.glob("*/images/*.jpg"))
    assert len(imgs) >= 2


def test_fusion_default_output_name_is_timestamp_merged(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    _write_split_dataset(tmp_path, "ds_a", "cat", 0, "a1")
    _write_split_dataset(tmp_path, "ds_b", "dog", 0, "b1")

    datasets_info = {
        "ds_a": {"classes": {"cat": 0}, "structure": "split"},
        "ds_b": {"classes": {"dog": 0}, "structure": "split"},
    }
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps(datasets_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (sd / CLASS_NAMES_FILE).write_text(
        json.dumps({"cat": "cat", "dog": "dog"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_a",
            "--dataset",
            "ds_b",
            "--classes",
            "cat,dog",
            "--include-partial-datasets",
        ]
    )


def test_fusion_unknown_dataset_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    _write_split_dataset(tmp_path, "ds_a", "cat", 0, "a1")
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps({"ds_a": {"classes": {"cat": 0}, "structure": "split"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (sd / CLASS_NAMES_FILE).write_text(
        json.dumps({"cat": "cat"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "missing_ds",
            "--classes",
            "cat",
        ]
    )
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "Unknown datasets" in out


def test_fusion_unknown_classes_for_selected_datasets_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    _write_split_dataset(tmp_path, "ds_a", "cat", 0, "a1")
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps({"ds_a": {"classes": {"cat": 0}, "structure": "split"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (sd / CLASS_NAMES_FILE).write_text(
        json.dumps({"cat": "cat"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--output-name",
            "merged_unknown_class",
            "--dataset",
            "ds_a",
            "--classes",
            "dog",
        ]
    )
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "contains unknown classes for the selected datasets" in out.lower()


def test_fusion_accepts_datasets_csv(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    _write_split_dataset(tmp_path, "ds_a", "cat", 0, "a1")
    _write_split_dataset(tmp_path, "ds_b", "dog", 0, "b1")
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps(
            {"ds_a": {"classes": {"cat": 0}, "structure": "split"}, "ds_b": {"classes": {"dog": 0}, "structure": "split"}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (sd / CLASS_NAMES_FILE).write_text(
        json.dumps({"cat": "cat", "dog": "dog"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--output-name",
            "merged_csv",
            "--datasets",
            "ds_a,ds_b",
            "--classes",
            "cat,dog",
            "--include-partial-datasets",
        ]
    )
    assert (tmp_path / "datasets" / "merged_csv" / "data.yaml").is_file()
    assert (tmp_path / "datasets" / "merged_csv" / "dataset_passport.json").is_file()


def test_fusion_partial_args_do_not_trigger_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    _write_split_dataset(tmp_path, "ds_a", "cat", 0, "a1")
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps({"ds_a": {"classes": {"cat": 0}, "structure": "split"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (sd / CLASS_NAMES_FILE).write_text(
        json.dumps({"cat": "cat"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--output-name",
            "merged_interactive",
            "--classes",
            "cat",
        ]
    )
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "incomplete arguments" in out.lower()
    assert not (tmp_path / "datasets" / "merged_interactive" / "data.yaml").is_file()


def test_fusion_interactive_options_apply_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    _write_split_dataset(tmp_path, "ds_a", "cat", 0, "a1")
    _write_split_dataset(tmp_path, "ds_b", "dog", 0, "b1")
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps(
            {"ds_a": {"classes": {"cat": 0}, "structure": "split"}, "ds_b": {"classes": {"dog": 0}, "structure": "split"}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (sd / CLASS_NAMES_FILE).write_text(
        json.dumps({"cat": "cat", "dog": "dog"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr("smartrain.workflows.datasets.dataset_former._prompt_dataset_selection", lambda available: ["ds_a", "ds_b"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))

    def _fake_options(args, default_output_name, class_candidates):
        args.output_name = "merged_from_interactive"
        args.classes = "cat,dog"
        args.include_partial_datasets = True
        args.fusion_split = "0.8,0.1,0.1"

    monkeypatch.setattr("smartrain.workflows.datasets.dataset_former._prompt_interactive_options", _fake_options)

    fusion_main([])
    assert (tmp_path / "datasets" / "merged_from_interactive" / "data.yaml").is_file()


def test_prune_output_empty_label_pairs_removes_orphans(tmp_path: Path) -> None:
    split = tmp_path / "train"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir(parents=True)
    (split / "labels" / "x.txt").write_text("\n\n", encoding="utf-8")
    _write_jpg(split / "images" / "x.jpg")

    n = prune_output_empty_label_pairs(str(tmp_path))
    assert n == 1
    assert not (split / "labels" / "x.txt").exists()
    assert not (split / "images" / "x.jpg").exists()


def test_fusion_dedup_same_image_and_equivalent_labels_keeps_one(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    ds_a = sd / "ds_a"
    ds_b = sd / "ds_b"
    _write_jpg(ds_a / "train" / "images" / "same_a.jpg", color=(100, 100, 100))
    # identical bytes, different name
    (ds_b / "train" / "images").mkdir(parents=True, exist_ok=True)
    shutil_src = ds_a / "train" / "images" / "same_a.jpg"
    (ds_b / "train" / "images" / "same_b.jpg").write_bytes(shutil_src.read_bytes())
    (ds_a / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (ds_b / "train" / "labels").mkdir(parents=True, exist_ok=True)
    # same class semantic, different local ids
    (ds_a / "train" / "labels" / "same_a.txt").write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (ds_b / "train" / "labels" / "same_b.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps(
            {
                "ds_a": {"classes": {"cat": 5}, "structure": "split"},
                "ds_b": {"classes": {"cat": 0}, "structure": "split"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (sd / CLASS_NAMES_FILE).write_text(json.dumps({"cat": "cat"}, ensure_ascii=False, indent=2), encoding="utf-8")
    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--output-name",
            "merged_dedup_1",
            "--dataset",
            "ds_a",
            "--dataset",
            "ds_b",
            "--classes",
            "cat",
        ]
    )
    out = tmp_path / "datasets" / "merged_dedup_1"
    assert len(list(out.glob("*/images/*.jpg"))) == 1
    labels = list(out.glob("*/labels/*.txt"))
    assert len(labels) == 1
    assert labels[0].read_text(encoding="utf-8").count("\n") == 1


def test_fusion_dedup_different_image_same_labels_keeps_both(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    ds_a = sd / "ds_a"
    ds_b = sd / "ds_b"
    _write_jpg(ds_a / "train" / "images" / "same.jpg", color=(10, 10, 10))
    _write_jpg(ds_b / "train" / "images" / "same.jpg", color=(200, 200, 200))
    (ds_a / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (ds_b / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (ds_a / "train" / "labels" / "same.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (ds_b / "train" / "labels" / "same.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps(
            {
                "ds_a": {"classes": {"cat": 0}, "structure": "split"},
                "ds_b": {"classes": {"cat": 0}, "structure": "split"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (sd / CLASS_NAMES_FILE).write_text(json.dumps({"cat": "cat"}, ensure_ascii=False, indent=2), encoding="utf-8")
    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--output-name",
            "merged_dedup_2",
            "--dataset",
            "ds_a",
            "--dataset",
            "ds_b",
            "--classes",
            "cat",
        ]
    )
    out = tmp_path / "datasets" / "merged_dedup_2"
    assert len(list(out.glob("*/images/*.jpg"))) == 2


def test_fusion_dedup_same_image_different_labels_unions_with_iou_dedup(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    ds_a = sd / "ds_a"
    ds_b = sd / "ds_b"
    _write_jpg(ds_a / "train" / "images" / "img1.jpg", color=(120, 80, 40))
    (ds_b / "train" / "images").mkdir(parents=True, exist_ok=True)
    (ds_b / "train" / "images" / "img1_other.jpg").write_bytes(
        (ds_a / "train" / "images" / "img1.jpg").read_bytes()
    )
    (ds_a / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (ds_b / "train" / "labels").mkdir(parents=True, exist_ok=True)
    # A: one cat box
    (ds_a / "train" / "labels" / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    # B: same cat box (should dedup by IoU), plus dog box
    (ds_b / "train" / "labels" / "img1_other.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n",
        encoding="utf-8",
    )
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps(
            {
                "ds_a": {"classes": {"cat": 0}, "structure": "split"},
                "ds_b": {"classes": {"cat": 0, "dog": 1}, "structure": "split"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (sd / CLASS_NAMES_FILE).write_text(
        json.dumps({"cat": "cat", "dog": "dog"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--output-name",
            "merged_dedup_3",
            "--dataset",
            "ds_a",
            "--dataset",
            "ds_b",
            "--classes",
            "cat,dog",
            "--include-partial-datasets",
        ]
    )
    out = tmp_path / "datasets" / "merged_dedup_3"
    assert len(list(out.glob("*/images/*.jpg"))) == 1
    labels = list(out.glob("*/labels/*.txt"))
    assert len(labels) == 1
    lines = [ln.strip() for ln in labels[0].read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2


def test_fusion_dedup_split_conflict_keeps_first_source_split(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    sd = tmp_path / "datasets"
    ds_a = sd / "ds_a"
    ds_b = sd / "ds_b"
    _write_jpg(ds_a / "train" / "images" / "same.jpg", color=(33, 66, 99))
    (ds_a / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (ds_a / "train" / "labels" / "same.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (ds_b / "test" / "images").mkdir(parents=True, exist_ok=True)
    (ds_b / "test" / "labels").mkdir(parents=True, exist_ok=True)
    (ds_b / "test" / "images" / "same_other.jpg").write_bytes(
        (ds_a / "train" / "images" / "same.jpg").read_bytes()
    )
    (ds_b / "test" / "labels" / "same_other.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (sd / DATASETS_INFO_FILE).write_text(
        json.dumps(
            {
                "ds_a": {"classes": {"cat": 0}, "structure": "split"},
                "ds_b": {"classes": {"cat": 0}, "structure": "split"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (sd / CLASS_NAMES_FILE).write_text(json.dumps({"cat": "cat"}, ensure_ascii=False, indent=2), encoding="utf-8")
    fusion_main(
        [
            "--workspace",
            str(tmp_path),
            "--output-name",
            "merged_dedup_4",
            "--dataset",
            "ds_a",
            "--dataset",
            "ds_b",
            "--classes",
            "cat",
            "--fusion-split",
            "1,0,0",
        ]
    )
    out = tmp_path / "datasets" / "merged_dedup_4"
    assert len(list((out / "train" / "images").glob("*.jpg"))) == 1
    assert len(list((out / "test" / "images").glob("*.jpg"))) == 0
