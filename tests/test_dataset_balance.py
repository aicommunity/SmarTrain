from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from smartrain.dataset_balance import main as balance_main
from smartrain.datasets_json_former import main as scan_main
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(path, format="JPEG", quality=85)


def _prepare_workspace(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_b"
    (raw / "train" / "images").mkdir(parents=True, exist_ok=True)
    (raw / "train" / "labels").mkdir(parents=True, exist_ok=True)
    _write_jpg(raw / "train" / "images" / "a.jpg")
    _write_jpg(raw / "train" / "images" / "b.jpg")
    (raw / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "train" / "labels" / "b.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 2\nnames: ['cat','dog']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])


def test_balance_creates_new_dataset_and_passport(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(["--workspace", str(tmp_path), "--dataset", "ds_b", "--strategy", "oversample", "--target", "1.5"])
    out = tmp_path / "datasets" / "ds_b_balanced"
    assert out.is_dir()
    assert (out / "dataset_passport.json").is_file()
    p = json.loads((out / "dataset_passport.json").read_text(encoding="utf-8"))
    assert p["command"] == "balance"
    info = json.loads((tmp_path / "datasets" / "datasets_info.json").read_text(encoding="utf-8"))
    assert "ds_b_balanced" in info
    assert info["ds_b_balanced"]["data_path"] == "datasets/ds_b_balanced"


def test_balance_name_increment_and_class_filter(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(["--workspace", str(tmp_path), "--dataset", "ds_b", "--class", "cat"])
    balance_main(["--workspace", str(tmp_path), "--dataset", "ds_b", "--classes", "cat,dog"])
    assert (tmp_path / "datasets" / "ds_b_balanced").is_dir()
    assert (tmp_path / "datasets" / "ds_b_balanced_2").is_dir()


def test_balance_weights_and_report_manifest(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--strategy",
            "weights",
            "--weight-mode",
            "effective",
            "--emit-balance-report",
        ]
    )
    out = tmp_path / "datasets" / "ds_b_balanced"
    manifest = json.loads((out / "balance_manifest.json").read_text(encoding="utf-8"))
    assert manifest["strategy"] == "weights"
    assert manifest["weight_mode"] == "effective"
    assert "class_counts_before_bbox" in manifest
    assert "class_counts_after_bbox" in manifest


def test_balance_preset_applies_defaults_and_allows_override(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--preset",
            "hybrid-default",
            "--target",
            "1.1",
            "--emit-balance-report",
        ]
    )
    out = tmp_path / "datasets" / "ds_b_balanced"
    manifest = json.loads((out / "balance_manifest.json").read_text(encoding="utf-8"))
    assert manifest["strategy"] == "hybrid"
    # Overridden explicitly by CLI flag; preset should not overwrite it.
    assert abs(float(manifest["target"]) - 1.1) < 1e-9
    assert manifest["weight_mode"] == "effective"


def test_balance_interactive_can_enable_emit_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_workspace(tmp_path)
    answers = iter(
        [
            "ds_b",        # Dataset
            "oversample",  # Strategy
            "",            # Output
            "1.0",         # target
            "3.0",         # max-ratio
            "all",         # classes mode
            "y",           # emit-balance-report
            "n",           # emit-train-config
            "y",           # eval-coverage
            "n",           # dry-run
        ]
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    monkeypatch.setattr("smartrain.dataset_balance.prompt", lambda *a, **k: next(answers))
    monkeypatch.setattr("smartrain.cli_prompts.prompt", lambda *a, **k: next(answers))
    balance_main([])
    out = tmp_path / "datasets" / "ds_b_balanced"
    assert (out / "balance_manifest.json").is_file()


def test_balance_ensures_non_empty_val_and_test_when_possible(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--strategy",
            "oversample",
            "--target",
            "1.5",
        ]
    )
    out = tmp_path / "datasets" / "ds_b_balanced"
    val_imgs = list((out / "val" / "images").glob("*.jpg"))
    test_imgs = list((out / "test" / "images").glob("*.jpg"))
    assert len(val_imgs) > 0
    assert len(test_imgs) > 0


def test_balance_can_disable_eval_coverage(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--strategy",
            "copy",
            "--no-eval-coverage",
        ]
    )
    out = tmp_path / "datasets" / "ds_b_balanced"
    # source had only train in this fixture, so with disabled eval_coverage
    # val/test remain empty.
    val_imgs = list((out / "val" / "images").glob("*.jpg"))
    test_imgs = list((out / "test" / "images").glob("*.jpg"))
    assert len(val_imgs) == 0
    assert len(test_imgs) == 0


def test_balance_enriches_eval_class_coverage_from_train(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_cov"
    for split in ("train", "valid", "test"):
        (raw / split / "images").mkdir(parents=True, exist_ok=True)
        (raw / split / "labels").mkdir(parents=True, exist_ok=True)
    # train has both classes
    _write_jpg(raw / "train" / "images" / "a.jpg")
    (raw / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_jpg(raw / "train" / "images" / "b.jpg")
    (raw / "train" / "labels" / "b.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    # eval splits initially miss class 1
    _write_jpg(raw / "valid" / "images" / "v.jpg")
    (raw / "valid" / "labels" / "v.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_jpg(raw / "test" / "images" / "t.jpg")
    (raw / "test" / "labels" / "t.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 2\nnames: ['cat','dog']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])

    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_cov",
            "--strategy",
            "copy",
        ]
    )
    out = tmp_path / "datasets" / "ds_cov_balanced"
    val_labels = (out / "val" / "labels")
    test_labels = (out / "test" / "labels")
    val_text = "\n".join(p.read_text(encoding="utf-8") for p in val_labels.glob("*.txt"))
    test_text = "\n".join(p.read_text(encoding="utf-8") for p in test_labels.glob("*.txt"))
    assert ("1 " in val_text) or ("1 " in test_text)

