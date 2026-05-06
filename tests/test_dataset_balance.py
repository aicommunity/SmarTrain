from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from PIL import Image

from smartrain.workflows.datasets.dataset_balance import (
    _auto_head_cap_multipliers,
    _parse_class_weight_multiplier,
    main as balance_main,
)
from smartrain.workflows.datasets.datasets_json_former import main as scan_main
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace


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
    _write_jpg(raw / "train" / "images" / "c.jpg")
    (raw / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "train" / "labels" / "b.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "train" / "labels" / "c.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 2\nnames: ['cat','dog']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])


def _prepare_workspace_tiny(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_tiny"
    (raw / "train" / "images").mkdir(parents=True, exist_ok=True)
    (raw / "train" / "labels").mkdir(parents=True, exist_ok=True)
    _write_jpg(raw / "train" / "images" / "a.jpg")
    _write_jpg(raw / "train" / "images" / "b.jpg")
    (raw / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "train" / "labels" / "b.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 2\nnames: ['cat','dog']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])


def _source_key_from_output_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"_\d+$", "", stem)
    if stem.endswith("_bal"):
        stem = stem[: -len("_bal")]
    return stem


def _collect_split_keys(dataset_dir: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for split in ("train", "val", "test"):
        for p in (dataset_dir / split / "images").glob("*.*"):
            out[split].add(_source_key_from_output_name(p.name))
    return out


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
    assert manifest["max_ratio"] == 3.0
    assert manifest["min_count"] == 1
    assert manifest["eval_coverage"] is True
    assert manifest["emit_balance_report"] is True
    assert manifest["emit_train_config"] is False
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
    assert manifest["preset"] == "hybrid-default"
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
    monkeypatch.setattr("smartrain.workflows.datasets.dataset_balance.prompt", lambda *a, **k: next(answers))
    monkeypatch.setattr("smartrain.cli_support.cli_prompts.prompt", lambda *a, **k: next(answers))
    balance_main([])
    out = tmp_path / "datasets" / "ds_b_balanced"
    assert (out / "balance_manifest.json").is_file()


def test_balance_interactive_defaults_enable_manifest_creation(
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
            "",            # emit-balance-report (default yes)
            "",            # emit-train-config (default yes)
            "",            # eval-coverage (default yes)
            "n",           # dry-run
        ]
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    monkeypatch.setattr("smartrain.workflows.datasets.dataset_balance.prompt", lambda *a, **k: next(answers))
    monkeypatch.setattr("smartrain.cli_support.cli_prompts.prompt", lambda *a, **k: next(answers))
    balance_main([])
    out = tmp_path / "datasets" / "ds_b_balanced"
    manifest_path = out / "balance_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["emit_balance_report"] is True
    assert manifest["emit_train_config"] is True


def test_balance_interactive_hybrid_aug_uses_mode_defaults_in_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _prepare_workspace(tmp_path)
    answers = iter(
        [
            "ds_b",         # Dataset
            "hybrid-aug",   # Strategy
            "",             # Output
            "1.0",          # target
            "3.0",          # max-ratio
            "all",          # classes mode
            "",             # emit-balance-report (default yes)
            "",             # emit-train-config (default yes)
            "",             # eval-coverage (default yes)
            "",             # eval-min-class-count (default 0)
            "",             # aug-preset (default geo-photo)
            "",             # aug-class-aware-geo (default yes)
            "",             # aug-total-bbox-cap-mult (default 1.1)
            "",             # aug-budget-tail-first (default yes)
            "",             # aug-budget-tail-gamma (default 1.0)
            "",             # train-head-bbox-undersample (default median-factor)
            "",             # train-head-bbox-cap-mult (default 5.0)
            "",             # eval-head-bbox-undersample (default median-factor)
            "",             # eval-head-bbox-cap-mult (default 8.0)
            "",             # eval-head-bbox-min-count (default 30)
            "",             # eval-head-bbox-max-remove-frac (default 0.35)
            "",             # aug-enable-bbox-copy (default no)
            "",             # keep-hybrid-intermediate (default no)
            "y",            # dry-run
        ]
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    monkeypatch.setattr("smartrain.workflows.datasets.dataset_balance.prompt", lambda *a, **k: next(answers))
    monkeypatch.setattr("smartrain.cli_support.cli_prompts.prompt", lambda *a, **k: next(answers))
    balance_main([])
    captured = capsys.readouterr().out
    assert "--strategy hybrid-aug" in captured
    assert "--aug-total-bbox-cap-mult 1.1" in captured
    assert "--train-head-bbox-undersample median-factor" in captured
    assert "--train-head-bbox-cap-mult 5.0" in captured
    assert "--eval-head-bbox-undersample median-factor" in captured
    assert "--eval-head-bbox-cap-mult 8.0" in captured
    assert "--eval-head-bbox-min-count 30" in captured
    assert "--eval-head-bbox-max-remove-frac 0.35" in captured
    assert "--aug-budget-tail-first" in captured
    assert "--aug-budget-tail-gamma 1.0" in captured


def test_balance_ensures_non_empty_val_and_test_when_possible(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--strategy",
            "copy",
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


def test_balance_oversample_has_no_cross_split_source_duplicates(tmp_path: Path) -> None:
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
            "3.0",
        ]
    )
    out = tmp_path / "datasets" / "ds_b_balanced"
    keys = _collect_split_keys(out)
    assert keys["train"].isdisjoint(keys["val"])
    assert keys["train"].isdisjoint(keys["test"])
    assert keys["val"].isdisjoint(keys["test"])


def test_balance_weights_has_no_cross_split_source_duplicates(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--strategy",
            "weights",
            "--target",
            "3.0",
            "--max-repeat-per-image",
            "6",
        ]
    )
    out = tmp_path / "datasets" / "ds_b_balanced"
    keys = _collect_split_keys(out)
    assert keys["train"].isdisjoint(keys["val"])
    assert keys["train"].isdisjoint(keys["test"])
    assert keys["val"].isdisjoint(keys["test"])


def test_balance_rfs_has_no_cross_split_source_duplicates(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--strategy",
            "rfs",
            "--target",
            "2.0",
            "--rfs-thresh",
            "2.0",
            "--rfs-power",
            "0.5",
            "--max-repeat-per-image",
            "6",
        ]
    )
    out = tmp_path / "datasets" / "ds_b_balanced"
    keys = _collect_split_keys(out)
    assert keys["train"].isdisjoint(keys["val"])
    assert keys["train"].isdisjoint(keys["test"])
    assert keys["val"].isdisjoint(keys["test"])


def test_balance_eval_coverage_degrades_safely_when_unique_images_insufficient(tmp_path: Path) -> None:
    _prepare_workspace_tiny(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_tiny",
            "--strategy",
            "oversample",
            "--target",
            "3.0",
        ]
    )
    out = tmp_path / "datasets" / "ds_tiny_balanced"
    keys = _collect_split_keys(out)
    assert keys["train"].isdisjoint(keys["val"])
    assert keys["train"].isdisjoint(keys["test"])
    assert keys["val"].isdisjoint(keys["test"])
    val_count = len(list((out / "val" / "images").glob("*.jpg")))
    test_count = len(list((out / "test" / "images").glob("*.jpg")))
    # In tiny dataset there may be not enough unique source images
    # to fill both eval splits without leakage.
    assert (val_count == 0) or (test_count == 0)


def test_parse_class_weight_multiplier_csv() -> None:
    parsed = _parse_class_weight_multiplier("other:0.6, tear_up:1.1")
    assert parsed == {"other": 0.6, "tear_up": 1.1}


def test_auto_head_cap_multipliers_reduce_head_classes() -> None:
    multipliers = _auto_head_cap_multipliers(
        {"other": 10000, "tear_up": 8000, "tear": 900, "strings": 120, "cloudy_plastic": 80},
        quantile=0.7,
        min_mult=0.3,
    )
    assert "other" in multipliers
    assert "tear_up" in multipliers
    assert multipliers["other"] < 1.0
    assert multipliers["tear_up"] < 1.0
    assert min(multipliers.values()) >= 0.3


def test_balance_manual_multiplier_shifts_distribution(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--strategy",
            "weights",
            "--target",
            "3.0",
            "--max-repeat-per-image",
            "6",
            "--replacement",
            "on",
            "--emit-balance-report",
            "--output-name",
            "ds_b_bal_base",
        ]
    )
    base_manifest = json.loads(
        (tmp_path / "datasets" / "ds_b_bal_base" / "balance_manifest.json").read_text(encoding="utf-8")
    )
    base_counts = base_manifest["class_counts_after_bbox"]

    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--strategy",
            "weights",
            "--target",
            "3.0",
            "--max-repeat-per-image",
            "6",
            "--replacement",
            "on",
            "--class-weight-multiplier",
            "cat:0.1,dog:2.0",
            "--emit-balance-report",
            "--output-name",
            "ds_b_bal_mul",
        ]
    )
    mul_manifest = json.loads(
        (tmp_path / "datasets" / "ds_b_bal_mul" / "balance_manifest.json").read_text(encoding="utf-8")
    )
    mul_counts = mul_manifest["class_counts_after_bbox"]
    base_ratio = float(base_counts["dog"]) / max(1.0, float(base_counts["cat"]))
    mul_ratio = float(mul_counts["dog"]) / max(1.0, float(mul_counts["cat"]))
    # Under max-repeat and split-safety constraints ratio growth can saturate,
    # but manual multipliers should never make dog/cat balance worse.
    assert mul_ratio >= base_ratio
    assert mul_manifest["applied_manual_class_multipliers"] == {"cat": 0.1, "dog": 2.0}


def test_balance_auto_head_cap_enabled_by_default(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--strategy",
            "weights",
            "--emit-balance-report",
            "--output-name",
            "ds_b_bal_auto_default",
        ]
    )
    manifest = json.loads(
        (tmp_path / "datasets" / "ds_b_bal_auto_default" / "balance_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["auto_head_cap"] is True
    assert "applied_effective_class_multipliers" in manifest


def test_balance_supports_eval_min_class_count_option(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--strategy",
            "weights",
            "--eval-min-class-count",
            "2",
            "--emit-balance-report",
            "--output-name",
            "ds_b_bal_eval_min",
        ]
    )
    manifest = json.loads(
        (tmp_path / "datasets" / "ds_b_bal_eval_min" / "balance_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["eval_coverage"] is True
    assert manifest["eval_min_class_count"] == 2


def test_hybrid_aug_dry_run_skips_augment_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _prepare_workspace(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--strategy",
            "hybrid-aug",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert "hybrid-aug: skipping augment" in captured.out


def test_hybrid_aug_creates_final_with_post_augment_manifest(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--strategy",
            "hybrid-aug",
            "--target",
            "1.0",
            "--emit-balance-report",
            "--output-name",
            "ds_b_ha",
        ]
    )
    info = json.loads((tmp_path / "datasets" / "datasets_info.json").read_text(encoding="utf-8"))
    final_keys = [k for k in info if k.startswith("ds_b_ha") and "__hybrid" not in k]
    assert final_keys, "expected final balanced_aug dataset in catalog"
    out = tmp_path / "datasets" / final_keys[0]
    manifest = json.loads((out / "balance_manifest.json").read_text(encoding="utf-8"))
    assert manifest["strategy"] == "hybrid-aug"
    assert manifest["post_augment"] is not None
    assert manifest["post_augment"]["preset"] == "geo-photo"
    assert manifest["post_augment"]["class_aware_geo"] is True
    assert manifest["post_augment"]["total_bbox_cap_mult"] == 1.1
    assert manifest["post_augment"]["budget_tail_first"] is True
    assert manifest["post_augment"]["budget_tail_gamma"] == 1.0
    assert manifest["train_head_bbox_undersample"] == "median-factor"
    assert manifest["eval_head_bbox_undersample"] == "median-factor"
    assert manifest["head_bbox_undersample"] is not None
    assert manifest["eval_head_bbox_undersample_stats"] is not None
    assert isinstance(manifest["post_augment"].get("train_bbox_sum_before_augment"), int)
    assert isinstance(manifest["post_augment"].get("train_bbox_sum_after_augment"), int)
    argv_sum = manifest["post_augment"].get("argv_summary") or []
    assert any("enable-flip" in str(x) for x in argv_sum)
    assert "--aug-class-aware-geo" in argv_sum
    assert manifest.get("hybrid_intermediate_name") is None
    assert (out / "train" / "images").is_dir()
    hybrid_keys = [k for k in info if "__hybrid" in k]
    assert not hybrid_keys, "intermediate hybrid dataset should be removed by default"


def test_head_bbox_undersample_median_factor(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    balance_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_b",
            "--strategy",
            "hybrid",
            "--target",
            "1.0",
            "--emit-balance-report",
            "--output-name",
            "ds_b_head",
            "--train-head-bbox-undersample",
            "median-factor",
            "--train-head-bbox-cap-mult",
            "0.1",
        ]
    )
    out = tmp_path / "datasets" / "ds_b_head"
    manifest = json.loads((out / "balance_manifest.json").read_text(encoding="utf-8"))
    h = manifest.get("head_bbox_undersample")
    assert h is not None
    assert h["mode"] == "median-factor"
    assert "per_class" in h

