from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from smartrain.dataset_stats import _scan_one_dataset, main as stats_main
from smartrain.workspace_paths import deploy_workspace


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(path, format="JPEG", quality=85)


def _write_ds(root: Path, name: str) -> Path:
    ds = root / "datasets" / name
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: cat\n  1: dog\n",
        encoding="utf-8",
    )
    return ds


def test_stats_classes_counts_and_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    deploy_workspace(str(tmp_path))
    ds = _write_ds(tmp_path, "ds_a")
    _write_jpg(ds / "train" / "images" / "a.jpg")
    (ds / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    _write_jpg(ds / "val" / "images" / "b.jpg")
    (ds / "val" / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "val" / "labels" / "b.txt").write_text("0 0.4 0.4 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(SystemExit) as e:
        stats_main(["--workspace", str(tmp_path), "classes", "--dataset", "ds_a"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "Статистика по классам" in out
    assert "cat" in out
    assert "dog" in out
    assert "Итог по дисбалансу" in out


def test_stats_datasets_quality_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    deploy_workspace(str(tmp_path))
    ds = _write_ds(tmp_path, "ds_warn")
    _write_jpg(ds / "train" / "images" / "a.jpg")
    (ds / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\nbad line\n", encoding="utf-8")
    (ds / "train" / "labels" / "orphan.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(SystemExit) as e:
        stats_main(["--workspace", str(tmp_path), "datasets", "--dataset", "ds_warn"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "Статистика по датасетам" in out
    assert "WARN" in out


def test_stats_no_legend_flag_hides_column_explanations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deploy_workspace(str(tmp_path))
    ds = _write_ds(tmp_path, "ds_a")
    _write_jpg(ds / "train" / "images" / "a.jpg")
    (ds / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(SystemExit) as e:
        stats_main(["--workspace", str(tmp_path), "classes", "--dataset", "ds_a", "--no-legend"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "Колонки classes:" not in out

    with pytest.raises(SystemExit) as e:
        stats_main(["--workspace", str(tmp_path), "datasets", "--dataset", "ds_a", "--no-legend"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "Колонки datasets:" not in out


def test_stats_datasets_flat_and_cvat11_non_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    deploy_workspace(str(tmp_path))
    flat = _write_ds(tmp_path, "ds_flat")
    _write_jpg(flat / "images" / "a.jpg")
    (flat / "labels").mkdir(parents=True, exist_ok=True)
    (flat / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    cvat = _write_ds(tmp_path, "ds_cvat")
    _write_jpg(cvat / "images" / "c1.jpg")
    (cvat / "annotations.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="c1.jpg" width="32" height="24">
    <box label="cat" occluded="0" source="manual" xtl="1" ytl="1" xbr="10" ybr="10" z_order="0"/>
  </image>
</annotations>
""",
        encoding="utf-8",
    )

    flat_stats = _scan_one_dataset(str(flat), "ds_flat")
    cvat_stats = _scan_one_dataset(str(cvat), "ds_cvat")
    assert flat_stats.images_total == 1
    assert flat_stats.instances_total == 1
    assert cvat_stats.images_total == 1
    assert cvat_stats.instances_total == 1

    with pytest.raises(SystemExit) as e:
        stats_main(["--workspace", str(tmp_path), "datasets", "--dataset", "ds_flat"])
    assert e.value.code == 0
    with pytest.raises(SystemExit) as e:
        stats_main(["--workspace", str(tmp_path), "datasets", "--dataset", "ds_cvat"])
    assert e.value.code == 0


def test_stats_classes_interactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    ds = _write_ds(tmp_path, "ds_i")
    _write_jpg(ds / "train" / "images" / "a.jpg")
    (ds / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def _fake_prompt(args, available_names, available_classes):
        args.dataset = ["ds_i"]
        args.classes = None
        args.sort = "total"
        args.desc = False
        args.limit = None

    monkeypatch.setattr("smartrain.dataset_stats._prompt_interactive_classes", _fake_prompt)
    with pytest.raises(SystemExit) as e:
        stats_main(["--workspace", str(tmp_path), "classes"])
    assert e.value.code == 0


def test_stats_interactive_prints_datasets_and_classes_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    deploy_workspace(str(tmp_path))
    ds = _write_ds(tmp_path, "ds_i_print")
    _write_jpg(ds / "train" / "images" / "a.jpg")
    (ds / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def _fake_prompt(args, available_names, available_classes):
        args.dataset = ["ds_i_print"]
        args.classes = None
        args.sort = "total"
        args.desc = False
        args.limit = None

    monkeypatch.setattr("smartrain.dataset_stats._prompt_interactive_classes", _fake_prompt)
    with pytest.raises(SystemExit) as e:
        stats_main(["--workspace", str(tmp_path), "classes"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "Доступные датасеты:" in out
    assert "ds_i_print" in out
    assert "Доступные классы:" in out
    assert "cat" in out


def test_stats_datasets_interactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    ds = _write_ds(tmp_path, "ds_i2")
    _write_jpg(ds / "train" / "images" / "a.jpg")
    (ds / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def _fake_prompt(args, available_names):
        args.dataset = ["ds_i2"]
        args.sort = "images"
        args.desc = False
        args.check_duplicates = False
        args.check_near_duplicates = False
        args.export_issues = False

    monkeypatch.setattr("smartrain.dataset_stats._prompt_interactive_datasets", _fake_prompt)
    with pytest.raises(SystemExit) as e:
        stats_main(["--workspace", str(tmp_path), "datasets"])
    assert e.value.code == 0

