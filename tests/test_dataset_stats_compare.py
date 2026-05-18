from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from smartrain.services.datasets.dataset_stats import main as stats_main
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(path, format="JPEG", quality=85)


def _write_ds(root: Path, name: str, cls_names: list[str]) -> Path:
    ds = root / "datasets" / name
    ds.mkdir(parents=True, exist_ok=True)
    names_map = "\n".join([f"  {i}: {n}" for i, n in enumerate(cls_names)])
    (ds / "data.yaml").write_text(
        f"train: train/images\nval: val/images\ntest: test/images\nnames:\n{names_map}\n",
        encoding="utf-8",
    )
    return ds


def test_stats_compare_summary_and_classes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    deploy_workspace(str(tmp_path))
    left = _write_ds(tmp_path, "left_ds", ["cat", "dog"])
    right = _write_ds(tmp_path, "right_ds", ["cat", "bird"])

    _write_jpg(left / "train" / "images" / "a.jpg")
    (left / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (left / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    _write_jpg(right / "train" / "images" / "b.jpg")
    (right / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (right / "train" / "labels" / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_jpg(right / "val" / "images" / "c.jpg")
    (right / "val" / "labels").mkdir(parents=True, exist_ok=True)
    (right / "val" / "labels" / "c.txt").write_text("0 0.4 0.4 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(SystemExit) as e:
        stats_main(
            [
                "compare",
                "--workspace",
                str(tmp_path),
                "--left",
                "left_ds",
                "--right",
                "right_ds",
                "--details",
                "all",
            ]
        )
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "Dataset comparison" in out
    assert "Diff by common classes" in out
    assert "Only in left" in out
    assert "Only in right" in out


def test_stats_compare_export_json_csv(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    left = _write_ds(tmp_path, "left_ds", ["cat"])
    right = _write_ds(tmp_path, "right_ds", ["cat"])
    _write_jpg(left / "train" / "images" / "a.jpg")
    (left / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (left / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_jpg(right / "train" / "images" / "b.jpg")
    (right / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (right / "train" / "labels" / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(SystemExit) as e:
        stats_main(
            [
                "compare",
                "--workspace",
                str(tmp_path),
                "--left",
                "left_ds",
                "--right",
                "right_ds",
                "--export-json",
                "--export-csv",
            ]
        )
    assert e.value.code == 0
    stats_dir = tmp_path / "analytics" / "stats"
    exported = sorted(stats_dir.glob("*/compare_report.json"))
    assert exported, "compare_report.json not exported"
    payload = json.loads(exported[-1].read_text(encoding="utf-8"))
    assert "summary" in payload and "classes" in payload
    csv_exported = sorted(stats_dir.glob("*/compare_classes.csv"))
    assert csv_exported, "compare_classes.csv not exported"


def test_stats_compare_interactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deploy_workspace(str(tmp_path))
    left = _write_ds(tmp_path, "left_ds", ["cat"])
    right = _write_ds(tmp_path, "right_ds", ["cat"])
    _write_jpg(left / "train" / "images" / "a.jpg")
    (left / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (left / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_jpg(right / "train" / "images" / "b.jpg")
    (right / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (right / "train" / "labels" / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))

    def _fake_prompt(args, available_names):
        assert "left_ds" in available_names and "right_ds" in available_names
        args.left = "left_ds"
        args.right = "right_ds"
        args.details = "summary"
        args.top_n = None
        args.abs = False
        args.export_json = False
        args.export_csv = False

    monkeypatch.setattr("smartrain.services.datasets.dataset_stats.prompt_interactive_compare_args", _fake_prompt)
    with pytest.raises(SystemExit) as e:
        stats_main(["compare"])
    assert e.value.code == 0

