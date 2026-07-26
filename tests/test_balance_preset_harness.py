"""Fixture test for balance preset harness."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace
from smartrain.workflows.datasets.datasets_json_former import main as scan_main
from scripts.balance_preset_harness import run_harness


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(path, format="JPEG", quality=85)


def test_balance_preset_harness_writes_table(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_harness"
    (raw / "train" / "images").mkdir(parents=True)
    (raw / "train" / "labels").mkdir(parents=True)
    for i, cls in enumerate([0, 0, 0, 1]):
        _write_jpg(raw / "train" / "images" / f"i{i}.jpg")
        (raw / "train" / "labels" / f"i{i}.txt").write_text(
            f"{cls} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    (raw / "data.yaml").write_text("nc: 2\nnames: ['a','b']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])

    result = run_harness(
        workspace=str(tmp_path),
        dataset="ds_harness",
        presets=["weights-safe", "rfs-aggressive", "hybrid-default", "irfs-default"],
        seed=0,
    )
    assert Path(result["csv_path"]).is_file()
    assert Path(result["json_path"]).is_file()
    assert len(result["presets"]) == 4
    csv_text = Path(result["csv_path"]).read_text(encoding="utf-8")
    assert "irfs-default" in csv_text
    assert "images_before" in csv_text
