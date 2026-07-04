from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace
from smartrain.services.datasets.dataset_scan_preflight import (
    run_quiet_workspace_scan,
    should_run_auto_scan,
    try_resolve_workspace_from_argv,
)


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(path, format="JPEG", quality=85)


def _prepare_raw_dataset(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_a"
    (raw / "train" / "images").mkdir(parents=True, exist_ok=True)
    (raw / "train" / "labels").mkdir(parents=True, exist_ok=True)
    _write_jpg(raw / "train" / "images" / "a.jpg")
    (raw / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 1\nnames: ['cat']\n", encoding="utf-8")


def test_should_run_auto_scan_false_for_help(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    assert should_run_auto_scan(["--help"]) is False
    assert should_run_auto_scan(["-h"]) is False


def test_should_run_auto_scan_false_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    assert should_run_auto_scan(["--dataset", "ds_a"], auto_scan_disabled=True) is False


def test_should_run_auto_scan_false_for_legacy_fusion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    argv = [
        "--source-path",
        "/a",
        "--target-path",
        "/b",
        "--datasets-info-path",
        "/c",
    ]
    assert should_run_auto_scan(argv) is False


def test_should_run_auto_scan_true_for_workspace_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    assert should_run_auto_scan(["--workspace", str(tmp_path), "--dataset", "ds_a"]) is True


def test_try_resolve_workspace_from_argv_uses_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    assert try_resolve_workspace_from_argv(["--dataset", "ds_a"]) == str(tmp_path.resolve())


def test_run_quiet_workspace_scan_creates_catalog(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _prepare_raw_dataset(tmp_path)
    info_path = tmp_path / "datasets" / "datasets_info.json"
    assert info_path.read_text(encoding="utf-8").strip() in ("", "{}")

    ok = run_quiet_workspace_scan(workspace_root=str(tmp_path))

    captured = capsys.readouterr()
    out = captured.out
    assert ok is True
    assert "[INFO] Running dataset scan…" in out
    assert "[INFO] Dataset scan completed." in out
    assert "[OK] Information saved" not in out
    assert "ds_a" in info_path.read_text(encoding="utf-8")


def test_cli_augment_auto_scan_populates_catalog(
    tmp_path: Path,
    subprocess_env: dict[str, str],
) -> None:
    import subprocess
    import sys as sys_mod

    _prepare_raw_dataset(tmp_path)
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path)

    r_disabled = subprocess.run(
        [
            sys_mod.executable,
            "-m",
            "smartrain",
            "augment",
            "--no-auto-scan",
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_a",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "datasets_info.json was not found or is empty" in r_disabled.stderr + r_disabled.stdout

    r_enabled = subprocess.run(
        [
            sys_mod.executable,
            "-m",
            "smartrain",
            "augment",
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_a",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = r_enabled.stderr + r_enabled.stdout
    assert "Running dataset scan" in combined
    assert "Dataset scan completed" in combined
    assert "datasets_info.json was not found or is empty" not in combined
    assert (tmp_path / "datasets" / "ds_a_aug").is_dir()
