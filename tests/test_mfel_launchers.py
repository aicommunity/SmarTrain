from __future__ import annotations

import builtins
from pathlib import Path

from smartrain.external_providers.launchers import mfel_infer_launcher, mfel_train_launcher


def test_mfel_train_launcher_reports_missing_dcnv4(monkeypatch, tmp_path: Path, capsys) -> None:
    repo = tmp_path / "mfel"
    repo.mkdir(parents=True, exist_ok=True)

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ultralytics":
            raise ModuleNotFoundError("No module named 'DCNv4'", name="DCNv4")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    rc = mfel_train_launcher.main(
        ["--repo", str(repo), "--data", str(repo / "data.yaml"), "--model", "yolov8n.pt"]
    )
    assert rc == 2
    assert "DCNv4" in capsys.readouterr().err


def test_mfel_infer_launcher_reports_missing_dcnv4(monkeypatch, tmp_path: Path, capsys) -> None:
    repo = tmp_path / "mfel"
    repo.mkdir(parents=True, exist_ok=True)

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ultralytics":
            raise ModuleNotFoundError("No module named 'DCNv4'", name="DCNv4")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    rc = mfel_infer_launcher.main(
        ["--repo", str(repo), "--model", "yolov8n.pt", "--source", str(repo / "images")]
    )
    assert rc == 2
    assert "DCNv4" in capsys.readouterr().err
