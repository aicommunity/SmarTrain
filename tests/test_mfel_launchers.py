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


def test_mfel_train_launcher_resolves_custom_model_aliases(tmp_path: Path) -> None:
    repo = tmp_path / "mfel"
    cfg_dir = repo / "ultralytics" / "cfg" / "MFEL-YOLO"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "MFEL-YOLO.yaml").write_text("nc: 1\n", encoding="utf-8")
    (cfg_dir / "E_PAN+.yaml").write_text("nc: 1\n", encoding="utf-8")

    resolved_main = mfel_train_launcher._resolve_mfel_model_spec(repo, "mfel-yolo")
    resolved_epan = mfel_train_launcher._resolve_mfel_model_spec(repo, "e_pan+")
    assert resolved_main.endswith("MFEL-YOLO.yaml")
    assert resolved_epan.endswith("E_PAN+.yaml")
