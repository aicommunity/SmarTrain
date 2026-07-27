from __future__ import annotations

from pathlib import Path

import pytest

from smartrain.external_providers import probe as probe_mod
from smartrain.external_providers.probe import probe_provider_repo
from smartrain.external_providers.registry import get_provider_spec


def test_probe_provider_repo_detects_entrypoints(tmp_path: Path) -> None:
    spec = get_provider_spec("dr-yolo")
    (tmp_path / "train.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "detect.py").write_text("print('infer')\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("torch\n", encoding="utf-8")
    res = probe_provider_repo(str(tmp_path), spec)
    assert res["repo_found"] is True
    assert res["entrypoints_ok"] is True


def test_probe_mfel_reports_missing_dcnv4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    spec = get_provider_spec("mfel-yolo")
    (tmp_path / "train.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "val.py").write_text("print('infer')\n", encoding="utf-8")
    if sys.platform == "win32":
        venv_bin = tmp_path / "venv" / "Scripts"
        venv_bin.mkdir(parents=True, exist_ok=True)
        (venv_bin / "python.exe").write_text("", encoding="utf-8")
    else:
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True, exist_ok=True)
        (venv_bin / "python").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    class _R:
        returncode = 1
        stderr = "ModuleNotFoundError: No module named 'DCNv4'"

    monkeypatch.setattr(probe_mod.subprocess, "run", lambda *a, **k: _R())
    res = probe_provider_repo(str(tmp_path), spec, str(tmp_path / "venv"))
    assert res["runtime_ok"] is False
    assert "DCNv4" in str(res["runtime_reason"])

