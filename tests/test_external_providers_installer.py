from __future__ import annotations

import zipfile
from pathlib import Path

from smartrain.external_providers.installer import _resolve_ssdm_repo_root
from smartrain.external_providers.installer import _resolve_effective_repo_dir


def test_resolve_ssdm_repo_root_unpacks_archive(tmp_path: Path) -> None:
    repo_dir = tmp_path / "ssdm-yolo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    archive = repo_dir / "SSDM-YOLO.zip"

    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("SSDM-YOLO/train.py", "print('train')\n")
        zf.writestr("SSDM-YOLO/detect.py", "print('detect')\n")
        zf.writestr("SSDM-YOLO/requirements.txt", "ultralytics\n")

    resolved = _resolve_ssdm_repo_root(repo_dir)

    assert (resolved / "train.py").is_file()
    assert (resolved / "detect.py").is_file()
    assert str(resolved).startswith(str(repo_dir / "_smartrain_unpacked"))


def test_resolve_ssdm_repo_root_is_idempotent_without_archive_change(tmp_path: Path) -> None:
    repo_dir = tmp_path / "ssdm-yolo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    archive = repo_dir / "SSDM-YOLO.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("SSDM-YOLO/train.py", "print('train')\n")
        zf.writestr("SSDM-YOLO/detect.py", "print('detect')\n")

    first = _resolve_ssdm_repo_root(repo_dir)
    sentinel = first / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    second = _resolve_ssdm_repo_root(repo_dir)
    assert first == second
    assert sentinel.is_file()


def test_resolve_enhanced_repo_root_detects_nested_code_dir(tmp_path: Path) -> None:
    repo_dir = tmp_path / "enhanced-yolov8"
    nested = repo_dir / "yolov8-main-Ghost"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "train.py").write_text("print('train')\n", encoding="utf-8")
    (nested / "detect.py").write_text("print('detect')\n", encoding="utf-8")

    resolved = _resolve_effective_repo_dir("enhanced-yolov8", repo_dir)
    assert resolved == nested
