from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from smartrain.core.runtime.workspace_paths import (
    archive_kind,
    extract_dataset_archive_to_cache,
    is_dataset_archive_path,
)
from smartrain.services.datasets.dataset_source_resolver import (
    MANUAL_SOURCE_OPTION,
    build_interactive_source_options,
    list_raw_data_candidates,
    peek_archive_structure,
    resolve_dataset_source,
    resolve_manual_source_token,
)


def _write_jpg(path: Path, *, size: tuple[int, int] = (200, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(30, 40, 50)).save(path, format="JPEG", quality=90)


def _write_cvsdcldet_sample(source: Path) -> None:
    _write_jpg(source / "img_a.jpg", size=(200, 100))
    (source / "img_a.json").write_text(
        json.dumps(
            {
                "detections": [
                    {"classId": 0, "class_name": "cat", "x": 10, "y": 20, "width": 50, "height": 30, "score": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )


def _make_cvsdcldet_zip(tmp_path: Path, *, inner_dir: str = "sample") -> Path:
    source = tmp_path / "build" / inner_dir
    source.mkdir(parents=True)
    _write_cvsdcldet_sample(source)
    zip_path = tmp_path / f"{inner_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(source.parent)))
    return zip_path


def _make_cvsdcldet_tar_gz(tmp_path: Path, *, inner_dir: str = "sample") -> Path:
    source = tmp_path / "build" / inner_dir
    source.mkdir(parents=True)
    _write_cvsdcldet_sample(source)
    archive_path = tmp_path / f"{inner_dir}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tf:
        for path in source.rglob("*"):
            if path.is_file():
                tf.add(path, arcname=str(path.relative_to(source.parent)))
    return archive_path


def test_is_dataset_archive_path() -> None:
    assert is_dataset_archive_path("data.zip")
    assert is_dataset_archive_path("data.tar.gz")
    assert is_dataset_archive_path("data.tgz")
    assert not is_dataset_archive_path("data.json")


def test_archive_kind() -> None:
    assert archive_kind("a.zip") == "zip"
    assert archive_kind("a.tar.gz") == "tar.gz"
    assert archive_kind("a.tgz") == "tgz"
    assert archive_kind("a.tar") == "tar"


def test_list_raw_data_candidates_dir_and_zip(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    raw = ws / "raw_data"
    raw.mkdir(parents=True)
    source_dir = raw / "folder_set"
    source_dir.mkdir()
    _write_cvsdcldet_sample(source_dir)
    zip_path = _make_cvsdcldet_zip(tmp_path)
    (raw / zip_path.name).write_bytes(zip_path.read_bytes())

    candidates = list_raw_data_candidates(str(ws))
    labels = [c.label for c in candidates]
    assert any("[raw_data] folder_set/" in label for label in labels)
    assert any("folder_set.zip" in label or "sample.zip" in label for label in labels)


def test_resolve_manual_source_token_raw_data_name(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    raw = ws / "raw_data"
    raw.mkdir(parents=True)
    zip_path = _make_cvsdcldet_zip(tmp_path)
    target = raw / zip_path.name
    target.write_bytes(zip_path.read_bytes())

    resolved = resolve_manual_source_token(str(ws), zip_path.stem)
    assert resolved == target.resolve()


def test_resolve_dataset_source_cvsdcldet_zip(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    zip_path = _make_cvsdcldet_zip(tmp_path)
    resolved = resolve_dataset_source(str(ws), zip_path)
    assert resolved.structure == "cvsdcldet"
    assert resolved.source_archive == zip_path.resolve()
    assert resolved.working_path.is_dir()


def test_resolve_dataset_source_cvsdcldet_tar_gz(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    archive_path = _make_cvsdcldet_tar_gz(tmp_path)
    resolved = resolve_dataset_source(str(ws), archive_path)
    assert resolved.structure == "cvsdcldet"
    assert resolved.source_archive == archive_path.resolve()
    assert resolved.working_path.is_dir()


def test_peek_archive_structure_cvat11_zip(tmp_path: Path) -> None:
    task = tmp_path / "task1"
    images = task / "images"
    images.mkdir(parents=True)
    _write_jpg(images / "img001.jpg")
    ann = task / "annotations.xml"
    ann.write_text("<annotations></annotations>", encoding="utf-8")
    zip_path = tmp_path / "cvat.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(ann, arcname="task1/annotations.xml")
        zf.write(images / "img001.jpg", arcname="task1/images/img001.jpg")
    assert peek_archive_structure(zip_path) == "cvat11_zip"


def test_resolve_dataset_source_cvat11_zip_without_extract(tmp_path: Path) -> None:
    task = tmp_path / "task1"
    images = task / "images"
    images.mkdir(parents=True)
    _write_jpg(images / "img001.jpg")
    ann = task / "annotations.xml"
    ann.write_text("<annotations></annotations>", encoding="utf-8")
    zip_path = tmp_path / "cvat.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(ann, arcname="task1/annotations.xml")
        zf.write(images / "img001.jpg", arcname="task1/images/img001.jpg")
    resolved = resolve_dataset_source(None, zip_path)
    assert resolved.is_cvat11_zip
    assert resolved.structure == "cvat11_zip"
    assert resolved.working_path == zip_path.resolve()


def test_external_candidates_from_datasets_list(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    raw = ws / "raw_data"
    raw.mkdir(parents=True)
    zip_path = _make_cvsdcldet_zip(tmp_path)
    external = tmp_path / "external.zip"
    external.write_bytes(zip_path.read_bytes())
    (raw / "datasets_list.txt").write_text(f"{external}\n", encoding="utf-8")

    from smartrain.services.datasets.dataset_source_resolver import list_external_candidates

    candidates = list_external_candidates(str(ws))
    assert len(candidates) == 1
    assert candidates[0].group == "external"
    assert candidates[0].structure_hint == "cvsdcldet"


def test_build_interactive_source_options_includes_manual(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "raw_data").mkdir(parents=True)
    (ws / "datasets").mkdir(parents=True)
    (ws / "datasets" / "datasets_info.json").write_text("{}", encoding="utf-8")
    candidates, manual = build_interactive_source_options(str(ws))
    assert manual == MANUAL_SOURCE_OPTION
    assert any(c.label == MANUAL_SOURCE_OPTION for c in candidates)


def test_extract_dataset_archive_to_cache_tar_gz(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    archive_path = _make_cvsdcldet_tar_gz(tmp_path)
    root1 = extract_dataset_archive_to_cache(str(ws), str(archive_path))
    root2 = extract_dataset_archive_to_cache(str(ws), str(archive_path))
    assert root1 == root2
    assert Path(root1).is_dir()


def test_unsafe_tar_path_raises(tmp_path: Path) -> None:
    from smartrain.core.runtime.workspace_paths import _safe_extract_tar

    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as tf:
        info = tarfile.TarInfo(name="../escape.txt")
        tf.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="unsafe path"):
        _safe_extract_tar(str(archive_path), str(tmp_path / "out"))
