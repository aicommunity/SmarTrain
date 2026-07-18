from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from smartrain.services.datasets.dataset_passport import write_dataset_passport
from smartrain.core.runtime.path_portable import (
    is_abs_like,
    posix_relpath,
    relativize_abs_paths_in_obj,
    relativize_if_under,
    resolve_stored_path_under_workspace,
    store_path_under_workspace,
    to_posix,
)
from smartrain.core.runtime.workspace_paths import extract_dataset_zip_to_cache


def test_is_abs_like_posix_and_drive() -> None:
    assert is_abs_like("/data/x") is True
    assert is_abs_like("C:\\x") is True
    assert is_abs_like("C:/x") is True
    assert is_abs_like("datasets/a") is False
    assert is_abs_like("") is False


def test_posix_relpath_uses_forward_slash(tmp_path: Path) -> None:
    a = tmp_path / "datasets" / "a"
    a.mkdir(parents=True)
    rel = posix_relpath(str(a), str(tmp_path))
    assert "\\" not in rel
    assert rel == "datasets/a"


def test_to_posix() -> None:
    assert to_posix(r"models\ds\run") == "models/ds/run"


def test_store_path_under_workspace_inside(tmp_path: Path) -> None:
    ws = str(tmp_path.resolve())
    sub = tmp_path / "datasets" / "a"
    sub.mkdir(parents=True)
    assert store_path_under_workspace(ws, str(sub.resolve())) == "datasets/a"


def test_store_path_under_workspace_outside(tmp_path: Path) -> None:
    ws = str(tmp_path.resolve())
    other = tmp_path.parent / "outside_ws_only"
    other.mkdir(exist_ok=True)
    stored = store_path_under_workspace(ws, str(other.resolve()))
    assert stored == str(other.resolve()) or os.path.isabs(stored)


def test_resolve_stored_accepts_backslash_legacy(tmp_path: Path) -> None:
    ws = str(tmp_path.resolve())
    (tmp_path / "datasets" / "a").mkdir(parents=True)
    got = resolve_stored_path_under_workspace(ws, r"datasets\a")
    assert Path(got).resolve() == (tmp_path / "datasets" / "a").resolve()


def test_resolve_roundtrip_posix(tmp_path: Path) -> None:
    ws = str(tmp_path.resolve())
    sub = tmp_path / "datasets" / "x"
    sub.mkdir(parents=True)
    rel = store_path_under_workspace(ws, str(sub.resolve()))
    assert resolve_stored_path_under_workspace(ws, rel) == str(sub.resolve())


def test_relativize_if_under_inside(tmp_path: Path) -> None:
    ws = str(tmp_path.resolve())
    sub = tmp_path / "datasets" / "a"
    sub.mkdir(parents=True)
    assert relativize_if_under(ws, str(sub.resolve())) == "datasets/a"


def test_relativize_if_under_outside_unchanged(tmp_path: Path) -> None:
    ws = str(tmp_path.resolve())
    other = "/other/root/file"
    assert relativize_if_under(ws, other) == other


def test_resolve_stored_roundtrip(tmp_path: Path) -> None:
    ws = str(tmp_path.resolve())
    rel = "datasets/x"
    assert resolve_stored_path_under_workspace(ws, rel) == str((tmp_path / "datasets" / "x").resolve())


def test_relativize_abs_paths_in_obj_nested(tmp_path: Path) -> None:
    ws = str(tmp_path.resolve())
    p = str((tmp_path / "raw" / "z.zip").resolve())
    os.makedirs(os.path.dirname(p), exist_ok=True)
    Path(p).write_bytes(b"x")
    obj = {"a": 1, "p": p, "nested": {"q": p}}
    out = relativize_abs_paths_in_obj(obj, ws)
    assert out["p"] == "raw/z.zip"
    assert out["nested"]["q"] == "raw/z.zip"


def test_write_dataset_passport_workspace_relative(tmp_path: Path) -> None:
    ws = str(tmp_path.resolve())
    out_ds = tmp_path / "datasets" / "ds1"
    out_ds.mkdir(parents=True)
    src = tmp_path / "raw" / "in.zip"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"")
    write_dataset_passport(
        output_dataset_dir=str(out_ds),
        command="test",
        source_datasets=[{"name": "in", "path": str(src.resolve()), "dataset_hash": "h"}],
        parameters={"workspace": ws},
        transformations=[],
        workspace_root=ws,
    )
    data = json.loads((out_ds / "dataset_passport.json").read_text(encoding="utf-8"))
    assert data["created_dataset"]["path"] == "datasets/ds1"
    assert data["source_dataset"][0]["path"] == "raw/in.zip"
    assert data["parameters"]["workspace"] == "."


def test_extract_zip_cache_meta_uses_relative_zip_path(tmp_path: Path) -> None:
    ws = tmp_path
    raw = ws / "raw_data"
    raw.mkdir(parents=True)
    zpath = raw / "blob.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("one/img.txt", "ok")
    root1 = extract_dataset_zip_to_cache(str(ws), str(zpath))
    assert (Path(root1) / "img.txt").is_file()
    cache_dirs = list((ws / "tmp" / "extracted_datasets").iterdir())
    assert len(cache_dirs) == 1
    meta = json.loads((cache_dirs[0] / "__meta__.json").read_text(encoding="utf-8"))
    assert meta["zip_path"] == "raw_data/blob.zip"
    assert "\\" not in str(meta.get("dataset_root_rel") or "")
    root2 = extract_dataset_zip_to_cache(str(ws), str(zpath))
    assert Path(root1).resolve() == Path(root2).resolve()


def test_extract_zip_cache_hit_with_legacy_absolute_zip_path_in_meta(tmp_path: Path) -> None:
    ws = tmp_path
    raw = ws / "raw_data"
    raw.mkdir(parents=True)
    zpath = raw / "legacy.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a/b.txt", "x")
    abs_zip = str(zpath.resolve())
    stat = zpath.stat()
    key_src = f"{abs_zip}|{stat.st_size}|{stat.st_mtime_ns}"
    import hashlib

    cache_key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]
    cache_dir = ws / "tmp" / "extracted_datasets" / cache_key
    cache_dir.mkdir(parents=True)
    (cache_dir / "a").mkdir()
    (cache_dir / "a" / "b.txt").write_text("x", encoding="utf-8")
    (cache_dir / "__meta__.json").write_text(
        json.dumps(
            {
                "zip_path": abs_zip,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "dataset_root_rel": "a",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    root = extract_dataset_zip_to_cache(str(ws), str(zpath))
    assert (Path(root) / "b.txt").read_text(encoding="utf-8") == "x"
