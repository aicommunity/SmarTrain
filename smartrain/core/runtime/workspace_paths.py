"""
Single root workspace: subdirectories and resolution of paths to datasets (data_path / directory by key).
"""
from __future__ import annotations

import json
import os
import hashlib
import shutil
import tarfile
import zipfile
from typing import Any

from smartrain.core.runtime.path_portable import relativize_if_under, resolve_stored_path_under_workspace

WORKSPACE_ENV_VAR = "SMART_TRAIN_WORKSPACE"

DATASETS_INFO_FILE = "datasets_info.json"
CLASS_NAMES_FILE = "class_names.json"
DATASETS_SCAN_SUMMARY_FILE = "datasets_scan_summary.json"
WORKSPACE_QUEUE_BASENAME = "queue.txt"

DATASET_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


class WorkspaceLayout:
    """All standard paths inside the workspace; root is the absolute normalized path."""

    def __init__(self, root: str):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.raw_data = os.path.join(self.root, "raw_data")
        self.datasets = os.path.join(self.root, "datasets")
        # Backward-compatible aliases for legacy code paths.
        self.source_datasets = self.raw_data
        self.work_datasets = self.datasets
        self.runs = os.path.join(self.root, "runs")
        self.analytics = os.path.join(self.root, "analytics")
        self.models = os.path.join(self.root, "models")
        self.inference = os.path.join(self.root, "inference")
        self.extracted_datasets = os.path.join(self.root, "tmp", "extracted_datasets")

    def source_datasets_info_path(self) -> str:
        return os.path.join(self.source_datasets, DATASETS_INFO_FILE)

    def source_class_names_path(self) -> str:
        return os.path.join(self.source_datasets, CLASS_NAMES_FILE)

    def work_datasets_info_path(self) -> str:
        return os.path.join(self.work_datasets, DATASETS_INFO_FILE)

    def work_class_names_path(self) -> str:
        return os.path.join(self.work_datasets, CLASS_NAMES_FILE)


def resolve_workspace_root(cli_workspace: str | None) -> str:
    """
    Root source: CLI argument (non-empty) overrides the SMART_TRAIN_WORKSPACE environment variable.
    Otherwise, it’s a clear mistake.
    """
    from smartrain.core.runtime.run_artifacts import reject_documentation_placeholder_path

    if cli_workspace is not None:
        w = cli_workspace.strip()
        if w:
            reject_documentation_placeholder_path(w, kind="workspace")
            return os.path.abspath(os.path.expanduser(w))
    env_val = os.environ.get(WORKSPACE_ENV_VAR)
    if env_val is not None:
        e = env_val.strip()
        if e:
            reject_documentation_placeholder_path(e, kind="workspace")
            return os.path.abspath(os.path.expanduser(e))
    raise ValueError(
        "The workspace root is not set: specify --workspace or an environment variable"
        f"{WORKSPACE_ENV_VAR}."
    )


def resolve_path_under_workspace(workspace_root: str, relative_or_absolute: str) -> str:
    """Absolute path as is; otherwise the path is relative to workspace_root."""
    p = relative_or_absolute.strip()
    if not p:
        raise ValueError("Empty data_path.")
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(workspace_root, os.path.normpath(p)))


def resolve_dataset_root(
    workspace_root: str,
    entry_key: str,
    entry_dict: dict,
    catalog_dir: str,
) -> str:
    """
    If the record contains the data_path key, resolve from workspace or absolute.
    Otherwise, data root: catalog_dir/entry_key.
    """
    if "data_path" in entry_dict:
        raw = entry_dict["data_path"]
        if not isinstance(raw, str):
            raise TypeError(f"data_path for {entry_key!r} must be a string.")
        return resolve_path_under_workspace(workspace_root, raw)
    return os.path.join(catalog_dir, entry_key)


def is_dataset_archive_path(path: str | os.PathLike[str]) -> bool:
    """Return True when path looks like a supported dataset archive."""
    name = os.path.basename(str(path)).lower()
    return any(name.endswith(suffix) for suffix in DATASET_ARCHIVE_SUFFIXES)


def archive_kind(path: str | os.PathLike[str]) -> str:
    """Return archive kind: zip, tar, tar.gz, or tgz."""
    name = os.path.basename(str(path)).lower()
    if name.endswith(".tar.gz"):
        return "tar.gz"
    if name.endswith(".tgz"):
        return "tgz"
    if name.endswith(".tar"):
        return "tar"
    if name.endswith(".zip"):
        return "zip"
    raise ValueError(f"Unsupported dataset archive: {path}")


def _safe_extract_zip(zip_path: str, target_dir: str) -> None:
    """
    Secure zip unpacking to target_dir with path traversal protection.
    """
    abs_target = os.path.abspath(target_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_name = member.filename
            out_path = os.path.abspath(os.path.join(abs_target, member_name))
            if not out_path.startswith(abs_target + os.sep) and out_path != abs_target:
                raise ValueError(f"The archive contains an unsafe path: {member_name!r}")
        zf.extractall(abs_target)


def _safe_extract_tar(archive_path: str, target_dir: str) -> None:
    """Secure tar/tar.gz unpacking to target_dir with path traversal protection."""
    abs_target = os.path.abspath(target_dir)
    kind = archive_kind(archive_path)
    mode = "r:gz" if kind in ("tar.gz", "tgz") else "r:"
    with tarfile.open(archive_path, mode) as tf:
        members = [m for m in tf.getmembers() if m.isfile() or m.isdir()]
        for member in members:
            out_path = os.path.abspath(os.path.join(abs_target, member.name))
            if not out_path.startswith(abs_target + os.sep) and out_path != abs_target:
                raise ValueError(f"The archive contains an unsafe path: {member.name!r}")
        for member in members:
            tf.extract(member, path=abs_target)


def _safe_extract_archive(archive_path: str, target_dir: str) -> None:
    kind = archive_kind(archive_path)
    if kind == "zip":
        _safe_extract_zip(archive_path, target_dir)
        return
    _safe_extract_tar(archive_path, target_dir)


def _choose_extracted_dataset_root(extract_dir: str) -> str:
    """
    If there is one top-level directory in the archive, we use it as the root of the dataset.
    Otherwise, we use the unpacking directory itself.
    """
    try:
        entries = [name for name in os.listdir(extract_dir) if name != "__meta__.json"]
    except FileNotFoundError:
        return extract_dir
    dirs = [name for name in entries if os.path.isdir(os.path.join(extract_dir, name))]
    files = [name for name in entries if os.path.isfile(os.path.join(extract_dir, name))]
    if len(dirs) == 1 and not files:
        return os.path.join(extract_dir, dirs[0])
    return extract_dir


def _resolved_archive_path_from_meta(workspace_root: str, meta: dict[str, Any]) -> str | None:
    raw = meta.get("archive_path")
    if not isinstance(raw, str) or not raw.strip():
        raw = meta.get("zip_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    if os.path.isabs(s):
        return os.path.abspath(s)
    try:
        return resolve_stored_path_under_workspace(workspace_root, s)
    except (OSError, ValueError):
        return None


def extract_dataset_archive_to_cache(workspace_root: str, archive_path: str) -> str:
    """
    Unpacks a dataset archive into workspace/tmp/extracted_datasets cache with invalidation
    by size and mtime. Returns the path to the root of the unpacked dataset.
    """
    abs_archive = os.path.abspath(os.path.expanduser(archive_path))
    if not os.path.isfile(abs_archive):
        raise FileNotFoundError(f"Dataset archive not found: {abs_archive}")
    if not is_dataset_archive_path(abs_archive):
        raise ValueError(
            f"Unsupported dataset archive: {abs_archive} "
            f"(supported: {', '.join(DATASET_ARCHIVE_SUFFIXES)})"
        )
    stat = os.stat(abs_archive)
    key_src = f"{abs_archive}|{stat.st_size}|{stat.st_mtime_ns}"
    cache_key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]

    layout = WorkspaceLayout(workspace_root)
    wr_abs = layout.root
    cache_root = layout.extracted_datasets
    cache_dir = os.path.join(cache_root, cache_key)
    meta_path = os.path.join(cache_dir, "__meta__.json")
    os.makedirs(cache_root, exist_ok=True)

    if os.path.isdir(cache_dir) and os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta_archive = _resolved_archive_path_from_meta(workspace_root, meta)
            if (
                meta_archive == abs_archive
                and meta.get("size") == stat.st_size
                and meta.get("mtime_ns") == stat.st_mtime_ns
            ):
                root_rel = meta.get("dataset_root_rel", "")
                root = os.path.join(cache_dir, root_rel) if root_rel else cache_dir
                if os.path.isdir(root):
                    return root
        except Exception:
            pass

    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    _safe_extract_archive(abs_archive, cache_dir)
    dataset_root = _choose_extracted_dataset_root(cache_dir)
    rel_root = os.path.relpath(dataset_root, cache_dir)

    archive_stored: str = abs_archive
    rel_archive = relativize_if_under(wr_abs, abs_archive)
    if isinstance(rel_archive, str) and rel_archive != abs_archive:
        archive_stored = rel_archive

    kind = archive_kind(abs_archive)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "archive_path": archive_stored,
                "archive_kind": kind,
                "zip_path": archive_stored if kind == "zip" else None,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "dataset_root_rel": "" if rel_root == "." else rel_root,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return dataset_root


def extract_dataset_zip_to_cache(workspace_root: str, zip_path: str) -> str:
    """Backward-compatible wrapper around extract_dataset_archive_to_cache for zip archives."""
    return extract_dataset_archive_to_cache(workspace_root, zip_path)


def resolve_or_extract_dataset_root(
    workspace_root: str,
    entry_key: str,
    entry_dict: dict,
    catalog_dir: str,
) -> str:
    """
    Like resolve_dataset_root, but if the path points to a dataset archive, returns
    the root of the unpacked dataset from the workspace cache.
    """
    root = resolve_dataset_root(workspace_root, entry_key, entry_dict, catalog_dir)
    if is_dataset_archive_path(root):
        return extract_dataset_archive_to_cache(workspace_root, root)
    return root


def workspace_queue_path(workspace_root: str) -> str:
    """Learning queue file in workspace root (`queue.txt`)."""
    root = os.path.abspath(os.path.expanduser(workspace_root))
    return os.path.join(root, WORKSPACE_QUEUE_BASENAME)


def workspace_queue_status_path(workspace_root: str) -> str:
    """Queue worker statuses: `workspace/tmp/status.txt`."""
    root = os.path.abspath(os.path.expanduser(workspace_root))
    return os.path.join(root, "tmp", "status.txt")


def deploy_workspace(target_root: str | None = None) -> dict[str, Any]:
    """
    Creates workspace directories and empty datasets_info.json if missing.
    target_root by default is the current directory (same as a custom workspace).
    """
    root = os.path.abspath(os.path.expanduser(target_root or os.getcwd()))
    layout = WorkspaceLayout(root)
    created_dirs: list[str] = []
    created_files: list[str] = []
    skipped: list[str] = []

    dir_specs = [
        ("raw_data", layout.raw_data),
        ("datasets", layout.datasets),
        ("runs", layout.runs),
        ("analytics", layout.analytics),
        ("models", layout.models),
        ("inference", layout.inference),
        ("tmp", os.path.join(root, "tmp")),
        ("extracted_datasets", layout.extracted_datasets),
    ]
    for name, dpath in dir_specs:
        if os.path.isdir(dpath):
            skipped.append(f"dir:{name}")
        else:
            os.makedirs(dpath, exist_ok=True)
            created_dirs.append(name)

    file_specs = [
        ("source_datasets_info", layout.source_datasets_info_path()),
        ("work_datasets_info", layout.work_datasets_info_path()),
        ("datasets_list", os.path.join(layout.raw_data, "datasets_list.txt")),
    ]
    for label, fpath in file_specs:
        if os.path.isfile(fpath):
            skipped.append(f"file:{label}")
        else:
            with open(fpath, "w", encoding="utf-8") as f:
                if label == "datasets_list":
                    f.write("")
                else:
                    json.dump({}, f, ensure_ascii=False, indent=2)
            created_files.append(label)

    return {
        "root": root,
        "created_dirs": created_dirs,
        "created_files": created_files,
        "skipped": skipped,
    }
