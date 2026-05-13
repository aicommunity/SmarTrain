from __future__ import annotations

import hashlib
import os
import zipfile
from typing import Any, Dict, List, Optional, Set, Tuple

from smartrain.core.runtime.workspace_paths import (
    WorkspaceLayout,
    resolve_or_extract_dataset_root,
)

from smartrain.workflows.datasets.datasets_json_normalize_service import (
    _normalize_path_for_data_path,
)


def _sorted_diff(old: Set[str], new: Set[str]) -> Tuple[List[str], List[str]]:
    added = sorted(new - old)
    removed = sorted(old - new)
    return added, removed


def _dir_has_content(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    with os.scandir(path) as it:
        for _ in it:
            return True
    return False


def _run_scan_folder_roots(
    folder_roots: list[tuple[str, str, dict]],
    *,
    process_dataset_cb,
) -> tuple[dict, dict]:
    """folder_roots: list (logical_name, folder_path on disk, overrides for datasets_info)."""
    datasets_info: dict = {}
    class_names: dict = {}

    for logical_name, folder_path, overrides in folder_roots:
        if not os.path.isdir(folder_path):
            print(f"[WARNING] Skipping {logical_name!r}: no directory {folder_path}")
            continue
        info = process_dataset_cb(folder_path, logical_name)
        if info:
            if overrides:
                info.update(overrides)
            datasets_info[logical_name] = info
            for class_name in info["classes"]:
                class_names[class_name] = class_name
    return datasets_info, class_names


def _load_datasets_list_file(list_path: str) -> list[str]:
    entries: list[str] = []
    list_dir = os.path.dirname(list_path)
    with open(list_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            expanded = os.path.expanduser(line)
            if not os.path.isabs(expanded):
                expanded = os.path.join(list_dir, expanded)
            entries.append(os.path.abspath(expanded))
    return entries


def _unique_dataset_key(base_name: str, used_names: set[str]) -> str:
    key = base_name or "dataset"
    if key not in used_names:
        used_names.add(key)
        return key
    idx = 2
    while f"{key}_{idx}" in used_names:
        idx += 1
    unique = f"{key}_{idx}"
    used_names.add(unique)
    return unique


def _zip_extract_path(temp_root: str, zip_path: str) -> str:
    abs_zip = os.path.abspath(zip_path)
    sig = hashlib.sha1(abs_zip.encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(abs_zip))[0]
    return os.path.join(temp_root, f"{stem}_{sig}")


def _extract_zip_for_scan(zip_path: str, temp_root: str) -> str:
    os.makedirs(temp_root, exist_ok=True)
    out_dir = _zip_extract_path(temp_root, zip_path)
    marker = os.path.join(out_dir, ".extract_done")
    if os.path.isfile(marker):
        return out_dir
    if os.path.isdir(out_dir):
        for root, dirs, files in os.walk(out_dir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok")
    return out_dir


def _append_roots_from_datasets_list(
    *,
    list_path: str,
    folder_roots: list[tuple[str, str, dict]],
    used_names: set[str],
    use_workspace: bool,
    layout: Optional[WorkspaceLayout],
    output_dir: str,
) -> None:
    entries = _load_datasets_list_file(list_path)
    workspace_root = layout.root if layout else None
    for src_path in entries:
        if not os.path.exists(src_path):
            print(f"[WARNING] Skipping from datasets-list: path not found: {src_path}")
            continue

        base_name = (
            os.path.splitext(os.path.basename(src_path))[0]
            if src_path.lower().endswith(".zip")
            else os.path.basename(src_path)
        )
        logical_name = _unique_dataset_key(base_name, used_names)
        data_path_value = _normalize_path_for_data_path(src_path, workspace_root)

        if os.path.isdir(src_path):
            folder_roots.append((logical_name, src_path, {"data_path": data_path_value}))
            continue

        if src_path.lower().endswith(".zip"):
            try:
                if use_workspace and layout:
                    extracted = resolve_or_extract_dataset_root(
                        layout.root,
                        logical_name,
                        {"data_path": data_path_value},
                        layout.raw_data,
                    )
                else:
                    extracted = _extract_zip_for_scan(
                        src_path,
                        os.path.join(output_dir, "tmp", "datasets_list_extract"),
                    )
            except Exception as e:
                print(f"[WARNING] Skipping archives from datasets-list {src_path!r}: {e}")
                continue
            folder_roots.append((logical_name, extracted, {"data_path": data_path_value}))
            continue

        print(
            "[WARNING] Skipping from datasets-list: only directories and .zip are supported,"
            f"received: {src_path}"
        )


def _compute_source_signature(path: str) -> str:
    ap = os.path.abspath(path)
    if os.path.isfile(ap) and ap.lower().endswith(".zip"):
        st = os.stat(ap)
        payload = f"zip|{ap}|{st.st_size}|{getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9))}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    rows: list[str] = []
    for root, dirs, files in os.walk(ap):
        dirs.sort()
        files.sort()
        rel_root = os.path.relpath(root, ap)
        rows.append(f"d:{rel_root}")
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, ap)
            rows.append(f"f:{rel}:{st.st_size}:{getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9))}")

    joined = "\n".join(rows).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()[:16]

