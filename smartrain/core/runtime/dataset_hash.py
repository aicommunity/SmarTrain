"""Dataset content-independent structure hash."""

from __future__ import annotations

import hashlib
import os


def calculate_dataset_hash(dataset_path: str) -> str:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    if not os.path.isdir(dataset_path):
        raise ValueError(f"The specified path is not a folder: {dataset_path}")
    hasher = hashlib.md5()
    ignored_files = {".DS_Store", "Thumbs.db", ".gitkeep", ".gitignore"}
    root_path = os.path.abspath(dataset_path)
    prefix_length = len(root_path) + 1
    items: list[tuple[str, str] | tuple[str, str, int]] = []
    for root, directories, files in os.walk(root_path):
        directories.sort()
        files.sort()
        relative_root = root[prefix_length:] if len(root) > prefix_length else ""
        for directory in directories:
            relative_path = os.path.join(relative_root, directory) if relative_root else directory
            items.append(("dir", relative_path))
        for file_name in files:
            if file_name in ignored_files:
                continue
            relative_path = os.path.join(relative_root, file_name) if relative_root else file_name
            try:
                items.append(("file", relative_path, os.path.getsize(os.path.join(root, file_name))))
            except OSError:
                continue
    items.sort()
    for item in items:
        hasher.update(item[0].encode("utf-8"))
        hasher.update(b":")
        hasher.update(item[1].encode("utf-8"))
        if item[0] == "file":
            hasher.update(b":")
            hasher.update(str(item[2]).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:8]
