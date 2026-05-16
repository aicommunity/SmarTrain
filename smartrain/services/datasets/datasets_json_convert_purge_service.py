from __future__ import annotations

import os
import shutil
import sys
from typing import Optional, Tuple

from smartrain.services.datasets.dataset_hash import calculate_dataset_hash


def _copy_source_to_training(
    src_root: str,
    dst_root: str,
    *,
    ensure_training_ready_after_copy_cb,
) -> None:
    if os.path.isdir(dst_root):
        shutil.rmtree(dst_root, ignore_errors=True)
    os.makedirs(os.path.dirname(dst_root), exist_ok=True)
    shutil.copytree(src_root, dst_root)
    ensure_training_ready_after_copy_cb(dst_root)


def _dataset_content_hash(path: str) -> Optional[str]:
    try:
        return str(calculate_dataset_hash(path))
    except Exception as e:
        print(f"[WARNING] Failed to calculate dataset_hash for {path!r}: {e}")
        return None


def _confirm_purge_processed_raw(paths: list[str]) -> bool:
    if not paths:
        return False
    print("[WARNING] Requested to remove processed sources from raw_data.")
    print("[WARNING] Will be removed:")
    for p in paths:
        print(f"  - {p}")
    if not sys.stdin.isatty():
        print("[WARNING] No interactive TTY: deletion cancelled.")
        return False
    ans = input("Continue deletion? [Y/n]: ").strip().lower()
    return ans in ("", "y", "yes", "1", "true", "yes", "d")


def _purge_raw_sources(paths: list[str]) -> Tuple[int, int]:
    removed = 0
    failed = 0
    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
                removed += 1
            elif os.path.isfile(p):
                os.remove(p)
                removed += 1
        except Exception as e:
            failed += 1
            print(f"[WARNING] Failed to delete {p}: {e}")
    return removed, failed

