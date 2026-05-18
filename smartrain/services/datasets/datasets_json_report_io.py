from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from smartrain.core.runtime.workspace_paths import (
    DATASETS_INFO_FILE,
    CLASS_NAMES_FILE,
    DATASETS_SCAN_SUMMARY_FILE,
)

OUTPUT_FILE = DATASETS_INFO_FILE
OUTPUT_CLASS_NAMES_FILE = CLASS_NAMES_FILE

# Dataset record fields saved during rescanning (manually in JSON).
_PRESERVED_DATASET_INFO_KEYS: Tuple[str, ...] = (
    "roi_auto",
    "tags",
    "data_path",
    "source_signature",
    "dataset_hash",
    "source_hash",
    "source_ref",
    "modified",
)


def _merge_preserved_dataset_fields(
    fresh: Dict[str, Any], previous: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Adds a fresh record from a scan with the roi_auto/tags fields from the old datasets_info.json."""
    if not previous:
        return fresh
    out = dict(fresh)
    for key in _PRESERVED_DATASET_INFO_KEYS:
        if key in previous:
            out[key] = previous[key]
    return out


def _write_scan_summary(
    *,
    output_dir: str,
    datasets_final: Dict[str, Any],
    class_names_final: Dict[str, Any],
    datasets_added: List[str],
    datasets_removed: List[str],
    class_names_added: List[str],
    class_names_removed: List[str],
) -> str:
    """Writes datasets_scan_summary.json; returns the file path."""
    path = os.path.join(output_dir, DATASETS_SCAN_SUMMARY_FILE)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "final": sorted(datasets_final.keys()),
            "count": len(datasets_final),
            "added": datasets_added,
            "removed": datasets_removed,
        },
        "class_names": {
            "final": sorted(class_names_final.keys()),
            "count": len(class_names_final),
            "added": class_names_added,
            "removed": class_names_removed,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _print_scan_report(
    *,
    summary_path: str,
    datasets_added: List[str],
    datasets_removed: List[str],
    class_names_added: List[str],
    class_names_removed: List[str],
    datasets_final_count: int,
    class_names_final_count: int,
    had_previous_datasets: bool,
    had_previous_class_names: bool,
) -> None:
    print(f"[INFO] Final datasets in {OUTPUT_FILE}: {datasets_final_count}")
    if datasets_added:
        print(
            f"[INFO] Added datasets ({len(datasets_added)}): {', '.join(datasets_added)}"
        )
    if datasets_removed:
        print(
            f"[INFO] Removed from catalog ({len(datasets_removed)}): {', '.join(datasets_removed)}"
        )
    if not datasets_added and not datasets_removed and had_previous_datasets:
        print("[INFO] The composition of the datasets relative to the previous file has not changed.")
    print(
        f"[INFO] Final class names in {OUTPUT_CLASS_NAMES_FILE}: {class_names_final_count}"
    )
    if class_names_added:
        print(
            f"[INFO] New class names ({len(class_names_added)}): {', '.join(class_names_added)}"
        )
    if class_names_removed:
        print(
            f"[INFO] Removed class names ({len(class_names_removed)}): {', '.join(class_names_removed)}"
        )
    if (
        not class_names_added
        and not class_names_removed
        and had_previous_class_names
    ):
        print("[INFO] The composition of class_names relative to the previous file has not changed.")
    print(f"[OK] Summary saved: {summary_path}")

