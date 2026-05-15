from __future__ import annotations

import os
import re
import shutil
from typing import Any, Callable

from smartrain.workflows.testing.ultralytics_test_contract import ULTRALYTICS_TEST_COLLECT_IMAGE_NAMES


def collect_ultralytics_test_artifacts(
    session_root: str,
    run_dirs: list[str],
    abbreviations: dict[str, str],
    *,
    run_test_backend_dir_cb: Callable[[str, str], str],
    build_run_record_canonical_cb: Callable[[str], Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = []
    out_root = os.path.join(session_root, "artifacts", "ultralytics-test")
    for rd in run_dirs:
        run_name = os.path.basename(rd.rstrip(os.sep))
        run_code = abbreviations.get(run_name, run_name)
        preferred_test_dir = str(run_test_backend_dir_cb(rd, "ultralytics"))
        legacy_test_dir = os.path.join(rd, "test")
        root_ultra_dir = os.path.join(rd, "test-ultralytics")
        candidates: list[str] = []
        for d in (preferred_test_dir, legacy_test_dir, root_ultra_dir):
            if os.path.isdir(d) and d not in candidates:
                candidates.append(d)
        if os.path.isdir(preferred_test_dir):
            test_dir = preferred_test_dir
        elif os.path.isdir(legacy_test_dir):
            test_dir = legacy_test_dir
        elif os.path.isdir(root_ultra_dir):
            test_dir = root_ultra_dir
        else:
            test_dir = preferred_test_dir
        rec = build_run_record_canonical_cb(rd)
        row: dict[str, Any] = {
            "run_dir": rd,
            "run_name": run_name,
            "run_code": run_code,
            "test_dir": test_dir,
            "exists": bool(candidates),
            "run_info": {
                "model": rec.model,
                "dataset_name": rec.dataset_name,
                "epochs": None,
                "batch_size": None,
                "train_image_size": None,
                "val_imgsz": None,
            },
            "machine_info": {},
            "files": [],
            "csv": {},
            "images": [],
        }
        if not candidates:
            rows.append(row)
            continue
        safe_code = re.sub(r"[^\w.\-+]+", "_", str(run_code), flags=re.UNICODE).strip("._") or "run"
        dst_dir = os.path.join(out_root, safe_code)
        os.makedirs(dst_dir, exist_ok=True)
        csv_names = ("pr.csv", "pr_per_class.csv")
        image_patterns = ULTRALYTICS_TEST_COLLECT_IMAGE_NAMES

        def _find_src(rel_name: str) -> str | None:
            for d in candidates:
                p = os.path.join(d, rel_name)
                if os.path.isfile(p):
                    return p
            return None

        for name in csv_names:
            src = _find_src(name)
            if src is None:
                continue
            dst = os.path.join(dst_dir, name)
            try:
                shutil.copy2(src, dst)
                rel = os.path.relpath(dst, session_root)
                row["csv"][name] = rel
                artifacts.append({"role": f"ultralytics_test_{name.replace('.', '_')}", "path": rel})
            except Exception:
                pass
        for name in image_patterns:
            src = _find_src(name)
            if src is None:
                continue
            dst = os.path.join(dst_dir, name)
            try:
                shutil.copy2(src, dst)
                rel = os.path.relpath(dst, session_root)
                row["images"].append(rel)
                artifacts.append({"role": "ultralytics_test_image", "path": rel})
            except Exception:
                pass
        merged: set[str] = set()
        for d in candidates:
            try:
                merged.update(os.listdir(d))
            except OSError:
                pass
        row["files"] = sorted(merged)
        rows.append(row)
    return rows, artifacts

