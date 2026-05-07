from __future__ import annotations

import os
import re
import shutil
from typing import Any, Callable


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
        test_dir = preferred_test_dir if os.path.isdir(preferred_test_dir) else legacy_test_dir
        rec = build_run_record_canonical_cb(rd)
        row: dict[str, Any] = {
            "run_dir": rd,
            "run_name": run_name,
            "run_code": run_code,
            "test_dir": test_dir,
            "exists": os.path.isdir(test_dir),
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
        if not os.path.isdir(test_dir):
            rows.append(row)
            continue
        safe_code = re.sub(r"[^\w.\-+]+", "_", str(run_code), flags=re.UNICODE).strip("._") or "run"
        dst_dir = os.path.join(out_root, safe_code)
        os.makedirs(dst_dir, exist_ok=True)
        csv_names = ("pr.csv", "pr_per_class.csv")
        image_patterns = (
            "PR_curve.png",
            "BoxPR_curve.png",
            "F1_curve.png",
            "BoxF1_curve.png",
            "P_curve.png",
            "BoxP_curve.png",
            "R_curve.png",
            "BoxR_curve.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "val_batch0_pred.jpg",
            "val_batch0_labels.jpg",
        )
        for name in csv_names:
            src = os.path.join(test_dir, name)
            if not os.path.isfile(src):
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
            src = os.path.join(test_dir, name)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(dst_dir, name)
            try:
                shutil.copy2(src, dst)
                rel = os.path.relpath(dst, session_root)
                row["images"].append(rel)
                artifacts.append({"role": "ultralytics_test_image", "path": rel})
            except Exception:
                pass
        row["files"] = sorted(os.listdir(test_dir))
        rows.append(row)
    return rows, artifacts

