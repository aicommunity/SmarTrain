from __future__ import annotations

import json
import os
import shutil
from typing import Any


def collect_eval_dataset_test_artifacts(
    session_root: str,
    run_dirs: list[str],
    abbreviations: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = []
    out_root = os.path.join(session_root, "artifacts", "eval-dataset-test")
    for rd in run_dirs:
        run_name = os.path.basename(rd.rstrip(os.sep))
        run_code = abbreviations.get(run_name, run_name)
        manifest_path = os.path.join(rd, "tests", "test_artifacts_manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            payload = json.loads(open(manifest_path, encoding="utf-8").read())
        except Exception:
            continue
        eval_datasets = payload.get("eval_datasets")
        if not isinstance(eval_datasets, dict):
            continue
        for slot_key, slot in eval_datasets.items():
            if not isinstance(slot, dict):
                continue
            formats = slot.get("formats")
            if not isinstance(formats, dict):
                continue
            fmt_pt = formats.get("pt")
            if not isinstance(fmt_pt, dict):
                continue
            row: dict[str, Any] = {
                "run_name": run_name,
                "run_code": run_code,
                "slot_key": str(slot_key),
                "dataset_yaml": str(slot.get("dataset_yaml") or ""),
                "source": str(slot.get("source") or ""),
                "metrics_csv": str(fmt_pt.get("metrics_csv") or ""),
                "test_dir": str(fmt_pt.get("test_dir") or ""),
                "status": str(fmt_pt.get("status") or ""),
            }
            rel_csv = row["metrics_csv"]
            if rel_csv:
                src_csv = os.path.join(rd, rel_csv)
                if os.path.isfile(src_csv):
                    dst_dir = os.path.join(out_root, run_code, str(slot_key))
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_csv = os.path.join(dst_dir, "test_metrics.csv")
                    shutil.copy2(src_csv, dst_csv)
                    rel = os.path.relpath(dst_csv, session_root)
                    row["session_metrics_csv"] = rel
                    artifacts.append({"role": "eval_dataset_test_metrics_csv", "path": rel})
            rows.append(row)
    return rows, artifacts

