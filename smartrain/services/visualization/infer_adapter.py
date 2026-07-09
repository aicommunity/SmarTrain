from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.services.inference_arg_parser import build_inference_arg_parser
from smartrain.services.inference_service import run_inference_job


def _latest_inference_report(layout: WorkspaceLayout) -> Path:
    all_reports = sorted(Path(layout.root).glob("inference/**/inference_results.json"))
    if not all_reports:
        raise FileNotFoundError("No inference_results.json found after inference run.")
    return all_reports[-1]


def run_inference_for_split(
    *,
    layout: WorkspaceLayout,
    weights_path: Path,
    split_dir: Path,
    device: str | None,
    conf: float | None,
    limit: int | None,
) -> dict[str, list[dict[str, Any]]]:
    parser = build_inference_arg_parser()
    argv = [
        "--workspace",
        str(layout.root),
        "--weights",
        str(weights_path),
        "--data-mode",
        "folder",
        "--source-dir",
        str(split_dir),
    ]
    if conf is not None:
        argv.extend(["--conf", str(conf)])
    if device:
        argv.extend(["--device", str(device)])
    if limit and int(limit) > 0:
        argv.extend(["--limit", str(int(limit))])
    args = parser.parse_args(argv)
    code, _always_exit = run_inference_job(args, layout)
    if code != 0:
        raise RuntimeError(f"Inference failed for split {split_dir} with code={code}")
    report_path = _latest_inference_report(layout)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        return {}
    by_path: dict[str, list[dict[str, Any]]] = {}
    for row in images:
        if not isinstance(row, dict):
            continue
        ap = row.get("image_path_absolute") or row.get("image_path") or row.get("source_path")
        if not isinstance(ap, str) or not ap.strip():
            continue
        dets = row.get("detections")
        if not isinstance(dets, list):
            dets = []
        by_path[str(Path(ap).resolve())] = [x for x in dets if isinstance(x, dict)]
    return by_path

