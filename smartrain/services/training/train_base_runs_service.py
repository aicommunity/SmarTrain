from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def extract_run_timestamp(run_name: str, run_dir: Path) -> datetime:
    match = re.match(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})", run_name)
    if match:
        try:
            return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y-%m-%d_%H-%M")
        except ValueError:
            pass
    return datetime.fromtimestamp(run_dir.stat().st_mtime)


def base_run_summary(args_yaml: Path, *, load_ultralytics_yaml_cb) -> dict[str, str]:
    summary = {
        "provider": "unknown",
        "model": "unknown",
        "task": "unknown",
        "batch": "?",
        "epochs": "?",
    }
    try:
        payload = load_ultralytics_yaml_cb(str(args_yaml))
    except Exception:
        return summary
    provider = (
        str(payload.get("external_provider") or "").strip().lower()
        if isinstance(payload, dict)
        else ""
    )
    summary["provider"] = provider or "ultralytics"
    if isinstance(payload, dict):
        model = str(payload.get("model") or "").strip()
        task = str(payload.get("task") or "").strip().lower()
        batch = payload.get("batch")
        epochs = payload.get("epochs")
        if model:
            summary["model"] = Path(model).name
        if task:
            summary["task"] = task
        if batch is not None and str(batch).strip():
            summary["batch"] = str(batch)
        if epochs is not None and str(epochs).strip():
            summary["epochs"] = str(epochs)
    return summary


def collect_available_base_runs(layout, selected_dataset: str, *, base_run_summary_cb) -> list[dict[str, str]]:
    out: list[dict[str, Any]] = []
    runs_root = Path(layout.runs)
    if not runs_root.is_dir():
        return []
    for ds_dir in sorted(runs_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_name = ds_dir.name
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            args_train = run_dir / "train-ultralytics" / "args.yaml"
            legacy_args_train = run_dir / "train" / "args.yaml"
            args_root = run_dir / "args.yaml"
            args_path: Path | None = None
            if args_train.is_file():
                args_path = args_train
            elif legacy_args_train.is_file():
                args_path = legacy_args_train
            elif args_root.is_file():
                args_path = args_root
            if args_path is None:
                continue
            if not run_dir.exists() or not args_path.exists():
                continue
            run_ts = extract_run_timestamp(run_dir.name, run_dir)
            run_rel = run_dir.relative_to(runs_root).as_posix()
            info = base_run_summary_cb(args_path)
            out.append(
                {
                    "dataset": ds_name,
                    "run_dir": str(run_dir),
                    "args_yaml": str(args_path),
                    "run_rel": run_rel,
                    "provider": info["provider"],
                    "model": info["model"],
                    "task": info["task"],
                    "batch": info["batch"],
                    "epochs": info["epochs"],
                    "_sort_ts": run_ts,
                }
            )
    out.sort(
        key=lambda x: (
            x["dataset"] != selected_dataset,
            x["_sort_ts"],
            x["run_rel"],
        )
    )
    for row in out:
        row.pop("_sort_ts", None)
    return out


def print_available_base_runs(selected_dataset: str, runs: list[dict[str, str]]) -> None:
    if not runs:
        print("[INFO] No base runs found in runs/.")
        return
    print("[INFO] Available base runs (selected dataset first, oldest first):")
    switched_to_other = False
    for i, row in enumerate(runs, start=1):
        if not switched_to_other and row["dataset"] != selected_dataset:
            print("      ---- other datasets ----")
            switched_to_other = True
        mark = " [selected-dataset]" if row["dataset"] == selected_dataset else ""
        task_part = (
            f" task:{row['task']}"
            if row.get("task", "unknown") not in ("", "detect", "unknown")
            else ""
        )
        print(
            f"  {i:>3}. {row.get('run_rel', row['run_dir'])}{mark}"
            f" | provider:{row.get('provider', 'unknown')}"
            f" | model:{row.get('model', 'unknown')}"
            f" | b={row.get('batch', '?')} e={row.get('epochs', '?')}{task_part}"
        )


def prompt_base_run_args_yaml(
    runs: list[dict[str, str]],
    default_path: str | None = None,
    *,
    prompt_input_cb,
) -> str | None:
    if not runs:
        return default_path
    while True:
        raw = prompt_input_cb(
            "Base run (number or path to args.yaml, empty=no base): ",
            default=str(default_path or ""),
        ).strip()
        if not raw:
            return default_path
        if os.path.isfile(raw):
            return raw
        try:
            idx = int(raw)
        except ValueError:
            print(f"[ERROR] Expected run number or path to args.yaml, received: {raw!r}")
            continue
        if 1 <= idx <= len(runs):
            return runs[idx - 1]["args_yaml"]
        print(f"[ERROR] Number out of range 1..{len(runs)}")

