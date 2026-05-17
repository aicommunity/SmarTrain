from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.core.runtime.path_portable import relativize_if_under
from smartrain.core.runtime.run_artifacts import preferred_run_model_path


def build_migrate_models_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Migration of legacy artifacts to the format of smartrain run directories (adds training_metadata.json)."
    )
    p.add_argument(
        "--models-root",
        type=str,
        required=True,
        help="The root of the directory with legacy models (search for run directories using train/results.csv).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing training_metadata.json.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what will be done, without writing files.",
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="If set and the dataset path lies under this root, store path_under_workspace instead of path_absolute.",
    )
    return p


def infer_run_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for results_csv in root.rglob("train/results.csv"):
        run_dir = results_csv.parent.parent
        if run_dir.is_dir():
            out.append(run_dir)
    uniq = sorted(set(p.resolve() for p in out))
    return [Path(p) for p in uniq]


def load_args_yaml(run_dir: Path) -> dict[str, Any]:
    args_path = run_dir / "train" / "args.yaml"
    if not args_path.exists():
        return {}
    with args_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _dataset_abs_under_workspace(dataset_abs: str, workspace_root: str | None) -> bool:
    if not workspace_root or not dataset_abs:
        return False
    a = os.path.abspath(dataset_abs)
    b = os.path.abspath(os.path.expanduser(workspace_root))
    return a == b or a.startswith(b + os.sep)


def build_metadata(
    run_dir: Path, args_data: dict[str, Any], workspace_root: str | None = None
) -> dict[str, Any]:
    model_raw = args_data.get("model", "unknown")
    model_name = Path(str(model_raw)).stem

    data_arg = args_data.get("data")
    if data_arg:
        dataset_name = Path(str(data_arg)).parent.name or Path(str(data_arg)).stem
        dataset_abs = str(Path(str(data_arg)).resolve())
    else:
        dataset_name = run_dir.parent.name
        dataset_abs = ""

    dataset_block: dict[str, Any] = {
        "name": dataset_name,
        "path_relative": ".",
        "hash": None,
    }
    if dataset_abs:
        if _dataset_abs_under_workspace(dataset_abs, workspace_root):
            rel = relativize_if_under(workspace_root, dataset_abs)
            if rel is not None:
                dataset_block["path_under_workspace"] = rel
        else:
            dataset_block["path_absolute"] = dataset_abs

    epochs = args_data.get("epochs")
    batch = args_data.get("batch")
    imgsz = args_data.get("imgsz")

    return {
        "training_info": {
            "framework": "ultralytics",
            "task_type": "detection",
            "model": model_name,
            "dataset": dataset_block,
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch,
                "image_size": imgsz,
            },
        },
        "timestamps": {
            "training": {
                "start": None,
                "end": None,
                "duration_seconds": None,
            },
            "testing": {
                "start": None,
                "end": None,
                "duration_seconds": None,
            },
        },
        "status": {
            "training": {
                "success": True,
                "error": None,
            },
            "testing": {
                "success": (run_dir / "test_metrics.csv").exists(),
                "error": None,
            },
        },
        "paths": {
            "model_directory": ".",
            "best_model": f"{run_dir.name}.pt" if Path(preferred_run_model_path(str(run_dir), ".pt")).exists() else None,
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = build_migrate_models_arg_parser().parse_args(argv)
    models_root = Path(args.models_root).expanduser().resolve()

    if not models_root.exists():
        raise SystemExit(f"[ERROR] Folder not found: {models_root}")

    run_dirs = infer_run_dirs(models_root)
    if not run_dirs:
        raise SystemExit("[ERROR] No run directory found with train/results.csv")

    created = 0
    skipped = 0
    updated = 0

    for run_dir in run_dirs:
        meta_path = run_dir / "training_metadata.json"
        existed_before = meta_path.exists()
        if existed_before and not args.overwrite:
            skipped += 1
            print(f"[SKIP] Already exists: {meta_path}")
            continue

        args_data = load_args_yaml(run_dir)
        ws = (args.workspace or "").strip() or None
        metadata = build_metadata(run_dir, args_data, workspace_root=ws)

        if args.dry_run:
            action = "UPDATE" if existed_before else "CREATE"
            print(f"[DRY-RUN] {action}: {meta_path}")
            continue

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        if existed_before and args.overwrite:
            updated += 1
            print(f"[OK] Updated: {meta_path}")
        else:
            created += 1
            print(f"[OK] Created by: {meta_path}")

    print(
        f"\nDone. created={created}, updated={updated}, skipped={skipped}, total={len(run_dirs)}"
    )


if __name__ == "__main__":
    main()
