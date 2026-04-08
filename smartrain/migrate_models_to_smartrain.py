from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from smartrain.cli_argparse import CliArgumentParser


def build_migrate_models_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Миграция legacy-артефактов в формат run-каталогов smartrain (добавляет training_metadata.json)."
    )
    p.add_argument(
        "--models-root",
        type=str,
        required=True,
        help="Корень каталога с legacy-моделями (поиск run-каталогов по train/results.csv).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать существующие training_metadata.json.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать, что будет сделано, без записи файлов.",
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


def build_metadata(run_dir: Path, args_data: dict[str, Any]) -> dict[str, Any]:
    model_raw = args_data.get("model", "unknown")
    model_name = Path(str(model_raw)).stem

    data_arg = args_data.get("data")
    if data_arg:
        dataset_name = Path(str(data_arg)).parent.name or Path(str(data_arg)).stem
        dataset_abs = str(Path(str(data_arg)).resolve())
    else:
        dataset_name = run_dir.parent.name
        dataset_abs = ""

    epochs = args_data.get("epochs")
    batch = args_data.get("batch")
    imgsz = args_data.get("imgsz")

    return {
        "training_info": {
            "framework": "ultralytics",
            "task_type": "detection",
            "model": model_name,
            "dataset": {
                "name": dataset_name,
                "path_absolute": dataset_abs,
                "path_relative": ".",
                "hash": None,
            },
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
            "best_model": "train/weights/best.pt" if (run_dir / "train/weights/best.pt").exists() else None,
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = build_migrate_models_arg_parser().parse_args(argv)
    models_root = Path(args.models_root).expanduser().resolve()

    if not models_root.exists():
        raise SystemExit(f"[ERROR] Папка не найдена: {models_root}")

    run_dirs = infer_run_dirs(models_root)
    if not run_dirs:
        raise SystemExit("[ERROR] Не найдено ни одного run-каталога с train/results.csv")

    created = 0
    skipped = 0
    updated = 0

    for run_dir in run_dirs:
        meta_path = run_dir / "training_metadata.json"
        existed_before = meta_path.exists()
        if existed_before and not args.overwrite:
            skipped += 1
            print(f"[SKIP] Уже есть: {meta_path}")
            continue

        args_data = load_args_yaml(run_dir)
        metadata = build_metadata(run_dir, args_data)

        if args.dry_run:
            action = "UPDATE" if existed_before else "CREATE"
            print(f"[DRY-RUN] {action}: {meta_path}")
            continue

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        if existed_before and args.overwrite:
            updated += 1
            print(f"[OK] Обновлен: {meta_path}")
        else:
            created += 1
            print(f"[OK] Создан: {meta_path}")

    print(
        f"\nГотово. created={created}, updated={updated}, skipped={skipped}, total={len(run_dirs)}"
    )


if __name__ == "__main__":
    main()
