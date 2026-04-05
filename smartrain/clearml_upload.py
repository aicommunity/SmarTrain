"""
Постфактум загрузка каталога прогона Ultralytics в ClearML.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from smartrain.cli_argparse import CliArgumentParser


def build_clearml_upload_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Загрузка каталога прогона в ClearML")
    p.add_argument(
        "run_dir",
        type=str,
        help="Каталог прогона (обычно .../runs/<dataset>/<timestamp_...>/ или с подкаталогом train)",
    )
    p.add_argument(
        "--project",
        type=str,
        default=None,
        help="Имя проекта ClearML (иначе CLEARML_PROJECT или smartrain)",
    )
    p.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Имя задачи (по умолчанию имя каталога прогона)",
    )
    p.add_argument(
        "--train-subdir",
        type=str,
        default="train",
        help="Подкаталог с args.yaml и weights (по умолчанию train)",
    )
    p.add_argument(
        "--no-images",
        action="store_true",
        help="Не загружать изображения из дерева прогона",
    )
    return p


def _find_train_dir(run_dir: Path, train_subdir: str) -> Path:
    run_dir = run_dir.resolve()
    t = run_dir / train_subdir
    if (t / "args.yaml").is_file() or (t / "weights" / "best.pt").is_file():
        return t
    if (run_dir / "args.yaml").is_file():
        return run_dir
    raise FileNotFoundError(
        f"Не найден args.yaml в {t} или {run_dir}. Укажите корень прогона или --train-subdir."
    )


def upload_run(
    run_dir: str,
    *,
    project: str | None = None,
    task_name: str | None = None,
    train_subdir: str = "train",
    upload_images: bool = True,
) -> None:
    try:
        from clearml import Task
    except ImportError as e:
        raise ImportError(
            "Установите clearml: pip install 'smartrain[clearml]' или pip install clearml"
        ) from e

    import yaml
    import pandas as pd
    from PIL import Image

    rd = Path(run_dir).expanduser().resolve()
    if not rd.is_dir():
        raise NotADirectoryError(run_dir)

    train_d = _find_train_dir(rd, train_subdir)
    proj = project or os.environ.get("CLEARML_PROJECT") or "smartrain"
    tname = task_name or rd.name

    task = Task.init(project_name=proj, task_name=tname, task_type=Task.TaskTypes.training)

    args_path = train_d / "args.yaml"
    if args_path.is_file():
        with open(args_path, encoding="utf-8") as f:
            hyperparameters = yaml.safe_load(f) or {}
        if isinstance(hyperparameters, dict):
            task.connect(hyperparameters)

    results_csv = train_d / "results.csv"
    if results_csv.is_file():
        results_data = pd.read_csv(results_csv)
        has_epoch = "epoch" in results_data.columns
        for i, (_, row) in enumerate(results_data.iterrows()):
            it = int(row["epoch"]) if has_epoch and pd.notna(row.get("epoch")) else i
            for column in results_data.columns:
                if column == "epoch":
                    continue
                try:
                    val = float(row[column])
                except (TypeError, ValueError):
                    continue
                task.get_logger().report_scalar(
                    title=column,
                    series="training",
                    iteration=it,
                    value=val,
                )

    if upload_images:
        exts = (".jpg", ".jpeg", ".png", ".webp")
        for root, _, files in os.walk(rd):
            for file in files:
                if not file.lower().endswith(exts):
                    continue
                file_path = Path(root) / file
                try:
                    image = Image.open(file_path).convert("RGB")
                except OSError:
                    continue
                rel = str(file_path.relative_to(rd))
                task.get_logger().report_image(rel, rel, iteration=0, image=image)

    best_pt = train_d / "weights" / "best.pt"
    if best_pt.is_file():
        task.upload_artifact("best_model", artifact_object=str(best_pt))

    task.close()
    print(f"[OK] ClearML: проект={proj!r}, задача={tname!r}")


def main(argv: list[str] | None = None) -> None:
    p = build_clearml_upload_arg_parser()
    args = p.parse_args(argv)
    upload_run(
        args.run_dir,
        project=args.project,
        task_name=args.task_name,
        train_subdir=args.train_subdir,
        upload_images=not args.no_images,
    )


if __name__ == "__main__":
    main()
