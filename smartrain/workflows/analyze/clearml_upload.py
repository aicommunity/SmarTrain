"""
After the fact, loading the Ultralytics run catalog into ClearML.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.core.runtime.run_artifacts import canonical_run_model_path, materialize_canonical_run_model


def build_clearml_upload_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Loading run directory in ClearML")
    p.add_argument(
        "run_dir",
        type=str,
        help="Run directory (usually .../runs/<dataset>/<timestamp_...>/ or with a train subdirectory)",
    )
    p.add_argument(
        "--project",
        type=str,
        default=None,
        help="ClearML project name (aka CLEARML_PROJECT or smartrain)",
    )
    p.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Task name (default run directory name)",
    )
    p.add_argument(
        "--train-subdir",
        type=str,
        default="train",
        help="Subdirectory with args.yaml and weights (default train)",
    )
    p.add_argument(
        "--no-images",
        action="store_true",
        help="Do not load images from the run tree",
    )
    return p


def _find_train_dir(run_dir: Path, train_subdir: str) -> Path:
    run_dir = run_dir.resolve()
    t = run_dir / train_subdir
    if (t / "args.yaml").is_file() or Path(canonical_run_model_path(str(run_dir), ".pt")).is_file():
        return t
    if (run_dir / "args.yaml").is_file():
        return run_dir
    raise FileNotFoundError(
        f"Args.yaml not found in {t} or {run_dir}. Please specify run root or --train-subdir."
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
            "Install clearml: pip install 'smartrain[clearml]' or pip install clearml"
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

    best_pt = Path(canonical_run_model_path(str(rd), ".pt"))
    if not best_pt.is_file():
        materialized = materialize_canonical_run_model(str(rd), ext=".pt", move=True, normalize_metadata=True)
        if materialized is not None:
            best_pt = Path(materialized)
    if best_pt.is_file():
        task.upload_artifact("best_model", artifact_object=str(best_pt))

    task.close()
    print(f"[OK] ClearML: project={proj!r}, task={tname!r}")


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
