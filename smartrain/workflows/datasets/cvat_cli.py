from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.workflows.datasets.cvat11_converter import import_cvat11_zip_to_yolo, export_yolo_to_cvat11_zip
from smartrain.workflows.datasets.dataset_passport import write_dataset_passport


def _workspace_root_if_inside(output_dir: str) -> str | None:
    try:
        from smartrain.core.runtime.workspace_paths import resolve_workspace_root
    except Exception:
        return None
    try:
        ws = resolve_workspace_root(None)
    except ValueError:
        return None
    out_abs = os.path.abspath(os.path.expanduser(output_dir))
    ws_abs = os.path.abspath(ws)
    if out_abs == ws_abs or out_abs.startswith(ws_abs + os.sep):
        return ws_abs
    return None


def build_cvat_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="CVAT 1.1 conversion (Images + bbox): import/export.")
    p.add_argument(
        "command",
        choices=("import", "export"),
        help="Subcommand: import (CVAT zip -> YOLO) or export (YOLO -> CVAT zip)",
    )

    # Common-ish
    p.add_argument("--force", action="store_true", help="Overwrite output if available.")
    p.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Directory for temporary files (default: ./tmp relative to current directory)",
    )

    # import
    p.add_argument("--cvat-zip", type=str, default=None, help="Path to CVAT 1.1 zip export.")
    p.add_argument("--output-dir", type=str, default=None, help="Where to write the YOLO dataset (folder).")
    p.add_argument("--task-name", type=str, default=None, help="Task name (for export zip root and meta).")

    # export
    p.add_argument("--dataset-dir", type=str, default=None, help="The root of the YOLO dataset (folder with images/labels/data.yaml).")
    p.add_argument("--zip-path", type=str, default=None, help="Path to output zip (default: <dataset-dir>.cvat11.zip).")
    p.add_argument(
        "--names",
        type=str,
        default=None,
        help="List of class names separated by commas (if there is no data.yaml or needs to be overridden).",
    )

    return p


def _parse_names_csv(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    out = [x.strip() for x in raw.split(",") if x.strip()]
    return out


def _load_names_from_data_yaml(dataset_dir: Path) -> list[str]:
    # Lightweight: reuse existing datasets_json_former YAML loader to keep behavior consistent.
    from smartrain.datasets_json_former import find_yaml_file, load_yaml

    y = find_yaml_file(str(dataset_dir))
    if not y:
        return []
    data = load_yaml(y)
    if not isinstance(data, dict):
        return []
    names = data.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        try:
            return [str(v) for _k, v in sorted(names.items())]
        except Exception:
            return []
    return []


def main(argv: list[str] | None = None) -> None:
    args = build_cvat_arg_parser().parse_args(argv)
    cmd = args.command
    tmp_base_dir = Path(args.tmp_dir).expanduser().resolve() if args.tmp_dir else Path.cwd()

    if cmd == "import":
        if not args.cvat_zip or not args.output_dir:
            raise SystemExit("import requires --cvat-zip and --output-dir")
        info = import_cvat11_zip_to_yolo(
            cvat_zip_path=Path(args.cvat_zip),
            output_dir=Path(args.output_dir),
            task_name=args.task_name,
            force=bool(args.force),
            tmp_base_dir=tmp_base_dir,
        )
        try:
            write_dataset_passport(
                output_dataset_dir=str(Path(info["output_dir"])),
                command="cvat import",
                source_datasets=[
                    {
                        "name": Path(args.cvat_zip).name,
                        "path": str(Path(args.cvat_zip).expanduser().resolve()),
                        "dataset_hash": None,
                    }
                ],
                parameters=vars(args),
                workspace_root=_workspace_root_if_inside(str(info["output_dir"])),
                transformations=[
                    {
                        "type": "cvat11_to_yolo",
                        "task_name": args.task_name,
                    }
                ],
                random_seed=None,
                stats_before={},
                stats_after={
                    "nc": info.get("nc"),
                    "images_count": info.get("images_count"),
                    "labels_count": info.get("labels_count"),
                },
            )
        except Exception as e:
            print(f"[WARNING] Failed to write dataset_passport.json: {e}")
        print(f"[OK] CVAT import -> YOLO: {info['output_dir']}")
        print(f"[OK] classes={info['nc']} images={info['images_count']} labels={info['labels_count']}")
        return

    if cmd == "export":
        if not args.dataset_dir:
            raise SystemExit("export requires --dataset-dir")
        dataset_dir = Path(args.dataset_dir)
        zip_path = Path(args.zip_path) if args.zip_path else Path(str(dataset_dir) + ".cvat11.zip")

        names = _parse_names_csv(args.names)
        if not names:
            names = _load_names_from_data_yaml(dataset_dir)
        if not names:
            raise SystemExit("Could not determine names: specify --names or put data.yaml with the names field.")

        info = export_yolo_to_cvat11_zip(
            dataset_dir=dataset_dir,
            task_name=args.task_name,
            output_zip_path=zip_path,
            names=names,
            force=bool(args.force),
            tmp_base_dir=tmp_base_dir,
        )
        print(f"[OK] YOLO export -> CVAT zip: {info['zip_path']}")
        print(f"[OK] task_name={info['task_name']} images={info['images_count']}")
        return

    raise SystemExit(f"Unknown command: {cmd}")

