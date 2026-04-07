from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from smartrain.cli_argparse import CliArgumentParser
from smartrain.cvat11_converter import import_cvat11_zip_to_yolo, export_yolo_to_cvat11_zip
from smartrain.dataset_passport import write_dataset_passport


def build_cvat_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="CVAT 1.1 конвертация (Images + bbox): import/export.")
    p.add_argument(
        "command",
        choices=("import", "export"),
        help="Подкоманда: import (CVAT zip -> YOLO) или export (YOLO -> CVAT zip)",
    )

    # Common-ish
    p.add_argument("--force", action="store_true", help="Перезаписать выход при наличии.")
    p.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Каталог для временных файлов (по умолчанию: ./tmp относительно текущего каталога)",
    )

    # import
    p.add_argument("--cvat-zip", type=str, default=None, help="Путь к CVAT 1.1 zip export.")
    p.add_argument("--output-dir", type=str, default=None, help="Куда записать YOLO-датасет (папка).")
    p.add_argument("--task-name", type=str, default=None, help="Имя task (для export zip root и meta).")

    # export
    p.add_argument("--dataset-dir", type=str, default=None, help="Корень YOLO-датасета (папка с images/labels/data.yaml).")
    p.add_argument("--zip-path", type=str, default=None, help="Путь к выходному zip (по умолчанию: <dataset-dir>.cvat11.zip).")
    p.add_argument(
        "--names",
        type=str,
        default=None,
        help="Список имён классов через запятую (если нет data.yaml или нужно переопределить).",
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
            raise SystemExit("import требует --cvat-zip и --output-dir")
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
            print(f"[WARNING] Не удалось записать dataset_passport.json: {e}")
        print(f"[OK] CVAT import -> YOLO: {info['output_dir']}")
        print(f"[OK] classes={info['nc']} images={info['images_count']} labels={info['labels_count']}")
        return

    if cmd == "export":
        if not args.dataset_dir:
            raise SystemExit("export требует --dataset-dir")
        dataset_dir = Path(args.dataset_dir)
        zip_path = Path(args.zip_path) if args.zip_path else Path(str(dataset_dir) + ".cvat11.zip")

        names = _parse_names_csv(args.names)
        if not names:
            names = _load_names_from_data_yaml(dataset_dir)
        if not names:
            raise SystemExit("Не удалось определить names: задайте --names или положите data.yaml с полем names.")

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

