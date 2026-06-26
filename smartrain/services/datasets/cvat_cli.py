from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import (
    prompt_choice,
    prompt_text,
    prompt_yes_no,
)
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.services.datasets.cvat11_converter import export_yolo_to_cvat11_zip, import_cvat11_zip_to_yolo
from smartrain.services.datasets.cvsdcldet_converter import (
    collect_cvsdcldet_class_names,
    convert_cvsdcldet_to_cvat11,
    is_cvsdcldet_dir,
    parse_rename_classes_args,
)
from smartrain.services.datasets.dataset_passport import next_dataset_name, write_dataset_passport


def _workspace_root_if_inside(output_dir: str) -> str | None:
    try:
        ws = resolve_workspace_root(None)
    except ValueError:
        return None
    out_abs = os.path.abspath(os.path.expanduser(output_dir))
    ws_abs = os.path.abspath(ws)
    if out_abs == ws_abs or out_abs.startswith(ws_abs + os.sep):
        return ws_abs
    return None


def _resolve_workspace(cli_workspace: str | None) -> str:
    return resolve_workspace_root(cli_workspace)


def _list_cvsdcldet_sources(raw_data: str) -> list[str]:
    root = Path(raw_data)
    if not root.is_dir():
        return []
    out: list[str] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and is_cvsdcldet_dir(entry):
            out.append(str(entry))
    return out


def _default_converted_output(workspace_root: str, source_dir: Path) -> Path:
    base = Path(workspace_root) / "converted_raw_data"
    name = next_dataset_name(str(base), source_dir.name)
    return base / name


def _prompt_source_dir(workspace_root: str) -> Path:
    layout = WorkspaceLayout(workspace_root)
    candidates = _list_cvsdcldet_sources(layout.raw_data)
    options = list(candidates)
    options.append("<enter path manually>")
    print("[INFO] Select CvsDclDet source directory (empty = choose from raw_data).")
    if not candidates:
        print(f"[INFO] No CvsDclDet folders found under {layout.raw_data}")
        manual = prompt_text("Source directory", default="").strip()
        if not manual:
            raise SystemExit("Source directory is required.")
        path = Path(manual).expanduser().resolve()
        if not is_cvsdcldet_dir(path):
            raise SystemExit(f"Not a CvsDclDet directory: {path}")
        return path
    selected = prompt_choice("Source directory", options, default=options[0])
    if selected == "<enter path manually>":
        manual = prompt_text("Source directory", default="").strip()
        if not manual:
            raise SystemExit("Source directory is required.")
        path = Path(manual).expanduser().resolve()
    else:
        path = Path(selected).expanduser().resolve()
    if not is_cvsdcldet_dir(path):
        raise SystemExit(f"Not a CvsDclDet directory: {path}")
    return path


def _prompt_output_dir(workspace_root: str, source_dir: Path) -> Path:
    default = _default_converted_output(workspace_root, source_dir)
    rel_default = default
    try:
        rel_default = default.relative_to(Path(workspace_root))
    except ValueError:
        pass
    print("[INFO] Output directory (empty = default under converted_raw_data/).")
    raw = prompt_text("Output directory", default=str(rel_default)).strip()
    if not raw:
        return default
    out = Path(raw).expanduser()
    if not out.is_absolute():
        out = Path(workspace_root) / out
    return out.resolve()


def _prompt_class_rename(source_dir: Path) -> dict[str, str]:
    classes = collect_cvsdcldet_class_names(source_dir)
    if not classes:
        return {}
    print(f"[INFO] Classes found: {', '.join(classes)}")
    if not prompt_yes_no("Rename classes?", default=False):
        return {}
    rename: dict[str, str] = {}
    for cls in classes:
        new_name = prompt_text(f"Rename class {cls!r}", default=cls).strip()
        if new_name and new_name != cls:
            rename[cls] = new_name
    return rename


def build_cvat_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="CVAT 1.1 conversion (Images + bbox): import/export/from-cvsdcldet."
    )
    p.add_argument(
        "command",
        choices=("import", "export", "from-cvsdcldet"),
        help="Subcommand: import (CVAT zip -> YOLO), export (YOLO -> CVAT zip), "
        "from-cvsdcldet (CvsDclDet -> CVAT 1.1)",
    )

    p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR})",
    )
    p.add_argument("--force", action="store_true", help="Overwrite output if available.")
    p.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Directory for temporary files (default: ./tmp relative to current directory)",
    )

    # import
    p.add_argument("--cvat-zip", type=str, default=None, help="Path to CVAT 1.1 zip export.")
    p.add_argument("--output-dir", type=str, default=None, help="Where to write the output dataset (folder).")
    p.add_argument("--task-name", type=str, default=None, help="Task name (for export zip root and meta).")

    # export
    p.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="The root of the YOLO dataset (folder with images/labels/data.yaml).",
    )
    p.add_argument("--zip-path", type=str, default=None, help="Path to output zip (default: <dataset-dir>.cvat11.zip).")
    p.add_argument(
        "--names",
        type=str,
        default=None,
        help="List of class names separated by commas (if there is no data.yaml or needs to be overridden).",
    )

    # from-cvsdcldet
    p.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="CvsDclDet source folder (paired image + json files).",
    )
    p.add_argument(
        "--rename-classes",
        nargs=2,
        metavar=("OLD", "NEW"),
        action="append",
        default=None,
        help="Rename a class: old_name new_name. Repeat for multiple renames.",
    )
    p.add_argument(
        "--zip",
        action="store_true",
        help="Also create a CVAT 1.1 zip next to the output directory.",
    )

    return p


def _parse_names_csv(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    out = [x.strip() for x in raw.split(",") if x.strip()]
    return out


def _load_names_from_data_yaml(dataset_dir: Path) -> list[str]:
    from smartrain.services.datasets.dataset_scan import find_yaml_file
    from smartrain.services.datasets.datasets_json_scan_core_service import load_yaml

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


def _run_from_cvsdcldet(args: argparse.Namespace, parser: argparse.ArgumentParser, argv: list[str] | None) -> None:
    interactive = is_interactive_allowed(argv)
    workspace_root: str | None = None
    try:
        workspace_root = _resolve_workspace(args.workspace)
    except ValueError:
        if interactive and args.command == "from-cvsdcldet" and not args.source_dir:
            raise SystemExit(
                f"Workspace is not set: specify --workspace or environment variable {WORKSPACE_ENV_VAR} "
                "for interactive source selection from raw_data."
            )

    source_path: Path | None = Path(args.source_dir).expanduser().resolve() if args.source_dir else None
    output_path: Path | None = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    class_rename: dict[str, str] = {}
    create_zip = bool(args.zip)

    if interactive and args.command == "from-cvsdcldet":
        if source_path is None:
            if workspace_root is None:
                manual = prompt_text("Source directory", default="").strip()
                if not manual:
                    raise SystemExit("Source directory is required.")
                source_path = Path(manual).expanduser().resolve()
            else:
                source_path = _prompt_source_dir(workspace_root)
        if output_path is None:
            if workspace_root is None:
                out_raw = prompt_text("Output directory", default="").strip()
                if not out_raw:
                    raise SystemExit("Output directory is required when workspace is not set.")
                output_path = Path(out_raw).expanduser().resolve()
            else:
                output_path = _prompt_output_dir(workspace_root, source_path)
        if not args.rename_classes:
            class_rename = _prompt_class_rename(source_path)
        if not args.zip:
            create_zip = prompt_yes_no("Also create CVAT zip?", default=False)

    if source_path is None:
        raise SystemExit("from-cvsdcldet requires --source-dir")
    if output_path is None:
        if workspace_root is not None:
            output_path = _default_converted_output(workspace_root, source_path)
        else:
            raise SystemExit("from-cvsdcldet requires --output-dir (or --workspace for default converted_raw_data/)")

    if args.rename_classes:
        try:
            class_rename = parse_rename_classes_args(args.rename_classes)
        except ValueError as e:
            raise SystemExit(str(e)) from e

    if not is_cvsdcldet_dir(source_path):
        raise SystemExit(f"Not a CvsDclDet directory: {source_path}")

    zip_path = Path(str(output_path) + ".cvat11.zip") if create_zip else None
    info = convert_cvsdcldet_to_cvat11(
        source_dir=source_path,
        output_dir=output_path,
        task_name=args.task_name,
        class_rename=class_rename or None,
        force=bool(args.force),
        create_zip=create_zip,
        zip_path=zip_path,
    )

    ws_for_passport = workspace_root or _workspace_root_if_inside(str(output_path))
    try:
        write_dataset_passport(
            output_dataset_dir=str(output_path),
            command="cvat from-cvsdcldet",
            source_datasets=[
                {
                    "name": source_path.name,
                    "path": str(source_path),
                    "dataset_hash": None,
                }
            ],
            parameters=vars(args),
            workspace_root=ws_for_passport,
            transformations=[{"type": "cvsdcldet_to_cvat11", "task_name": info.get("task_name")}],
            random_seed=None,
            stats_before={"classes": collect_cvsdcldet_class_names(source_path)},
            stats_after={
                "nc": info.get("nc"),
                "images_count": info.get("images_count"),
                "boxes_count": info.get("boxes_count"),
            },
        )
    except Exception as e:
        print(f"[WARNING] Failed to write dataset_passport.json: {e}")

    print(f"[OK] CvsDclDet -> CVAT 1.1: {info['output_dir']}")
    print(
        f"[OK] task_name={info['task_name']} classes={info['nc']} "
        f"images={info['images_count']} boxes={info['boxes_count']}"
    )
    if info.get("zip_path"):
        print(f"[OK] CVAT zip: {info['zip_path']}")

    if interactive and sys.stdin.isatty():
        replay_args = argparse.Namespace(**vars(args))
        replay_args.command = None
        replay_args.source_dir = str(source_path)
        replay_args.output_dir = str(output_path)
        replay_args.rename_classes = [[k, v] for k, v in class_rename.items()] if class_rename else None
        replay_args.zip = create_zip
        replay_cmd = build_non_interactive_command("cvat from-cvsdcldet", parser, replay_args)
        print_replay_command("after execution", replay_cmd)


def main(argv: list[str] | None = None) -> None:
    parser = build_cvat_arg_parser()
    args = parser.parse_args(argv)
    cmd = args.command
    tmp_base_dir = Path(args.tmp_dir).expanduser().resolve() if args.tmp_dir else Path.cwd()

    if cmd == "from-cvsdcldet":
        _run_from_cvsdcldet(args, parser, argv)
        return

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
