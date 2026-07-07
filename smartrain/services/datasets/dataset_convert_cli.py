from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import (
    print_numbered_options,
    prompt_choice,
    prompt_text,
    prompt_yes_no,
)
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.services.datasets.cvsdcldet_converter import (
    collect_cvsdcldet_class_names,
    is_cvsdcldet_dir,
    parse_rename_classes_args,
)
from smartrain.services.datasets.dataset_cli_common import load_dataset_catalog
from smartrain.services.datasets.dataset_convert_service import (
    TARGET_CVAT11,
    TARGET_CVAT11_ZIP,
    TARGET_YOLO,
    ConvertOptions,
    DatasetSource,
    detect_source_structure,
    list_available_targets,
    resolve_source,
    run_conversion,
    structure_display_name,
    target_display_name,
)
from smartrain.services.datasets.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.services.datasets.datasets_json_scan_core_service import detect_structure


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


def _default_output_dir(workspace_root: str, source: DatasetSource, target: str) -> Path:
    suffix = target.replace("_zip", "")
    if source.dataset_key:
        base = Path(workspace_root) / "datasets"
        name = next_dataset_name(str(base), f"{source.dataset_key}_{suffix}")
        return base / name
    if is_cvsdcldet_dir(source.path) or str(source.path).startswith(str(Path(workspace_root) / "raw_data")):
        base = Path(workspace_root) / "converted_raw_data"
        name = next_dataset_name(str(base), f"{source.name}_{suffix}")
        return base / name
    base = Path(workspace_root) / "converted_raw_data"
    name = next_dataset_name(str(base), f"{source.name}_{suffix}")
    return base / name


def _list_raw_data_sources(raw_data: str) -> list[tuple[str, str]]:
    root = Path(raw_data)
    if not root.is_dir():
        return []
    out: list[tuple[str, str]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        structure = detect_structure(str(entry))
        if structure == "unknown":
            continue
        out.append((str(entry), structure))
    return out


def _prompt_source(workspace_root: str) -> DatasetSource:
    layout = WorkspaceLayout(workspace_root)
    catalog = load_dataset_catalog(layout)
    dataset_names = sorted(catalog.keys())
    raw_entries = _list_raw_data_sources(layout.raw_data)

    options: list[str] = []
    option_map: dict[str, DatasetSource | str] = {}

    for name in dataset_names:
        key = f"[datasets] {name}"
        entry = catalog[name]
        structure = str(entry.get("structure") or "unknown")
        options.append(key)
        option_map[key] = DatasetSource(
            path=Path("."),
            structure=structure,
            name=name,
            dataset_key=name,
        )

    for path, structure in raw_entries:
        key = f"[raw_data] {Path(path).name}"
        options.append(key)
        option_map[key] = DatasetSource(
            path=Path(path),
            structure=structure,
            name=Path(path).name,
        )

    options.append("<enter directory path>")
    options.append("<enter CVAT zip path>")

    print("[INFO] Select dataset source.")
    if dataset_names:
        print_numbered_options("workspace datasets", dataset_names)
    if raw_entries:
        print(f"[INFO] raw_data sources: {', '.join(Path(p).name for p, _ in raw_entries)}")

    selected = prompt_choice("Source", options, default=options[0] if options else "<enter directory path>")

    if selected == "<enter directory path>":
        manual = prompt_text("Source directory", default="").strip()
        if not manual:
            raise SystemExit("Source directory is required.")
        path = Path(manual).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"Not a directory: {path}")
        structure = detect_structure(str(path))
        if structure == "unknown":
            raise SystemExit(f"Unsupported dataset structure: {path}")
        return DatasetSource(path=path, structure=structure, name=path.name)

    if selected == "<enter CVAT zip path>":
        manual = prompt_text("CVAT 1.1 zip path", default="").strip()
        if not manual:
            raise SystemExit("CVAT zip path is required.")
        zip_path = Path(manual).expanduser().resolve()
        if not zip_path.is_file():
            raise SystemExit(f"File not found: {zip_path}")
        structure = detect_source_structure(zip_path)
        if structure != "cvat11_zip":
            raise SystemExit(f"Not a CVAT 1.1 zip: {zip_path}")
        return DatasetSource(
            path=zip_path,
            structure=structure,
            name=zip_path.stem,
            source_zip=zip_path,
        )

    picked = option_map[selected]
    if isinstance(picked, DatasetSource) and picked.dataset_key:
        return resolve_source(workspace_root=workspace_root, dataset_key=picked.dataset_key)
    if isinstance(picked, DatasetSource):
        return picked
    raise SystemExit("Invalid source selection.")


def _prompt_target(source: DatasetSource) -> str:
    targets = list_available_targets(source.structure)
    if not targets:
        raise SystemExit(f"No conversion targets available for {source.display_structure}.")
    print(f"[INFO] Source format: {source.display_structure}")
    labels = [t.label for t in targets]
    ids = [t.target_id for t in targets]
    selected_label = prompt_choice("Convert to", labels, default=labels[0])
    idx = labels.index(selected_label)
    return ids[idx]


def _prompt_output_dir(workspace_root: str, source: DatasetSource, target: str) -> Path:
    default = _default_output_dir(workspace_root, source, target)
    if target == TARGET_CVAT11_ZIP:
        default = default.parent / f"{default.name}.cvat11.zip"
    rel_default = default
    try:
        rel_default = default.relative_to(Path(workspace_root))
    except ValueError:
        pass
    print("[INFO] Output path (empty = default).")
    raw = prompt_text("Output path", default=str(rel_default)).strip()
    if not raw:
        return default.resolve()
    out = Path(raw).expanduser()
    if not out.is_absolute():
        out = Path(workspace_root) / out
    return out.resolve()


def _prompt_class_rename(source_dir: Path) -> dict[str, str]:
    if not is_cvsdcldet_dir(source_dir):
        return {}
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


def build_dataset_convert_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Convert datasets between supported formats (empty call starts interactive mode)."
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR})",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset key from datasets_info.json",
    )
    p.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Source dataset directory (workspace or external path)",
    )
    p.add_argument(
        "--source-zip",
        type=str,
        default=None,
        help="Source CVAT for images 1.1 zip archive",
    )
    p.add_argument(
        "--to",
        type=str,
        default=None,
        choices=(TARGET_YOLO, TARGET_CVAT11, TARGET_CVAT11_ZIP),
        help="Target format: yolo, cvat11, cvat11_zip",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory or zip path (for cvat11_zip)",
    )
    p.add_argument("--task-name", type=str, default=None, help="Task name for CVAT export/meta.")
    p.add_argument(
        "--names",
        type=str,
        default=None,
        help="Comma-separated class names override for YOLO -> CVAT export.",
    )
    p.add_argument(
        "--rename-classes",
        nargs=2,
        metavar=("OLD", "NEW"),
        action="append",
        default=None,
        help="Rename a CvsDclDet class (repeatable).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing output.")
    p.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Directory for temporary files (default: current directory)",
    )
    zip_group = p.add_mutually_exclusive_group()
    zip_group.add_argument("--zip", action="store_true", help="Pack folder output into a zip archive.")
    zip_group.add_argument("--no-zip", action="store_true", help="Do not pack folder output into zip.")
    del_group = p.add_mutually_exclusive_group()
    del_group.add_argument(
        "--delete-after-zip",
        action="store_true",
        help="Delete output folder after creating zip (default when --zip).",
    )
    del_group.add_argument(
        "--no-delete-after-zip",
        action="store_true",
        help="Keep output folder after creating zip.",
    )
    return p


def _parse_names_csv(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _write_passport(
    *,
    source: DatasetSource,
    result_target: str,
    result_dir: Path | None,
    args: argparse.Namespace,
    workspace_root: str | None,
    class_rename: dict[str, str] | None,
    stats_before: dict | None = None,
    stats_after: dict | None = None,
    transformation_type: str,
) -> None:
    if result_dir is None or not result_dir.is_dir():
        return
    try:
        write_dataset_passport(
            output_dataset_dir=str(result_dir),
            command="dataset convert",
            source_datasets=[
                {
                    "name": source.name,
                    "path": str(source.path),
                    "dataset_hash": None,
                }
            ],
            parameters={**vars(args), "to": result_target, "rename_classes": class_rename},
            workspace_root=workspace_root or _workspace_root_if_inside(str(result_dir)),
            transformations=[{"type": transformation_type, "target": result_target}],
            random_seed=None,
            stats_before=stats_before or {},
            stats_after=stats_after or {},
        )
    except Exception as e:
        print(f"[WARNING] Failed to write dataset_passport.json: {e}")


def _transformation_type(source_structure: str, target: str) -> str:
    if source_structure == "cvsdcldet" and target == TARGET_CVAT11:
        return "cvsdcldet_to_cvat11"
    if source_structure in ("cvat11", "cvat11_zip") and target == TARGET_YOLO:
        return "cvat11_to_yolo"
    if target in (TARGET_CVAT11, TARGET_CVAT11_ZIP):
        return "yolo_to_cvat11"
    return f"{source_structure}_to_{target}"


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    parser = build_dataset_convert_arg_parser()
    args = parser.parse_args(argv)

    interactive = is_interactive_allowed(argv) and len(argv) == 0 and sys.stdin.isatty()
    interactive_used = False

    workspace_root: str | None = None
    try:
        workspace_root = resolve_workspace_root(args.workspace)
    except ValueError:
        if interactive:
            raise SystemExit(
                f"Workspace is not set: specify --workspace or environment variable {WORKSPACE_ENV_VAR} "
                "for interactive mode."
            )

    source: DatasetSource | None = None
    target: str | None = args.to
    output_path: Path | None = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    class_rename: dict[str, str] = {}
    create_zip: bool | None = True if args.zip else False if args.no_zip else None
    delete_after_zip: bool = not args.no_delete_after_zip

    if interactive:
        interactive_used = True
        source = _prompt_source(workspace_root)
        target = _prompt_target(source)
        output_path = _prompt_output_dir(workspace_root, source, target)
        if source.structure == "cvsdcldet" and not args.rename_classes:
            class_rename = _prompt_class_rename(source.path)
        if target != TARGET_CVAT11_ZIP:
            if create_zip is None:
                create_zip = prompt_yes_no("Save output to zip archive?", default=False)
            if create_zip:
                delete_after_zip = prompt_yes_no("Delete output folder after zip?", default=True)
            else:
                delete_after_zip = False
    else:
        try:
            source = resolve_source(
                workspace_root=workspace_root,
                dataset_key=args.dataset,
                source_dir=args.source_dir,
                source_zip=args.source_zip,
            )
        except (ValueError, KeyError, FileNotFoundError) as e:
            raise SystemExit(str(e)) from e

        if not target:
            raise SystemExit("--to is required in non-interactive mode.")
        if output_path is None:
            if workspace_root is None:
                raise SystemExit("--output-dir is required when workspace is not set.")
            output_path = _default_output_dir(workspace_root, source, target)
            if target == TARGET_CVAT11_ZIP:
                output_path = output_path.parent / f"{output_path.name}.cvat11.zip"

        if create_zip is None:
            create_zip = False

    assert source is not None and target is not None and output_path is not None

    if args.rename_classes:
        try:
            class_rename = parse_rename_classes_args(args.rename_classes)
        except ValueError as e:
            raise SystemExit(str(e)) from e

    names = _parse_names_csv(args.names)
    if not names and target in (TARGET_CVAT11, TARGET_CVAT11_ZIP) and source.structure not in (
        "cvsdcldet",
        "cvat11",
        "cvat11_zip",
    ):
        names = _load_names_from_data_yaml(source.path)

    tmp_base_dir = Path(args.tmp_dir).expanduser().resolve() if args.tmp_dir else Path.cwd()
    opts = ConvertOptions(
        task_name=args.task_name,
        names=names,
        class_rename=class_rename or None,
        force=bool(args.force),
        tmp_base_dir=tmp_base_dir,
        create_zip=bool(create_zip) and target != TARGET_CVAT11_ZIP,
        delete_after_zip=delete_after_zip,
        zip_path=None,
    )

    stats_before: dict = {}
    if source.structure == "cvsdcldet":
        stats_before = {"classes": collect_cvsdcldet_class_names(source.path)}

    try:
        result = run_conversion(source, target, output_path, opts=opts)
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        raise SystemExit(str(e)) from e

    transformation = _transformation_type(source.structure, target)
    passport_dir = result.output_dir
    if passport_dir is None and result.zip_path and not delete_after_zip:
        passport_dir = output_path if output_path.is_dir() else None

    _write_passport(
        source=source,
        result_target=target,
        result_dir=passport_dir,
        args=args,
        workspace_root=workspace_root,
        class_rename=class_rename or None,
        stats_before=stats_before,
        stats_after=result.info,
        transformation_type=transformation,
    )

    src_label = structure_display_name(source.structure)
    tgt_label = target_display_name(target)
    print(f"[OK] {src_label} -> {tgt_label}")
    if result.output_dir is not None:
        print(f"[OK] Output folder: {result.output_dir}")
    if result.zip_path is not None:
        print(f"[OK] Output zip: {result.zip_path}")
    for key in ("nc", "images_count", "labels_count", "boxes_count", "task_name"):
        if key in result.info and result.info[key] is not None:
            print(f"[OK] {key}={result.info[key]}")

    if interactive_used:
        replay = argparse.Namespace(**vars(args))
        if source.dataset_key:
            replay.dataset = source.dataset_key
            replay.source_dir = None
            replay.source_zip = None
        elif source.source_zip:
            replay.source_zip = str(source.source_zip)
            replay.dataset = None
            replay.source_dir = None
        else:
            replay.source_dir = str(source.path)
            replay.dataset = None
            replay.source_zip = None
        replay.to = target
        replay.output_dir = str(output_path)
        replay.rename_classes = [[k, v] for k, v in class_rename.items()] if class_rename else None
        replay.zip = bool(create_zip)
        if delete_after_zip:
            replay.delete_after_zip = True
            replay.no_delete_after_zip = False
        else:
            replay.no_delete_after_zip = True
        replay_cmd = build_non_interactive_command("dataset convert", parser, replay)
        print_replay_command("dataset convert", replay_cmd)


if __name__ == "__main__":
    main()
