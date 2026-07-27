from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import (
    print_grouped_numbered_options,
    prompt_choice,
    prompt_text,
    prompt_yes_no,
)
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, resolve_workspace_root
from smartrain.services.datasets.cvsdcldet_converter import (
    collect_cvsdcldet_class_names,
    is_cvsdcldet_dir,
    parse_rename_classes_args,
)
from smartrain.services.datasets.dataset_convert_service import (
    TARGET_CVAT11,
    TARGET_YOLO,
    ConvertOptions,
    DatasetSource,
    list_available_targets,
    resolve_source,
    run_conversion,
    structure_display_name,
    target_display_name,
)
from smartrain.services.datasets.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.services.datasets.dataset_source_resolver import (
    MANUAL_SOURCE_OPTION,
    build_interactive_source_options,
    resolved_to_dataset_source,
    resolve_dataset_source,
    resolve_manual_source_token,
    replay_source_path,
)


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


def _is_raw_data_like_source(workspace_root: str, source: DatasetSource) -> bool:
    raw_root = str(Path(workspace_root) / "raw_data")
    for candidate in (source.path, source.source_archive):
        if candidate is not None and str(candidate).startswith(raw_root):
            return True
    return source.structure == "cvsdcldet"


def _default_output_dir(workspace_root: str, source: DatasetSource, target: str) -> Path:
    suffix = target.replace("_zip", "")
    if source.dataset_key:
        base = Path(workspace_root) / "datasets"
        name = next_dataset_name(str(base), f"{source.dataset_key}_{suffix}")
        return base / name
    if _is_raw_data_like_source(workspace_root, source):
        base = Path(workspace_root) / "converted_raw_data"
        name = next_dataset_name(str(base), f"{source.name}_{suffix}")
        return base / name
    base = Path(workspace_root) / "converted_raw_data"
    name = next_dataset_name(str(base), f"{source.name}_{suffix}")
    return base / name


def _prompt_source(workspace_root: str) -> DatasetSource:
    candidates, manual_option = build_interactive_source_options(workspace_root)
    if not candidates:
        raise SystemExit("No dataset sources found.")

    groups: list[tuple[str, list[str]]] = []
    workspace_opts = [c.label for c in candidates if c.group == "datasets"]
    raw_opts = [c.label for c in candidates if c.group == "raw_data"]
    external_opts = [c.label for c in candidates if c.group == "external"]
    manual_opts = [c.label for c in candidates if c.group == "manual"]
    if workspace_opts:
        groups.append(("Workspace datasets", workspace_opts))
    if raw_opts:
        groups.append(("raw_data", raw_opts))
    if external_opts:
        groups.append(("External (datasets_list.txt)", external_opts))
    if manual_opts:
        groups.append(("Manual", manual_opts))

    option_map = {c.label: c for c in candidates}
    flat_options = print_grouped_numbered_options(groups)
    print("[INFO] Select dataset source.")
    default = flat_options[0] if flat_options else manual_option
    selected = prompt_choice("Source", flat_options, default=default, show_options=False)

    if selected == MANUAL_SOURCE_OPTION:
        manual = prompt_text("Source path (directory or archive)", default="").strip()
        if not manual:
            raise SystemExit("Source path is required.")
        try:
            path = resolve_manual_source_token(workspace_root, manual)
        except (ValueError, FileNotFoundError) as e:
            raise SystemExit(str(e)) from e
        try:
            resolved = resolve_dataset_source(workspace_root, path)
        except (ValueError, FileNotFoundError) as e:
            raise SystemExit(str(e)) from e
        return resolved_to_dataset_source(resolved)

    picked = option_map[selected]
    if picked.dataset_key:
        return resolve_source(workspace_root=workspace_root, dataset_key=picked.dataset_key)
    try:
        resolved = resolve_dataset_source(workspace_root, picked.path, dataset_key=picked.dataset_key)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e)) from e
    return resolved_to_dataset_source(resolved)


def _prompt_target(source: DatasetSource) -> str:
    targets = list_available_targets(source.all_structures)
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
        help="Source dataset directory or archive (workspace or external path)",
    )
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source dataset directory or archive (.zip, .tar, .tar.gz, .tgz)",
    )
    p.add_argument(
        "--to",
        type=str,
        default=None,
        choices=(TARGET_YOLO, TARGET_CVAT11),
        help="Target format: yolo, cvat11",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
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
                    "path": replay_source_path(source) or str(source.path),
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
    if source_structure == "cvat11" and target == TARGET_YOLO:
        return "cvat11_to_yolo"
    if target == TARGET_CVAT11:
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
        if "cvsdcldet" in source.all_structures and not args.rename_classes:
            class_rename = _prompt_class_rename(source.path)
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
                source=args.source,
            )
        except (ValueError, KeyError, FileNotFoundError) as e:
            raise SystemExit(str(e)) from e

        if not target:
            raise SystemExit("--to is required in non-interactive mode.")
        if output_path is None:
            if workspace_root is None:
                raise SystemExit("--output-dir is required when workspace is not set.")
            output_path = _default_output_dir(workspace_root, source, target)

        if create_zip is None:
            create_zip = False

    if source is None or target is None or output_path is None:
        raise SystemExit("internal error: source/target/output_path must be resolved before conversion")

    if args.rename_classes:
        try:
            class_rename = parse_rename_classes_args(args.rename_classes)
        except ValueError as e:
            raise SystemExit(str(e)) from e

    names = _parse_names_csv(args.names)
    if not names and target == TARGET_CVAT11 and source.structure not in (
        "cvsdcldet",
        "cvat11",
    ):
        names = _load_names_from_data_yaml(source.path)

    tmp_base_dir = Path(args.tmp_dir).expanduser().resolve() if args.tmp_dir else Path.cwd()
    opts = ConvertOptions(
        task_name=args.task_name,
        names=names,
        class_rename=class_rename or None,
        force=bool(args.force),
        tmp_base_dir=tmp_base_dir,
        create_zip=bool(create_zip),
        delete_after_zip=delete_after_zip,
        zip_path=None,
    )

    stats_before: dict = {}
    if "cvsdcldet" in source.all_structures:
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
            replay.source = None
            replay.source_dir = None
        else:
            replay_path = replay_source_path(source)
            replay.source = replay_path
            replay.source_dir = None
            replay.dataset = None
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
