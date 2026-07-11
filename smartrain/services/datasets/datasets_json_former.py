import os
import copy
import yaml
import json
import sys
import argparse
import hashlib
import zipfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.services.datasets.cvat11_converter import generate_temp_yolo_labels_from_cvat11_extracted
from smartrain.services.datasets.dataset_hash import calculate_dataset_hash
from smartrain.services.datasets.dataset_passport import write_dataset_passport
from smartrain.services.datasets.dataset_scan import (
    find_obj_data_file,
    find_obj_names_file,
    find_yaml_file,
    load_obj_data,
    load_obj_names,
)
from smartrain.core.runtime.workspace_paths import (
    WORKSPACE_ENV_VAR,
    WorkspaceLayout,
    resolve_workspace_root,
    resolve_or_extract_dataset_root,
    DATASETS_INFO_FILE,
    CLASS_NAMES_FILE,
)
from smartrain.providers.core.global_index import reconcile_stale_provider_paths
from smartrain.services.datasets.datasets_json_normalize_service import _normalize_path_for_data_path
from smartrain.services.datasets.datasets_json_report_io import (
    _merge_preserved_dataset_fields,
    _print_scan_report,
    _write_scan_summary,
)
from smartrain.services.datasets.datasets_json_scan_index_service import (
    _append_roots_from_datasets_list as _svc_append_roots_from_datasets_list,
    _compute_source_signature as _svc_compute_source_signature,
    _dir_has_content as _svc_dir_has_content,
    _extract_zip_for_scan as _svc_extract_zip_for_scan,
    _load_datasets_list_file as _svc_load_datasets_list_file,
    _run_scan_folder_roots as _svc_run_scan_folder_roots,
    _sorted_diff as _svc_sorted_diff,
    _unique_dataset_key as _svc_unique_dataset_key,
    _zip_extract_path as _svc_zip_extract_path,
)
from smartrain.services.datasets.dataset_class_cleanup import strip_unused_classes
from smartrain.services.datasets.datasets_json_convert_purge_service import (
    _confirm_purge_processed_raw as _svc_confirm_purge_processed_raw,
    _copy_source_to_training as _svc_copy_source_to_training,
    _dataset_content_hash as _svc_dataset_content_hash,
    _purge_raw_sources as _svc_purge_raw_sources,
)
from smartrain.services.datasets.datasets_json_cvat11_normalize_service import (
    _ensure_training_ready_after_copy as _svc_ensure_training_ready_after_copy,
)
from smartrain.services.datasets.datasets_json_scan_core_service import (
    _find_cvat_annotations_xml as _svc_find_cvat_annotations_xml,
    _cvat_has_images_dir_near_xml as _svc_cvat_has_images_dir_near_xml,
    _is_cvat11_images_xml as _svc_is_cvat11_images_xml,
    _load_cvat11_label_names as _svc_load_cvat11_label_names,
    _is_split_name as _svc_is_split_name,
    yolo_flat_image_label_buckets as _svc_yolo_flat_image_label_buckets,
    detect_structure as _svc_detect_structure,
    load_yaml as _svc_load_yaml,
    count_elements as _svc_count_elements,
    process_dataset as _svc_process_dataset,
)


sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_FILE = DATASETS_INFO_FILE
OUTPUT_CLASS_NAMES_FILE = CLASS_NAMES_FILE
DEFAULT_DATASETS_LIST_FILE = "datasets_list.txt"
SOURCE_SIGNATURE_KEY = "source_signature"
DATASET_HASH_KEY = "dataset_hash"
SOURCE_HASH_KEY = "source_hash"
SOURCE_REF_KEY = "source_ref"
MODIFIED_KEY = "modified"


def _find_cvat_annotations_xml(folder_path: str) -> Optional[str]:
    return _svc_find_cvat_annotations_xml(folder_path)


def _cvat_has_images_dir_near_xml(xml_path: str) -> bool:
    return _svc_cvat_has_images_dir_near_xml(xml_path)


def _is_cvat11_images_xml(xml_path: str) -> bool:
    return _svc_is_cvat11_images_xml(xml_path)


def _load_cvat11_label_names(xml_path: str) -> list[str]:
    return _svc_load_cvat11_label_names(xml_path)


IMAGE_EXTS_FLAT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _is_split_name(dir_name):
    return _svc_is_split_name(dir_name)


def yolo_flat_image_label_buckets(folder_path):
    return _svc_yolo_flat_image_label_buckets(folder_path)


def detect_structure(folder_path):
    return _svc_detect_structure(folder_path)


def load_yaml(file_path):
    return _svc_load_yaml(file_path)


def count_elements(folder_path, structure):
    return _svc_count_elements(folder_path, structure)



def process_dataset(folder_path, folder_name):
    return _svc_process_dataset(folder_path, folder_name)


def build_datasets_json_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        description="Processing datasets and creating JSON files with information about classes and structure"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Root workspace (otherwise {WORKSPACE_ENV_VAR}); scan reads raw_data/ and writes datasets/",
    )

    parser.add_argument(
        "--datasets-path",
        type=str,
        default=None,
        help="Mode without workspace: root of dataset directories for scanning",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Mode without workspace: directory for datasets_info.json and class_names.json",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=("scan", "refresh"),
        default="scan",
        help="scan: crawl raw_data subdirectories (or --datasets-path); "
        "refresh: Rescan only data_path from existing datasets_info.json",
    )
    parser.add_argument(
        "--datasets-list",
        type=str,
        default=None,
        help="TXT file with a list of paths to datasets (one per line): directories or .zip",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Explicitly specify the dataset (name from raw_data or path). The flag can be repeated.",
    )
    parser.add_argument(
        "--purge-processed-raw",
        action="store_true",
        help="Workspace/scan only: after confirmation, remove from raw_data the sources processed in the current run.",
    )
    parser.add_argument(
        "--repair-relative-paths",
        action="store_true",
        dest="repair_relative_paths",
        help="Workspace only: after a successful scan, rewrite absolute paths under the workspace in datasets/, runs/, tmp cache.",
    )
    parser.add_argument(
        "--repair-relative-paths-dry-run",
        action="store_true",
        dest="repair_relative_paths_dry_run",
        help="Like --repair-relative-paths but only print planned changes (no writes).",
    )
    parser.add_argument(
        "--repair-relative-paths-include-datasets-list",
        action="store_true",
        dest="repair_relative_paths_include_datasets_list",
        help="With path repair, also rewrite absolute paths inside raw_data/datasets_list.txt when they lie under the workspace.",
    )
    parser.add_argument(
        "--strip-unused-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan only: for newly added datasets, remove class names with zero label instances "
        "(remaps class ids in annotations). Default: on; use --no-strip-unused-classes to disable.",
    )
    parser.add_argument(
        "--auto-scan",
        action="store_true",
        help="Internal: quiet preflight scan invoked before other commands.",
    )

    return parser


def parse_args(argv=None):
    return build_datasets_json_arg_parser().parse_args(argv)


def _sorted_diff(old: Set[str], new: Set[str]) -> Tuple[List[str], List[str]]:
    return _svc_sorted_diff(old, new)


def _run_scan_folder_roots(folder_roots: list[tuple[str, str, dict]]) -> tuple[dict, dict]:
    return _svc_run_scan_folder_roots(folder_roots, process_dataset_cb=process_dataset)


def _dir_has_content(path: str) -> bool:
    return _svc_dir_has_content(path)


def _load_datasets_list_file(list_path: str) -> list[str]:
    return _svc_load_datasets_list_file(list_path)


def _unique_dataset_key(base_name: str, used_names: set[str]) -> str:
    return _svc_unique_dataset_key(base_name, used_names)


def _zip_extract_path(temp_root: str, zip_path: str) -> str:
    return _svc_zip_extract_path(temp_root, zip_path)


def _extract_zip_for_scan(zip_path: str, temp_root: str) -> str:
    return _svc_extract_zip_for_scan(zip_path, temp_root)


def _append_roots_from_datasets_list(
    *,
    list_path: str,
    folder_roots: list[tuple[str, str, dict]],
    used_names: set[str],
    use_workspace: bool,
    layout: Optional[WorkspaceLayout],
    output_dir: str,
) -> None:
    _svc_append_roots_from_datasets_list(
        list_path=list_path,
        folder_roots=folder_roots,
        used_names=used_names,
        use_workspace=use_workspace,
        layout=layout,
        output_dir=output_dir,
    )


def _compute_source_signature(path: str) -> str:
    return _svc_compute_source_signature(path)


def _copy_source_to_training(src_root: str, dst_root: str) -> None:
    _svc_copy_source_to_training(
        src_root,
        dst_root,
        ensure_training_ready_after_copy_cb=_ensure_training_ready_after_copy,
    )


def _ensure_training_ready_after_copy(dataset_root: str) -> bool:
    """
    Normalizes the copied dataset to a form suitable for training.
    Now the critical case: cvat11 (annotations.xml + images/) -> YOLO labels + data.yaml.
    """
    return _svc_ensure_training_ready_after_copy(
        dataset_root,
        detect_structure_cb=detect_structure,
        find_cvat_annotations_xml_cb=_find_cvat_annotations_xml,
        load_cvat11_label_names_cb=_load_cvat11_label_names,
        generate_temp_yolo_labels_cb=generate_temp_yolo_labels_from_cvat11_extracted,
    )


def _dataset_content_hash(path: str) -> Optional[str]:
    return _svc_dataset_content_hash(path)


def _ensure_prev_entry(previous_info: dict, key: str) -> dict:
    entry = previous_info.get(key)
    if not isinstance(entry, dict):
        entry = {}
        previous_info[key] = entry
    return entry


def _append_to_datasets_list(list_path: str, value: str) -> None:
    existing: set[str] = set()
    if os.path.isfile(list_path):
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    existing.add(s)
    if value in existing:
        return
    with open(list_path, "a", encoding="utf-8") as f:
        if existing:
            f.write("\n")
        f.write(value)
        f.write("\n")


def _confirm_purge_processed_raw(paths: list[str]) -> bool:
    return _svc_confirm_purge_processed_raw(paths)


def _purge_raw_sources(paths: list[str]) -> tuple[int, int]:
    return _svc_purge_raw_sources(paths)


def _ensure_scan_initial_passport(
    *,
    dataset_dir: str,
    dataset_name: str,
    entry: dict,
    workspace_root: str,
) -> None:
    passport_path = os.path.join(dataset_dir, "dataset_passport.json")
    if os.path.isfile(passport_path):
        return
    src_ref = entry.get(SOURCE_REF_KEY)
    src_payload: dict[str, Any] = {
        "name": str(src_ref) if src_ref else dataset_name,
        "path": str(src_ref) if src_ref else None,
        "dataset_hash": entry.get(SOURCE_HASH_KEY),
        "source_signature": entry.get(SOURCE_SIGNATURE_KEY),
    }
    # Clean None to keep passport compact and deterministic.
    src_payload = {k: v for k, v in src_payload.items() if v is not None}
    write_dataset_passport(
        output_dataset_dir=dataset_dir,
        command="scan",
        source_datasets=[src_payload],
        parameters={
            "workspace": workspace_root,
            "kind": "initial",
        },
        workspace_root=workspace_root,
        transformations=[
            {
                "type": "initial_sync_from_raw_data",
                "source_ref": src_ref,
                "structure": entry.get("structure"),
            }
        ],
        random_seed=None,
        stats_before={},
        stats_after={
            "classes": len(entry.get("names", []) or []),
            "dataset_hash": entry.get(DATASET_HASH_KEY),
        },
    )


def _append_explicit_dataset(
    *,
    raw: str,
    layout: WorkspaceLayout,
    output_file: str,
    folder_roots: list[tuple[str, str, dict]],
    used_names: set[str],
) -> None:
    token = (raw or "").strip()
    if not token:
        return
    candidate = os.path.abspath(os.path.expanduser(token))
    if os.path.exists(candidate):
        src_path = candidate
        base_name = os.path.splitext(os.path.basename(src_path))[0] if src_path.lower().endswith(".zip") else os.path.basename(src_path)
        list_value = src_path
    else:
        name = token[:-4] if token.lower().endswith(".zip") else token
        dir_candidate = os.path.join(layout.raw_data, name)
        zip_candidate = os.path.join(layout.raw_data, f"{name}.zip")
        if os.path.isdir(dir_candidate):
            src_path = dir_candidate
            base_name = name
            list_value = name
        elif os.path.isfile(zip_candidate):
            src_path = zip_candidate
            base_name = name
            list_value = f"{name}.zip"
        else:
            print(f"[WARNING] --dataset {raw!r}: path/name not found")
            return
    logical_name = _unique_dataset_key(base_name, used_names)
    sig = _compute_source_signature(src_path)
    prev_sig = None
    if os.path.isfile(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict) and isinstance(prev.get(logical_name), dict):
                prev_sig = prev[logical_name].get(SOURCE_SIGNATURE_KEY)
        except Exception:
            prev_sig = None
    dst = os.path.join(layout.datasets, logical_name)
    if src_path.lower().endswith(".zip"):
        try:
            extracted = resolve_or_extract_dataset_root(
                layout.root,
                logical_name,
                {"data_path": os.path.relpath(src_path, layout.root)},
                layout.raw_data,
            )
        except Exception as e:
            print(f"[WARNING] --dataset {raw!r}: failed to unpack zip ({e})")
            return
        source_for_copy = extracted
    else:
        source_for_copy = src_path
    if not (prev_sig == sig and _dir_has_content(dst)):
        _copy_source_to_training(source_for_copy, dst)
    else:
        print(f"[INFO] Skipping {logical_name!r}: source has not changed.")
    folder_roots.append(
        (
            logical_name,
            dst,
            {"data_path": os.path.relpath(dst, layout.root), SOURCE_SIGNATURE_KEY: sig},
        )
    )
    _append_to_datasets_list(os.path.join(layout.raw_data, DEFAULT_DATASETS_LIST_FILE), list_value)


def _auto_scan_can_skip(layout: WorkspaceLayout) -> bool:
    """True when raw_data sources match datasets_info signatures and materialized dirs exist."""
    info_path = layout.work_datasets_info_path()
    if not os.path.isdir(layout.raw_data):
        return True
    catalog: dict = {}
    if os.path.isfile(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                catalog = loaded
        except Exception:
            return False
    for src_name in os.listdir(layout.raw_data):
        src_path = os.path.join(layout.raw_data, src_name)
        if src_name == DEFAULT_DATASETS_LIST_FILE:
            continue
        if not (os.path.isdir(src_path) or src_name.lower().endswith(".zip")):
            continue
        logical_name = os.path.splitext(src_name)[0] if src_name.lower().endswith(".zip") else src_name
        sig = _compute_source_signature(src_path)
        entry = catalog.get(logical_name)
        if not isinstance(entry, dict):
            return False
        if entry.get(SOURCE_SIGNATURE_KEY) != sig:
            return False
        dst = os.path.join(layout.datasets, logical_name)
        if not _dir_has_content(dst):
            return False
    return True


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    if args.datasets_path:
        use_workspace = False
        layout = None
    else:
        try:
            root = resolve_workspace_root(args.workspace)
        except ValueError as e:
            print(f"[ERROR] {e}")
            return
        use_workspace = True
        layout = WorkspaceLayout(root)
        os.makedirs(layout.raw_data, exist_ok=True)
        os.makedirs(layout.datasets, exist_ok=True)
        if getattr(args, "auto_scan", False) and _auto_scan_can_skip(layout):
            return

    if use_workspace:
        # The source of truth for the index is the datasets directory (ready-made datasets).
        # raw_data is used only as a source of new/updated data.
        output_dir = layout.datasets
        output_file = os.path.join(output_dir, OUTPUT_FILE)
        output_class_names_file = os.path.join(output_dir, OUTPUT_CLASS_NAMES_FILE)
        datasets_dir = layout.datasets
        raw_source_dir = layout.raw_data
    else:
        datasets_dir = os.path.abspath(os.path.expanduser(args.datasets_path))
        if args.output_path:
            output_dir = os.path.abspath(os.path.expanduser(args.output_path))
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = datasets_dir
        output_file = os.path.join(output_dir, OUTPUT_FILE)
        output_class_names_file = os.path.join(output_dir, OUTPUT_CLASS_NAMES_FILE)

    datasets_info: dict = {}
    class_names: dict = {}
    processed_raw_sources: set[str] = set()
    stripped_dataset_names: set[str] = set()

    if args.mode == "refresh" and not use_workspace:
        print("[ERROR] --mode refresh is only supported without --datasets-path (via workspace).")
        return
    if args.purge_processed_raw and (not use_workspace or args.mode != "scan"):
        print("[ERROR] --purge-processed-raw is only available in workspace with --mode scan.")
        return

    if use_workspace and args.mode == "refresh":
        if not os.path.isfile(output_file):
            print(f"[ERROR] Refresh mode: {output_file} not found")
            return
        with open(output_file, "r", encoding="utf-8") as f:
            previous_full = json.load(f)
        if not isinstance(previous_full, dict):
            print(f"[ERROR] {output_file}: JSON object with datasets expected.")
            return
        folder_roots = []
        for name, prev_entry in previous_full.items():
            if not isinstance(prev_entry, dict):
                continue
            if "data_path" not in prev_entry:
                folder_roots.append((name, os.path.join(datasets_dir, name), {}))
            else:
                dp = prev_entry["data_path"]
                if not isinstance(dp, str):
                    print(f"[WARNING] Skipping {name!r}: data_path is not a string")
                    continue
                root_path = resolve_or_extract_dataset_root(layout.root, name, prev_entry, datasets_dir)
                folder_roots.append((name, root_path, {"data_path": dp}))
        datasets_info, class_names = _run_scan_folder_roots(folder_roots)
        previous_info = previous_full
    else:
        if not os.path.exists(datasets_dir):
            print(f"[ERROR] Folder '{datasets_dir}' not found.")
            return
        previous_info = {}
        if os.path.isfile(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    loaded_prev = json.load(f)
                if isinstance(loaded_prev, dict):
                    previous_info = loaded_prev
            except Exception as e:
                print(f"[WARNING] Failed to read existing {output_file}: {e}")
        previous_info_for_diff = copy.deepcopy(previous_info)
        folder_roots = []
        used_names: set[str] = set()

        def _sync_one_source(*, logical_name: str, source_for_copy: str, source_ref: str, source_signature: str) -> bool:
            dst_dir = os.path.join(layout.datasets, logical_name)
            prev_entry = previous_info.get(logical_name)
            normalized_on_skip = False
            if isinstance(prev_entry, dict) and bool(prev_entry.get(MODIFIED_KEY)):
                print(f"[WARNING] Skipping synchronization {logical_name!r}: modified=true.")
                return False

            source_hash = _dataset_content_hash(source_for_copy)
            if source_hash:
                for ds_name in os.listdir(layout.datasets):
                    ds_path = os.path.join(layout.datasets, ds_name)
                    if not os.path.isdir(ds_path):
                        continue
                    if ds_name == logical_name:
                        continue
                    ds_hash = None
                    if isinstance(previous_info.get(ds_name), dict):
                        ds_hash = previous_info[ds_name].get(DATASET_HASH_KEY)
                    if not ds_hash:
                        ds_hash = _dataset_content_hash(ds_path)
                    if ds_hash and ds_hash == source_hash:
                        # Recreate datasets/<logical_name> after manual deletion: the slot may still
                        # be listed in datasets_info.json while the directory is gone; do not treat
                        # another folder with the same content as a reason to skip materializing it.
                        if not _dir_has_content(dst_dir) and logical_name in previous_info:
                            break
                        print(
                            f"[WARNING] Skipping source {logical_name!r}: data matches datasets/{ds_name!r}."
                        )
                        return False

            prev_sig = prev_entry.get(SOURCE_SIGNATURE_KEY) if isinstance(prev_entry, dict) else None
            if prev_sig == source_signature and _dir_has_content(dst_dir):
                print(f"[INFO] Skipping {logical_name!r}: source has not changed.")
                # Even with skip we maintain compatibility: we configure the training-ready layout.
                normalized_on_skip = _ensure_training_ready_after_copy(dst_dir)
            else:
                _copy_source_to_training(source_for_copy, dst_dir)

            entry = _ensure_prev_entry(previous_info, logical_name)
            entry[SOURCE_SIGNATURE_KEY] = source_signature
            entry[SOURCE_REF_KEY] = source_ref
            entry[MODIFIED_KEY] = bool(entry.get(MODIFIED_KEY, False))
            if source_hash:
                entry[SOURCE_HASH_KEY] = source_hash
            # dataset_hash should reflect the actual contents of datasets/<name>
            # after possible normalization in training-ready layout.
            # Do not overwrite dataset_hash with normal skip (otherwise we will lose detection of manual edits).
            # We update it only after actual synchronization or normalization of cvat11.
            if not (prev_sig == source_signature and _dir_has_content(dst_dir)) or normalized_on_skip:
                current_dst_hash = _dataset_content_hash(dst_dir)
                if current_dst_hash:
                    entry[DATASET_HASH_KEY] = current_dst_hash
            return True

        if use_workspace:
            if os.path.isdir(raw_source_dir):
                for src_name in os.listdir(raw_source_dir):
                    src_path = os.path.join(raw_source_dir, src_name)
                    if not (os.path.isdir(src_path) or src_name.lower().endswith(".zip")):
                        continue
                    logical_name = os.path.splitext(src_name)[0] if src_name.lower().endswith(".zip") else src_name
                    if src_name.lower().endswith(".zip"):
                        sig = _compute_source_signature(src_path)
                        try:
                            extracted = resolve_or_extract_dataset_root(
                                layout.root,
                                logical_name,
                                {"data_path": os.path.relpath(src_path, layout.root)},
                                raw_source_dir,
                            )
                        except Exception as e:
                            print(f"[WARNING] Skipping archive {src_name!r} from raw_data: {e}")
                            continue
                        synced = _sync_one_source(
                            logical_name=logical_name,
                            source_for_copy=extracted,
                            source_ref=os.path.relpath(src_path, layout.root),
                            source_signature=sig,
                        )
                        if synced:
                            processed_raw_sources.add(os.path.abspath(src_path))
                    else:
                        synced = _sync_one_source(
                            logical_name=logical_name,
                            source_for_copy=src_path,
                            source_ref=os.path.relpath(src_path, layout.root),
                            source_signature=_compute_source_signature(src_path),
                        )
                        if synced:
                            processed_raw_sources.add(os.path.abspath(src_path))

        if use_workspace and args.dataset:
            used_names_for_explicit = {
                d for d in os.listdir(layout.datasets) if os.path.isdir(os.path.join(layout.datasets, d))
            }
            for raw_item in args.dataset:
                token = (raw_item or "").strip()
                if not token:
                    continue
                candidate = os.path.abspath(os.path.expanduser(token))
                if os.path.exists(candidate):
                    src_path = candidate
                    base_name = (
                        os.path.splitext(os.path.basename(src_path))[0]
                        if src_path.lower().endswith(".zip")
                        else os.path.basename(src_path)
                    )
                    list_value = src_path
                else:
                    name = token[:-4] if token.lower().endswith(".zip") else token
                    dir_candidate = os.path.join(layout.raw_data, name)
                    zip_candidate = os.path.join(layout.raw_data, f"{name}.zip")
                    if os.path.isdir(dir_candidate):
                        src_path = dir_candidate
                        base_name = name
                        list_value = name
                    elif os.path.isfile(zip_candidate):
                        src_path = zip_candidate
                        base_name = name
                        list_value = f"{name}.zip"
                    else:
                        print(f"[WARNING] --dataset {raw_item!r}: path/name not found")
                        continue
                logical_name = _unique_dataset_key(base_name, used_names_for_explicit)
                if src_path.lower().endswith(".zip"):
                    sig = _compute_source_signature(src_path)
                    try:
                        extracted = resolve_or_extract_dataset_root(
                            layout.root,
                            logical_name,
                            {"data_path": os.path.relpath(src_path, layout.root)},
                            layout.raw_data,
                        )
                    except Exception as e:
                        print(f"[WARNING] --dataset {raw_item!r}: failed to unpack zip ({e})")
                        continue
                    _sync_one_source(
                        logical_name=logical_name,
                        source_for_copy=extracted,
                        source_ref=src_path,
                        source_signature=sig,
                    )
                else:
                    _sync_one_source(
                        logical_name=logical_name,
                        source_for_copy=src_path,
                        source_ref=src_path,
                        source_signature=_compute_source_signature(src_path),
                    )
                _append_to_datasets_list(
                    os.path.join(layout.raw_data, DEFAULT_DATASETS_LIST_FILE), list_value
                )
        if args.datasets_list:
            list_path = os.path.abspath(os.path.expanduser(args.datasets_list))
            if not os.path.isfile(list_path):
                print(f"[ERROR] File not found --datasets-list: {list_path}")
                return
        elif use_workspace:
            auto_list = os.path.join(raw_source_dir, DEFAULT_DATASETS_LIST_FILE)
            list_path = auto_list if os.path.isfile(auto_list) else None
        else:
            list_path = None
        if list_path:
            # In workspace mode, datasets_list.txt describes external sources,
            # of which you need to prepare copies in datasets and include them in the index.
            if use_workspace:
                entries = _load_datasets_list_file(list_path)
                for src_path in entries:
                    if not os.path.exists(src_path):
                        print(f"[WARNING] Skipping from datasets-list: path not found: {src_path}")
                        continue
                    base_name = (
                        os.path.splitext(os.path.basename(src_path))[0]
                        if src_path.lower().endswith(".zip")
                        else os.path.basename(src_path)
                    )
                    logical_name = _unique_dataset_key(base_name, used_names)
                    dst_dir = os.path.join(layout.datasets, logical_name)
                    sig = _compute_source_signature(src_path)
                    prev_sig = None
                    if os.path.isfile(output_file):
                        try:
                            with open(output_file, "r", encoding="utf-8") as pf:
                                prev_j = json.load(pf)
                            if isinstance(prev_j, dict) and isinstance(prev_j.get(logical_name), dict):
                                prev_sig = prev_j[logical_name].get(SOURCE_SIGNATURE_KEY)
                        except Exception:
                            prev_sig = None
                    if src_path.lower().endswith(".zip"):
                        try:
                            extracted = _extract_zip_for_scan(
                                src_path,
                                os.path.join(output_dir, "tmp", "datasets_list_extract"),
                            )
                        except Exception as e:
                            print(f"[WARNING] Skipping archives from datasets-list {src_path!r}: {e}")
                            continue
                        source_for_copy = extracted
                    else:
                        source_for_copy = src_path
                    _sync_one_source(
                        logical_name=logical_name,
                        source_for_copy=source_for_copy,
                        source_ref=src_path,
                        source_signature=sig,
                    )
            else:
                _append_roots_from_datasets_list(
                    list_path=list_path,
                    folder_roots=folder_roots,
                    used_names=used_names,
                    use_workspace=use_workspace,
                    layout=layout,
                    output_dir=output_dir,
                )

        for folder_name in os.listdir(datasets_dir):
            folder_path = os.path.join(datasets_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            overrides: dict[str, Any] = {}
            if use_workspace:
                rel = os.path.relpath(folder_path, layout.root)
                overrides = {"data_path": rel}
            folder_roots.append((folder_name, folder_path, overrides))
            used_names.add(folder_name)

        if use_workspace and args.mode == "scan" and args.strip_unused_classes:
            class_names_map: dict[str, str] = {}
            if os.path.isfile(output_class_names_file):
                try:
                    with open(output_class_names_file, "r", encoding="utf-8") as f:
                        prev_cn = json.load(f)
                    if isinstance(prev_cn, dict):
                        class_names_map = {str(k): str(v) for k, v in prev_cn.items()}
                except Exception:
                    pass
            newly_added = {name for name, _, _ in folder_roots if name not in previous_info_for_diff}
            for name in sorted(newly_added):
                folder_path = next(fp for n, fp, _ in folder_roots if n == name)
                structure = detect_structure(folder_path)
                stats = strip_unused_classes(
                    folder_path,
                    structure,
                    {},
                    class_names_map=class_names_map,
                    ensure_cb=_ensure_training_ready_after_copy,
                )
                if stats.removed_class_names:
                    stripped_dataset_names.add(name)
                    print(
                        f"[INFO] strip-unused-classes: {name!r} removed "
                        f"{stats.removed_class_names} "
                        f"({stats.classes_before} -> {stats.classes_after} classes)"
                    )
                elif stats.skipped and stats.skip_reason:
                    print(f"[WARNING] strip-unused-classes: skipped {name!r}: {stats.skip_reason}")

        datasets_info, class_names = _run_scan_folder_roots(folder_roots)

    if use_workspace and args.mode == "scan" and args.purge_processed_raw:
        root_raw = os.path.abspath(layout.raw_data)
        list_path = os.path.abspath(os.path.join(root_raw, DEFAULT_DATASETS_LIST_FILE))
        purge_candidates = sorted(
            p
            for p in processed_raw_sources
            if p.startswith(root_raw + os.sep) and os.path.abspath(p) != list_path
        )
        if _confirm_purge_processed_raw(purge_candidates):
            removed, failed = _purge_raw_sources(purge_candidates)
            print(f"[INFO] Removed from raw_data: {removed}, errors: {failed}")
        else:
            print("[INFO] Removal of processed sources from raw_data cancelled.")

    for name in list(datasets_info.keys()):
        if name in previous_info and isinstance(previous_info[name], dict):
            datasets_info[name] = _merge_preserved_dataset_fields(
                datasets_info[name], previous_info[name]
            )
        if use_workspace and name in stripped_dataset_names:
            datasets_info[name][MODIFIED_KEY] = True
        if use_workspace:
            ds_path = os.path.join(layout.datasets, name)
            current_hash = _dataset_content_hash(ds_path)
            if current_hash:
                prev_hash = datasets_info[name].get(DATASET_HASH_KEY)
                was_modified = bool(datasets_info[name].get(MODIFIED_KEY, False))
                if prev_hash and prev_hash != current_hash and not was_modified:
                    print(
                        f"[WARNING] Dataset {name!r} was manually changed in datasets; "
                        "set modified=true, synchronization from raw_data is disabled."
                    )
                    datasets_info[name][MODIFIED_KEY] = True
                elif not was_modified:
                    datasets_info[name][MODIFIED_KEY] = False
                else:
                    datasets_info[name][MODIFIED_KEY] = True
                datasets_info[name][DATASET_HASH_KEY] = current_hash
            elif MODIFIED_KEY not in datasets_info[name]:
                datasets_info[name][MODIFIED_KEY] = bool(datasets_info[name].get(MODIFIED_KEY, False))

    if use_workspace:
        for name, entry in datasets_info.items():
            if not isinstance(entry, dict):
                continue
            try:
                dataset_root = resolve_or_extract_dataset_root(layout.root, name, entry, datasets_dir)
            except Exception:
                dataset_root = os.path.join(datasets_dir, name)
            if not os.path.isdir(dataset_root):
                continue
            try:
                _ensure_scan_initial_passport(
                    dataset_dir=dataset_root,
                    dataset_name=name,
                    entry=entry,
                    workspace_root=layout.root,
                )
            except Exception as e:
                print(f"[WARNING] Failed to write initial passport for {name!r}: {e}")

    old_ds_keys: Set[str] = set()
    if 'previous_info_for_diff' in locals() and isinstance(previous_info_for_diff, dict):
        old_ds_keys = {str(k) for k in previous_info_for_diff.keys()}
    elif isinstance(previous_info, dict):
        old_ds_keys = {str(k) for k in previous_info.keys()}
    new_ds_keys = set(datasets_info.keys())
    ds_added, ds_removed = _sorted_diff(old_ds_keys, new_ds_keys)

    previous_cn_keys: Set[str] = set()
    if os.path.isfile(output_class_names_file):
        try:
            with open(output_class_names_file, "r", encoding="utf-8") as f:
                prev_cn = json.load(f)
            if isinstance(prev_cn, dict):
                previous_cn_keys = set(prev_cn.keys())
        except Exception as e:
            print(f"[WARNING] Failed to read previous {OUTPUT_CLASS_NAMES_FILE} for summary: {e}")

    new_cn_keys = set(class_names.keys())
    cn_added, cn_removed = _sorted_diff(previous_cn_keys, new_cn_keys)

    try:
        from smartrain.core.runtime.workspace_coordination import catalog_write_lock, get_active_session
        from smartrain.services.datasets.dataset_cli_common import _write_json_atomic

        session = get_active_session()
        if use_workspace and layout is not None:
            with catalog_write_lock(layout, session=session):
                _write_json_atomic(output_file, datasets_info)
                _write_json_atomic(output_class_names_file, class_names)
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(datasets_info, f, ensure_ascii=False, indent=4)
            with open(output_class_names_file, "w", encoding="utf-8") as f:
                json.dump(class_names, f, ensure_ascii=False, indent=4)
        print(f"[OK] Information saved successfully in {output_file}")
        print(f"[OK] Information saved successfully in {output_class_names_file}")
    except Exception as e:
        print(f"[ERROR] Failed to write JSON: {e}")
        return

    summary_path = _write_scan_summary(
        output_dir=output_dir,
        datasets_final=datasets_info,
        class_names_final=class_names,
        datasets_added=ds_added,
        datasets_removed=ds_removed,
        class_names_added=cn_added,
        class_names_removed=cn_removed,
    )
    _print_scan_report(
        summary_path=summary_path,
        datasets_added=ds_added,
        datasets_removed=ds_removed,
        class_names_added=cn_added,
        class_names_removed=cn_removed,
        datasets_final_count=len(datasets_info),
        class_names_final_count=len(class_names),
        had_previous_datasets=bool(old_ds_keys),
        had_previous_class_names=bool(previous_cn_keys),
    )

    if use_workspace:
        reconcile_stats = reconcile_stale_provider_paths()
        if reconcile_stats.get("stale_marked", 0) > 0:
            print(
                "[INFO] Provider index sync: marked stale records="
                f"{reconcile_stats['stale_marked']}/{reconcile_stats['total']}"
            )

    if use_workspace and (
        getattr(args, "repair_relative_paths", False)
        or getattr(args, "repair_relative_paths_dry_run", False)
    ):
        from smartrain.core.runtime.workspace_path_repair import print_repair_report, repair_workspace_paths

        dry = bool(getattr(args, "repair_relative_paths_dry_run", False))
        rep = repair_workspace_paths(
            layout.root,
            dry_run=dry,
            include_datasets_list=bool(
                getattr(args, "repair_relative_paths_include_datasets_list", False)
            ),
        )
        print_repair_report(rep, dry_run=dry)


if __name__ == "__main__":
    main()
