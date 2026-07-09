import os
import json
import shutil
import random
import argparse
import tempfile
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.services.datasets.cvat11_converter import YOLO_IMAGE_EXTS
from smartrain.services.datasets.dataset_access import (
    find_dataset_paths,
    iter_image_label_buckets,
    resolve_dataset_root_for_entry,
)
from smartrain.services.datasets.dataset_passport import write_dataset_passport
from smartrain.services.datasets.dataset_class_cleanup import strip_unused_classes
from smartrain.services.datasets.image_label_pairs import collect_label_image_pairs as _collect_label_image_pairs
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import (
    WORKSPACE_ENV_VAR,
    WorkspaceLayout,
    resolve_workspace_root,
    DATASETS_INFO_FILE,
    CLASS_NAMES_FILE,
)

from smartrain.services.datasets.dataset_split_core import (
    DEFAULT_RANDOM_SEED,
    TEST_PART,
    TRAIN_PART,
    VAL_PART,
    parse_split_ratio_arg,
    split_pairs_by_ratio,
)

# Default directory name suffix in workspace (prefix is date-time, see main).
FUSION_DEFAULT_DIR_SUFFIX = "merged"
parse_fusion_split_arg = parse_split_ratio_arg
random.seed(DEFAULT_RANDOM_SEED)


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def _unique_merge_stem(dataset_name, src_image_path, used_stems):
    """
    Name without extension for the image/label pair: “dataset_name-source_file”.
    used_stems — set of already used names in the target split (collisions: suffixes __2, __3, ...).
    """
    base = os.path.splitext(os.path.basename(src_image_path))[0]
    safe_ds = dataset_name.replace(os.sep, "_").replace("/", "_")
    safe_base = base.replace(os.sep, "_").replace("/", "_")
    stem = f"{safe_ds}-{safe_base}"
    if stem not in used_stems:
        used_stems.add(stem)
        return stem
    n = 2
    while True:
        cand = f"{stem}__{n}"
        if cand not in used_stems:
            used_stems.add(cand)
            return cand
        n += 1


def build_dataset_former_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        description="Combining and filtering datasets by selected classes"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Root workspace (otherwise {WORKSPACE_ENV_VAR}); work only with datasets/",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="The name of the output dataset (subdirectory datasets/) in workspace; "
        "if not specified - YYYY-MM-DD_HH-MM-SS-merged",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Name of input dataset to merge (can be repeated).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="CSV list of input datasets to merge (for example ds1,ds2).",
    )

    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Legacy: parent dataset directory (together with --target-path and --datasets-info-path)",
    )

    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Legacy: full path to the output dataset; in workspace - overrides datasets/<output-name>",
    )

    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Class names separated by commas; if not specified, combine all classes from all datasets in datasets_info.json (except the output one)",
    )
    parser.add_argument(
        "--exclude-classes",
        type=str,
        default=None,
        help="Class names separated by commas to exclude from the resulting class set (applied after --classes or auto-union)",
    )

    parser.add_argument(
        "--datasets-info-path",
        type=str,
        default=None,
        help="Legacy: directory with datasets_info.json; not needed in workspace (always datasets/)",
    )

    parser.add_argument(
        "--exclude-test",
        action="store_true",
        help="Exclude test data from selected datasets",
    )

    parser.add_argument(
        "--merge-classes",
        nargs=2,
        metavar=("SOURCES", "TARGET"),
        action="append",
        default=None,
        help="Merge classes: comma-separated name string and target name in --classes. Repeat flag for multiple groups.",
    )

    parser.add_argument(
        "--common-classes-only",
        action="store_true",
        help="Keep only the classes from the set (--classes or auto-union) present in each intersection group dataset; the rest are discarded with a warning",
    )
    parser.add_argument(
        "--include-partial-datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge datasets that contain at least one of the selected classes"
        "(enabled by default); use --no-include-partial-datasets,"
        "to require all selected classes to be present in each dataset",
    )
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help="After merging, remove image+label pairs in the output directory without a single valid YOLO line in .txt",
    )
    parser.add_argument(
        "--strip-unused-classes",
        action="store_true",
        help="After merging, remove output classes with zero label instances (remaps class ids in .txt)",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Directory for temporary files (default: <workspace>/tmp or <source-path>/tmp in legacy mode)",
    )

    parser.add_argument(
        "--fusion-split",
        type=str,
        default=None,
        help="Fusion only: shares train,val,test with random repartition of frames within each "
        "bucket of the original dataset (three numbers separated by commas, sum 1.0). Default "
        f"{TRAIN_PART},{VAL_PART},{TEST_PART}. Does not affect scan, train, roi, etc.",
    )

    return parser


def parse_args(argv=None):
    return build_dataset_former_arg_parser().parse_args(argv)


def _parse_selected_datasets(args) -> list[str]:
    out: list[str] = []
    if args.dataset:
        for item in args.dataset:
            name = str(item).strip()
            if name:
                out.append(name)
    if args.datasets:
        for part in str(args.datasets).split(","):
            name = part.strip()
            if name:
                out.append(name)
    uniq: list[str] = []
    seen: set[str] = set()
    for name in out:
        if name not in seen:
            seen.add(name)
            uniq.append(name)
    return uniq


def _prompt_dataset_selection(available: list[str]) -> list[str]:
    from smartrain.cli_entrypoints.support.cli_prompts import prompt_multi_choice_csv

    print("[INFO] --dataset/--datasets not specified: interactive selection of input datasets.")
    parsed = prompt_multi_choice_csv("Input datasets", available, default_values=[])
    uniq: list[str] = []
    seen: set[str] = set()
    for name in parsed:
        if name not in seen:
            seen.add(name)
            uniq.append(name)
    return uniq


def _prompt_yes_no(label: str, default: bool = False) -> bool:
    from smartrain.cli_entrypoints.support.cli_prompts import prompt_yes_no

    return prompt_yes_no(label, default=default)


def _validate_interactive_merge_rules(
    merge_rules: list[list[str]],
    *,
    class_names_map: dict,
    selected_classes: list[str],
    class_candidates: list[str],
) -> tuple[bool, str | None]:
    if not merge_rules:
        return True, None
    unknown_targets = []
    for _sources_csv, target in merge_rules:
        t = str(target).strip()
        if t and _normalize_name(t, class_names_map) not in {
            _normalize_name(cls, class_names_map) for cls in selected_classes
        }:
            unknown_targets.append(t)
    if unknown_targets:
        return (
            False,
            "Merge targets are missing in selected classes: "
            + ", ".join(sorted(set(unknown_targets)))
            + ".",
        )

    flat_sources: list[str] = []
    for sources_csv, _target in merge_rules:
        flat_sources.extend(_parse_csv_classes(sources_csv))
    if flat_sources:
        ok_sources, missing_sources = _validate_requested_classes(
            flat_sources,
            class_candidates,
            class_names_map,
        )
        if not ok_sources:
            return (
                False,
                "Merge sources contain unknown classes: "
                + ", ".join(missing_sources)
                + ".",
            )

    try:
        build_merge_config(merge_rules, class_names_map, selected_classes)
    except ValueError as exc:
        return False, str(exc)
    return True, None


def _derive_auto_classes_with_merge(
    class_candidates: list[str],
    merge_rules: list[list[str]] | None,
    class_names_map: dict,
) -> list[str]:
    if not merge_rules:
        return list(class_candidates)
    source_norm: set[str] = set()
    target_order: list[str] = []
    seen_targets: set[str] = set()
    for sources_csv, target in merge_rules:
        for src in _parse_csv_classes(sources_csv):
            source_norm.add(_normalize_name(src, class_names_map))
        tgt = str(target).strip()
        if not tgt:
            continue
        tgt_norm = _normalize_name(tgt, class_names_map)
        if tgt_norm not in seen_targets:
            seen_targets.add(tgt_norm)
            target_order.append(tgt)

    out: list[str] = []
    seen: set[str] = set()
    for cls in class_candidates:
        n = _normalize_name(cls, class_names_map)
        if n in source_norm:
            continue
        if n in seen:
            continue
        seen.add(n)
        out.append(cls)
    for tgt in target_order:
        n = _normalize_name(tgt, class_names_map)
        if n in seen:
            continue
        seen.add(n)
        out.append(tgt)
    return out


def _prompt_interactive_merge_rules(
    *,
    class_names_map: dict,
    selected_classes: list[str],
    class_candidates: list[str],
    initial_rules: list[list[str]] | None = None,
    auto_mode: bool = False,
) -> list[list[str]] | None:
    from smartrain.cli_entrypoints.support.cli_prompts import prompt_text

    rules: list[list[str]] = [list(x) for x in (initial_rules or [])]
    if not class_candidates:
        print("[WARN] Merge classes is skipped: no available classes.")
        return None
    print("[INFO] Configure --merge-classes (sources_csv -> target).")
    print("[INFO] Example: class_a,class_b -> class_ab")
    if auto_mode:
        print(
            "[INFO] Auto classes mode: classes not listed as merge sources are kept. "
            "Only source classes are replaced by their merge targets."
        )
    while True:
        raw = prompt_text("Merge rule (empty = finish)", default="").strip()
        if not raw:
            break
        if "->" not in raw:
            print("[WARN] Invalid format. Use: source1,source2 -> target")
            continue
        sources_raw, target_raw = raw.split("->", 1)
        target = target_raw.strip()
        sources = _parse_csv_classes(sources_raw)
        if not sources or not target:
            print("[WARN] Sources and target are required.")
            continue
        candidate = rules + [[",".join(sources), target]]
        selected_for_validation = (
            _derive_auto_classes_with_merge(class_candidates, candidate, class_names_map)
            if auto_mode
            else selected_classes
        )
        ok, err = _validate_interactive_merge_rules(
            candidate,
            class_names_map=class_names_map,
            selected_classes=selected_for_validation,
            class_candidates=class_candidates,
        )
        if not ok:
            print(f"[WARN] {err}")
            continue
        rules = candidate
        print(f"[INFO] Added merge rule: {','.join(sources)} -> {target}")
    return rules or None


def _prompt_interactive_options(
    args,
    *,
    default_output_name: str,
    class_candidates: list[str],
    class_names_map: dict,
) -> None:
    from smartrain.cli_entrypoints.support.cli_prompts import prompt_prefilled_text, prompt_text

    print("[INFO] Interactively configure merge parameters (Enter = default value).")
    if class_candidates:
        print(
            "[INFO] Available classes of selected datasets: "
            + ", ".join(class_candidates)
        )
    else:
        print("[WARN] No classes were found in the metadata for the selected datasets.")
    if sys.stdin.isatty():
        out_name = prompt_prefilled_text("Output dataset name", default_output_name).strip()
    else:
        out_name = prompt_text("Output dataset name", default=default_output_name).strip()
    args.output_name = out_name or default_output_name

    classes_raw = prompt_text(
        "Comma separated classes (empty = auto-union)",
        default=(args.classes or ""),
        choices=class_candidates,
    ).strip()
    args.classes = classes_raw or None
    exclude_classes_raw = prompt_text(
        "Comma separated classes to exclude (empty = exclude nothing)",
        default=(args.exclude_classes or ""),
        choices=class_candidates,
    ).strip()
    args.exclude_classes = exclude_classes_raw or None

    enable_merge = _prompt_yes_no(
        "Configure class merge rules (--merge-classes)",
        default=bool(args.merge_classes),
    )
    if enable_merge:
        auto_mode = not bool(_parse_csv_classes(args.classes))
        selected_classes = _parse_csv_classes(args.classes) or list(class_candidates)
        args.merge_classes = _prompt_interactive_merge_rules(
            class_names_map=class_names_map,
            selected_classes=selected_classes,
            class_candidates=class_candidates,
            initial_rules=args.merge_classes,
            auto_mode=auto_mode,
        )
        if auto_mode and args.merge_classes:
            auto_classes = _derive_auto_classes_with_merge(
                class_candidates,
                args.merge_classes,
                class_names_map,
            )
            args.classes = ",".join(auto_classes)
            print(
                "[INFO] Auto-selected output classes after merge "
                "(all non-source classes are preserved): "
                + ", ".join(auto_classes)
            )
    else:
        args.merge_classes = None

    split_default = args.fusion_split or f"{TRAIN_PART},{VAL_PART},{TEST_PART}"
    if sys.stdin.isatty():
        args.fusion_split = prompt_prefilled_text(
            "Fusion split train,val,test (summa=1.0)",
            split_default,
        ).strip()
    else:
        args.fusion_split = prompt_text(
            "Fusion split train,val,test (summa=1.0)",
            default=split_default,
        ).strip()

    args.include_partial_datasets = _prompt_yes_no(
        "Include partial datasets (--include-partial-datasets)",
        default=bool(args.include_partial_datasets),
    )
    args.common_classes_only = _prompt_yes_no(
        "Keep only common classes (--common-classes-only)",
        default=bool(args.common_classes_only),
    )
    args.exclude_test = _prompt_yes_no(
        "Exclude test parts of sources (--exclude-test)",
        default=bool(args.exclude_test),
    )
    args.drop_empty_images = _prompt_yes_no(
        "Delete pairs without valid objects (--drop-empty-images)",
        default=bool(args.drop_empty_images),
    )

    tmp_default = args.tmp_dir or ""
    tmp_value = prompt_text("Tmp directory (empty = default)", default=tmp_default).strip()
    args.tmp_dir = tmp_value or None


def _validate_requested_classes(
    selected_classes: list[str],
    class_candidates: list[str],
    class_names_map: dict,
) -> tuple[bool, list[str]]:
    """
    Checking that custom classes are available among the selected datasets
    taking into account the normalization of class_names.
    """
    if not selected_classes:
        return True, []
    available_norm = {_normalize_name(c, class_names_map) for c in class_candidates}
    missing = [
        cls
        for cls in selected_classes
        if _normalize_name(cls, class_names_map) not in available_norm
    ]
    return (len(missing) == 0), missing


def _parse_csv_classes(raw: str | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in str(raw).split(","):
        name = part.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _normalize_name(name, class_names_map):
    return class_names_map.get(name, name)


def dataset_normalized_keys(info, class_names_map):
    if "classes" not in info:
        return set()
    return {_normalize_name(k, class_names_map) for k in info["classes"].keys()}


def all_classes_union_from_datasets(datasets_info, output_dataset_name, class_names_map):
    """
    Combining normalized class names across all datasets from datasets_info,
    except for the day off. The order is lexicographic based on normalized names.
    """
    normalized = set()
    for name, info in datasets_info.items():
        if name == output_dataset_name:
            continue
        if "classes" not in info:
            continue
        for k in info["classes"].keys():
            normalized.add(_normalize_name(k, class_names_map))
    return sorted(normalized)


def request_normalized_tokens(selected_classes, merge_args, class_names_map):
    """Normalized names from --classes and from all --merge-classes."""
    tokens = {_normalize_name(c, class_names_map) for c in selected_classes}
    if merge_args:
        for sources_csv, target in merge_args:
            tokens.add(_normalize_name(target.strip(), class_names_map))
            for part in sources_csv.split(","):
                p = part.strip()
                if p:
                    tokens.add(_normalize_name(p, class_names_map))
    return tokens


def candidate_datasets_for_common_mode(datasets_info, output_dataset_name, request_tokens, class_names_map):
    """Datasets (except output) that have at least one class from the request."""
    out = []
    for name, info in datasets_info.items():
        if name == output_dataset_name:
            continue
        if dataset_normalized_keys(info, class_names_map) & request_tokens:
            out.append((name, info))
    return out


def reduce_selected_to_common_in_candidates(
    selected_classes, class_names_map, candidates, merge_targets_to_sources
):
    """
    Leaves the classes from selected_classes in order that satisfy
    dataset_matches_selection(..., [cls], ...) for each dataset from candidates.
    """
    effective = []
    for out in selected_classes:
        if all(
            dataset_matches_selection(info, class_names_map, [out], merge_targets_to_sources)
            for _, info in candidates
        ):
            effective.append(out)
    return effective


def _canonical_class_label(name_n, selected_classes, class_names_map):
    """The name from --classes` that matches name_n after normalization."""
    for sc in selected_classes:
        if _normalize_name(sc, class_names_map) == name_n:
            return sc
    return None


def build_merge_config(merge_args, class_names_map, selected_classes):
    """
    merge_args: list of [sources_csv, target] pairs or None.
    Returns (normalized_to_output_name, merge_targets_to_sources).
    The merge_targets_to_sources keys are the same as in the --classes list.
    """
    if not merge_args:
        return None, {}

    merge_targets_to_sources = {}
    used_sources = set()

    for sources_csv, target in merge_args:
        target_n = _normalize_name(target.strip(), class_names_map)
        canonical_target = _canonical_class_label(target_n, selected_classes, class_names_map)
        if canonical_target is None:
            raise ValueError(
                f"Merge target class {target.strip()!r} not found in --classes: {selected_classes}"
            )
        sources = [_normalize_name(s.strip(), class_names_map) for s in sources_csv.split(",") if s.strip()]
        if not sources:
            raise ValueError(f"Empty list of source classes for target {target!r}")
        if canonical_target in merge_targets_to_sources:
            raise ValueError(f"Merge target {canonical_target!r} specified twice")
        merge_targets_to_sources[canonical_target] = set(sources)
        for s in sources:
            if s in used_sources:
                raise ValueError(f"Source class {s!r} participates in more than one --merge-classes group")
            used_sources.add(s)

    normalized_to_output_name = {}
    for canonical_target, src_set in merge_targets_to_sources.items():
        for s in src_set:
            normalized_to_output_name[s] = canonical_target

    for out in selected_classes:
        out_n = _normalize_name(out, class_names_map)
        if out in merge_targets_to_sources:
            continue
        if out_n in used_sources:
            raise ValueError(
                f"Class {out!r} as a merge source only; remove from --classes or set a separate target"
            )
        if out_n in normalized_to_output_name:
            raise ValueError(f"Name conflict for {out!r}")
        normalized_to_output_name[out_n] = out

    return normalized_to_output_name, merge_targets_to_sources


def dataset_matches_selection(
    info,
    class_names_map,
    selected_classes,
    merge_targets_to_sources,
    *,
    require_all_classes: bool = True,
):
    """
    require_all_classes=True (default): the dataset must contain all selected classes
    (for the --merge-classes group - at least one source from each group).
    require_all_classes=False: intersection with any of the selected classes is sufficient.
    """
    if "classes" not in info:
        return False
    normalized_in_ds = {_normalize_name(k, class_names_map) for k in info["classes"].keys()}

    if require_all_classes:
        for out in selected_classes:
            out_n = _normalize_name(out, class_names_map)
            sources = merge_targets_to_sources.get(out)
            if sources:
                if not (normalized_in_ds & sources):
                    return False
            else:
                if out_n not in normalized_in_ds:
                    return False
        return True

    for out in selected_classes:
        out_n = _normalize_name(out, class_names_map)
        sources = merge_targets_to_sources.get(out)
        if sources:
            if normalized_in_ds & sources:
                return True
        else:
            if out_n in normalized_in_ds:
                return True
    return False


def _label_file_has_valid_yolo_annotation(path: str) -> bool:
    """Does the file have at least one line with an integer class_id (as in filter_label_file)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    int(parts[0])
                except ValueError:
                    continue
                return True
    except OSError:
        return False
    return False


def prune_output_empty_label_pairs(target_dir: str) -> int:
    """
    Removes label+image pairs in train/valid/test, where .txt is empty or without valid annotations.
    Returns the number of deleted label files.
    """
    removed = 0
    for split in ("train", "valid", "test"):
        labels_dir = os.path.join(target_dir, split, "labels")
        images_dir = os.path.join(target_dir, split, "images")
        if not os.path.isdir(labels_dir):
            continue
        for name in os.listdir(labels_dir):
            if not name.endswith(".txt"):
                continue
            lp = os.path.join(labels_dir, name)
            if _label_file_has_valid_yolo_annotation(lp):
                continue
            stem = os.path.splitext(name)[0]
            try:
                os.remove(lp)
            except OSError:
                pass
            if os.path.isdir(images_dir):
                for fn in os.listdir(images_dir):
                    if os.path.splitext(fn)[0] == stem:
                        ip = os.path.join(images_dir, fn)
                        try:
                            os.remove(ip)
                        except OSError:
                            pass
            removed += 1
    return removed


def filter_label_file(
    src_label_path,
    dst_label_path,
    class_map,
    class_names_map,
    selected_classes,
    normalized_to_output_name=None,
):
    id_to_normalized = {}
    for name, idx in class_map.items():
        normalized = class_names_map.get(name, name)
        id_to_normalized[idx] = normalized

    new_id_map = {cls: i for i, cls in enumerate(selected_classes)}

    filtered_lines = []

    with open(src_label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        normalized_name = id_to_normalized.get(class_id)
        if normalized_name is None:
            continue

        if normalized_to_output_name is not None:
            output_name = normalized_to_output_name.get(normalized_name)
            if output_name is None:
                continue
        else:
            output_name = normalized_name
            if output_name not in selected_classes:
                continue

        if output_name not in new_id_map:
            continue
        new_id = new_id_map[output_name]
        parts[0] = str(new_id)
        filtered_lines.append(" ".join(parts) + "\n")

    if filtered_lines:
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.writelines(filtered_lines)
        return True
    return False


def _image_content_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _parse_filtered_label_objects(
    src_label_path: str,
    class_map: dict,
    class_names_map: dict,
    selected_classes: list[str],
    normalized_to_output_name: dict | None = None,
) -> list[tuple[int, tuple[float, ...]]]:
    id_to_normalized = {}
    for name, idx in class_map.items():
        normalized = class_names_map.get(name, name)
        id_to_normalized[idx] = normalized
    new_id_map = {cls: i for i, cls in enumerate(selected_classes)}
    out: list[tuple[int, tuple[float, ...]]] = []
    try:
        with open(src_label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return out
    for raw in lines:
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        normalized_name = id_to_normalized.get(class_id)
        if normalized_name is None:
            continue
        if normalized_to_output_name is not None:
            output_name = normalized_to_output_name.get(normalized_name)
            if output_name is None:
                continue
        else:
            output_name = normalized_name
            if output_name not in selected_classes:
                continue
        new_id = new_id_map.get(output_name)
        if new_id is None:
            continue
        try:
            coords = tuple(round(float(x), 8) for x in parts[1:])
        except ValueError:
            continue
        out.append((new_id, coords))
    return out


def _canonical_label_signature(objs: list[tuple[int, tuple[float, ...]]]) -> tuple:
    return tuple(sorted(objs))


def _bbox_iou_xywh(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ax, ay, aw, ah = a[0], a[1], a[2], a[3]
    bx, by, bw, bh = b[0], b[1], b[2], b[3]
    ax1, ay1, ax2, ay2 = ax - aw / 2.0, ay - ah / 2.0, ax + aw / 2.0, ay + ah / 2.0
    bx1, by1, bx2, by2 = bx - bw / 2.0, by - bh / 2.0, bx + bw / 2.0, by + bh / 2.0
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = area_a + area_b - inter
    if den <= 0.0:
        return 0.0
    return inter / den


def _union_label_objects(
    existing: list[tuple[int, tuple[float, ...]]],
    incoming: list[tuple[int, tuple[float, ...]]],
    *,
    iou_threshold: float = 0.95,
) -> tuple[list[tuple[int, tuple[float, ...]]], int]:
    merged = list(existing)
    seen_exact = set(merged)
    removed_by_dedup = 0
    for cls_id, coords in incoming:
        item = (cls_id, coords)
        if item in seen_exact:
            removed_by_dedup += 1
            continue
        is_dup = False
        if len(coords) >= 4:
            for e_cls, e_coords in merged:
                if e_cls != cls_id or len(e_coords) < 4:
                    continue
                if _bbox_iou_xywh(coords, e_coords) >= iou_threshold:
                    is_dup = True
                    break
        if is_dup:
            removed_by_dedup += 1
            continue
        merged.append(item)
        seen_exact.add(item)
    return merged, removed_by_dedup


def _write_label_objects(path: str, objs: list[tuple[int, tuple[float, ...]]]) -> bool:
    if not objs:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return True
    lines = []
    for cls_id, coords in sorted(objs):
        if coords:
            vals = " ".join(f"{v:.8f}" for v in coords)
            lines.append(f"{cls_id} {vals}\n")
        else:
            lines.append(f"{cls_id}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return True


def _update_datasets_sidecar(
    layout: WorkspaceLayout,
    output_key: str,
    selected_classes: list,
    target_dir: str,
) -> None:
    os.makedirs(layout.datasets, exist_ok=True)
    rel = os.path.relpath(os.path.abspath(target_dir), layout.root)
    entry = {
        "classes": {name: idx for idx, name in enumerate(selected_classes)},
        "structure": "split",
        "elements_count": None,
        "data_path": rel,
    }
    info_path = layout.work_datasets_info_path()
    previous: dict = {}
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            previous = json.load(f)
        if not isinstance(previous, dict):
            previous = {}
    previous[output_key] = entry
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(previous, f, ensure_ascii=False, indent=4)

    cn_path = layout.work_class_names_path()
    class_names_out: dict = {}
    if os.path.isfile(cn_path):
        with open(cn_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            class_names_out = dict(loaded)
    for c in selected_classes:
        class_names_out[c] = c
    with open(cn_path, "w", encoding="utf-8") as f:
        json.dump(class_names_out, f, ensure_ascii=False, indent=4)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = build_dataset_former_arg_parser()
    args = parser.parse_args(argv)
    interactive_allowed = is_interactive_allowed(argv)
    preselected_dataset_names = _parse_selected_datasets(args)
    if not preselected_dataset_names and not interactive_allowed:
        print("[ERROR] Incomplete arguments: specify --dataset/--datasets.")
        return

    legacy = (
        args.source_path is not None
        and args.target_path is not None
        and args.datasets_info_path is not None
    )
    layout: WorkspaceLayout | None = None
    workspace_root: str | None = None

    if legacy:
        source_dir = os.path.abspath(os.path.expanduser(args.source_path))
        target_dir = os.path.abspath(os.path.expanduser(args.target_path))
        info_dir = os.path.abspath(os.path.expanduser(args.datasets_info_path))
    else:
        try:
            workspace_root = resolve_workspace_root(args.workspace)
        except ValueError as e:
            print(f"[ERROR] {e}")
            print(
                "[ERROR] Either specify workspace or all three flags: "
                "--source-path, --target-path, --datasets-info-path."
            )
            return
        layout = WorkspaceLayout(workspace_root)
        os.makedirs(layout.datasets, exist_ok=True)
        info_dir = layout.datasets
        source_dir = layout.datasets
        if args.target_path:
            target_dir = os.path.abspath(os.path.expanduser(args.target_path))
        else:
            raw_out = (args.output_name or "").strip()
            if raw_out:
                workspace_out = raw_out
            else:
                workspace_out = (
                    f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{FUSION_DEFAULT_DIR_SUFFIX}"
                )
            target_dir = os.path.join(layout.datasets, workspace_out)

    json_file = os.path.join(info_dir, DATASETS_INFO_FILE)
    class_names_file = os.path.join(info_dir, CLASS_NAMES_FILE)
    missing_metadata = [p for p in (json_file, class_names_file) if not os.path.isfile(p)]
    if missing_metadata:
        run_mode = "legacy flags mode" if legacy else "workspace mode"
        print("[ERROR] Fusion metadata files were not found.")
        print(f"[ERROR] Mode: {run_mode}")
        if workspace_root:
            print(f"[ERROR] Workspace root: {workspace_root}")
        print(f"[ERROR] Metadata directory: {info_dir}")
        print("[ERROR] Missing files:")
        for path in missing_metadata:
            print(f"[ERROR] - {path}")
        print(
            "[ERROR] Use --workspace /data/MarsSmarTrain or prepare metadata with "
            "'smartrain scan'/'smartrain deploy'."
        )
        return

    with open(json_file, "r", encoding="utf-8") as f:
        datasets_info = json.load(f)

    with open(class_names_file, "r", encoding="utf-8") as f:
        class_names_map = json.load(f)

    output_dataset_name = os.path.basename(target_dir)
    selected_dataset_names = preselected_dataset_names
    interactive_mode = not selected_dataset_names and interactive_allowed
    available_dataset_names = sorted(
        [
            n
            for n in datasets_info.keys()
            if n != output_dataset_name and isinstance(datasets_info.get(n), dict)
        ]
    )

    if not selected_dataset_names:
        if not interactive_allowed:
            print("[ERROR] Incomplete arguments: specify --dataset/--datasets.")
            return
        if not sys.stdin.isatty():
            print(
                "[ERROR] Interactive fusion mode requires a terminal (TTY). "
                "Pass --dataset/--datasets or run command without arguments in TTY."
            )
            return
        try:
            selected_dataset_names = _prompt_dataset_selection(available_dataset_names)
        except Exception as e:
            print(f"[ERROR] Failed to start interactive selection of datasets: {e}")
            return
        # Save the interactive selection in args so that the replay command
        # reflected an explicit list of input datasets.
        args.dataset = list(selected_dataset_names)
        args.datasets = None

    if not selected_dataset_names:
        print("[ERROR] No dataset has been selected for merging.")
        return

    unknown = [n for n in selected_dataset_names if n not in datasets_info]
    if unknown:
        known = ", ".join(available_dataset_names)
        print(
            f"[ERROR] Unknown datasets: {', '.join(unknown)}."
            f"Available: {known}"
        )
        return

    if interactive_mode:
        class_candidates = all_classes_union_from_datasets(
            {k: v for k, v in datasets_info.items() if k in set(selected_dataset_names)},
            output_dataset_name,
            class_names_map,
        )
        try:
            _prompt_interactive_options(
                args,
                default_output_name=output_dataset_name,
                class_candidates=class_candidates,
                class_names_map=class_names_map,
            )
        except Exception as e:
            print(f"[ERROR] Error interacting with fusion parameters: {e}")
            return
        if layout is not None and not args.target_path:
            out_key = (args.output_name or "").strip() or output_dataset_name
            target_dir = os.path.join(layout.datasets, out_key)
            output_dataset_name = out_key
    replay_cmd = None
    if interactive_mode:
        replay_cmd = build_non_interactive_command("merge", parser, args)
        print_replay_command("before launch", replay_cmd)

    try:
        train_part, val_part, test_part = parse_fusion_split_arg(args.fusion_split)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    if args.fusion_split and str(args.fusion_split).strip():
        print(
            f"[INFO] --fusion-split: train={train_part}, val={val_part}, test={test_part} "
            "(repartitioning within each bucket of the original dataset)"
        )

    class_candidates_for_selected = all_classes_union_from_datasets(
        {k: v for k, v in datasets_info.items() if k in set(selected_dataset_names)},
        output_dataset_name,
        class_names_map,
    )

    # Selected classes
    if args.classes:
        selected_classes = _parse_csv_classes(args.classes)
        if not selected_classes:
            print("[ERROR] The --classes parameter is specified, but the name list is empty.")
            return
        is_valid, missing_classes = _validate_requested_classes(
            selected_classes,
            class_candidates_for_selected,
            class_names_map,
        )
        if not is_valid:
            print(
                "[ERROR] --classes contains unknown classes for the selected datasets: "
                f"{', '.join(missing_classes)}"
            )
            if class_candidates_for_selected:
                print(
                    "[INFO] Available classes of selected datasets: "
                    f"{', '.join(class_candidates_for_selected)}"
                )
            return
        classes_auto = False
    else:
        selected_classes = class_candidates_for_selected
        classes_auto = True
        if not selected_classes:
            print(
                "[ERROR] --classes not specified: no classes found in datasets"
                "(check datasets_info.json and classes sections)."
            )
            return
        print(
            f"[INFO] --classes is not specified: a union of classes from all datasets is used"
            f"({len(selected_classes)}): {', '.join(selected_classes)}"
        )

    excluded_classes = _parse_csv_classes(args.exclude_classes)
    if excluded_classes:
        is_valid_excluded, missing_excluded = _validate_requested_classes(
            excluded_classes,
            class_candidates_for_selected,
            class_names_map,
        )
        if not is_valid_excluded:
            print(
                "[ERROR] --exclude-classes contains unknown classes for the selected datasets: "
                f"{', '.join(missing_excluded)}"
            )
            if class_candidates_for_selected:
                print(
                    "[INFO] Available classes of selected datasets: "
                    f"{', '.join(class_candidates_for_selected)}"
                )
            return
        excluded_set = set(excluded_classes)
        selected_classes = [cls for cls in selected_classes if cls not in excluded_set]
        if not selected_classes:
            print("[ERROR] --exclude-classes removed all selected classes.")
            return
        print(
            f"[INFO] Excluded classes ({len(excluded_classes)}): {', '.join(excluded_classes)}"
        )

    try:
        normalized_to_output_name, merge_targets_to_sources = build_merge_config(
            args.merge_classes, class_names_map, selected_classes
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    requested_classes = list(selected_classes)

    if args.common_classes_only:
        request_tokens = request_normalized_tokens(
            requested_classes, args.merge_classes, class_names_map
        )
        candidates = candidate_datasets_for_common_mode(
            datasets_info, output_dataset_name, request_tokens, class_names_map
        )
        if not candidates:
            req_label = "classes from --classes" if not classes_auto else "auto-assembled set of classes"
            print(
                f"[ERROR] No dataset intersects with {req_label} "
                "(and from --merge-classes if necessary). Check names and datasets_info.json."
            )
            return

        effective_classes = reduce_selected_to_common_in_candidates(
            requested_classes, class_names_map, candidates, merge_targets_to_sources
        )
        if not effective_classes:
            req_label = "classes from --classes" if not classes_auto else "autoassembled classes"
            print(
                f"[ERROR] None of the {req_label} are present at the same time"
                "in all datasets of the group (there is an intersection with the request)."
            )
            return

        if effective_classes != requested_classes:
            dropped = [c for c in requested_classes if c not in effective_classes]
            print("[WARN] --common-classes-only mode: the resulting set of classes is narrowed.")
            src_label = "Requested in --classes" if not classes_auto else "Initial set (auto)"
            print(f"[WARN] {src_label}: {', '.join(requested_classes)}")
            print(f"[WARN] Will be used: {', '.join(effective_classes)}")
            print(
                f"[WARN] Excluded (no coverage in at least one dataset from the intersection group): "
                f"{', '.join(dropped)}"
            )
            print(
                f"[INFO] Group of datasets for checking intersection: "
                f"{len(candidates)} items ({', '.join(n for n, _ in candidates)})"
            )

        selected_classes = effective_classes
        try:
            normalized_to_output_name, merge_targets_to_sources = build_merge_config(
                args.merge_classes, class_names_map, selected_classes
            )
        except ValueError as e:
            print(
                f"[ERROR] After class reduction --merge-classes is inconsistent with the resulting list: {e}"
            )
            return

    for split in ["train", "valid", "test"]:
        safe_mkdir(os.path.join(target_dir, split, "images"))
        safe_mkdir(os.path.join(target_dir, split, "labels"))

    require_all_classes = not args.include_partial_datasets
    if args.include_partial_datasets:
        print(
            "[INFO] --include-partial-datasets: merge includes datasets, "
            "who have at least one of the selected classes."
        )

    matching_datasets = []
    for dataset_name, info in datasets_info.items():
        if dataset_name == output_dataset_name:
            continue
        if dataset_name not in selected_dataset_names:
            continue
        if dataset_matches_selection(
            info,
            class_names_map,
            selected_classes,
            merge_targets_to_sources,
            require_all_classes=require_all_classes,
        ):
            matching_datasets.append((dataset_name, info))

    if not matching_datasets:
        if require_all_classes:
            print("[ERROR] No dataset contains all selected classes.")
            print(
                "[INFO] Hint: --include-partial-datasets — take datasets with any "
                "a subset of selected classes and combine frames from all such sources."
            )
        else:
            print("[ERROR] No dataset overlaps with the selected classes.")
        return

    print(f"[INFO] Found {len(matching_datasets)} matching dataset:")
    for name, _ in matching_datasets:
        print(f"   - {name}")

    temp_ctx = None
    buckets_by_dataset: dict[str, list[tuple[str, str]]] = {}
    if args.tmp_dir:
        temp_root = os.path.abspath(os.path.expanduser(args.tmp_dir))
        os.makedirs(temp_root, exist_ok=True)
    elif layout is not None:
        temp_root = os.path.join(layout.root, "tmp")
        os.makedirs(temp_root, exist_ok=True)
    else:
        # Legacy mode: temporary files only next to work data, not in system /tmp.
        legacy_tmp_parent = os.path.join(source_dir, "tmp")
        os.makedirs(legacy_tmp_parent, exist_ok=True)
        temp_ctx = tempfile.TemporaryDirectory(prefix="smartrain_cvat11_", dir=legacy_tmp_parent)
        temp_root = temp_ctx.name

    total_labels = 0
    try:
        for dataset_name, info in matching_datasets:
            if "structure" not in info:
                print(f"[ERROR] Record {dataset_name!r} does not have structure field.")
                return
            dataset_path = resolve_dataset_root_for_entry(
                dataset_name,
                info,
                workspace_root=workspace_root,
                source_catalog_dir=layout.datasets if layout else source_dir,
                legacy_source_parent=source_dir,
            )
            buckets = iter_image_label_buckets(
                dataset_path,
                info["structure"],
                info,
                dataset_name=dataset_name,
                temp_root=temp_root,
                exclude_test=args.exclude_test,
            )
            buckets_by_dataset[dataset_name] = buckets
            for _, labels_path in buckets:
                lp = Path(labels_path)
                if lp.is_dir():
                    total_labels += sum(1 for _ in lp.rglob("*.txt"))

        used_stems = {split: set() for split in ("train", "valid", "test")}
        dedup_map: dict[str, dict] = {}
        copied_count = 0
        processed_pairs = 0
        skipped_equivalent = 0
        merged_annotations = 0
        iou_dedup_removed_boxes = 0

        with tqdm(total=total_labels, desc="Dataset Processing", unit="file") as pbar:
            for dataset_name, info in matching_datasets:
                buckets = buckets_by_dataset[dataset_name]
                for images_path, labels_path in buckets:

                    pairs = _collect_label_image_pairs(images_path, labels_path)

                    if not pairs:
                        continue

                    splits_data = split_pairs_by_ratio(
                        pairs, train_part, val_part, test_part, rng=random
                    )

                    for split_name, split_pairs in splits_data.items():
                        for image_src, label_src in split_pairs:
                            processed_pairs += 1
                            objs = _parse_filtered_label_objects(
                                label_src,
                                info["classes"],
                                class_names_map,
                                selected_classes,
                                normalized_to_output_name,
                            )
                            if not objs and args.drop_empty_images:
                                pbar.update(1)
                                continue
                            img_hash = _image_content_hash(image_src)
                            sig = _canonical_label_signature(objs)
                            entry = dedup_map.get(img_hash)
                            if entry is None:
                                dedup_map[img_hash] = {
                                    "image_src": image_src,
                                    "split_name": split_name,  # keep_source_priority
                                    "dataset_name": dataset_name,
                                    "objs": objs,
                                    "sig": sig,
                                }
                                pbar.update(1)
                                continue
                            if entry["sig"] == sig:
                                skipped_equivalent += 1
                                pbar.update(1)
                                continue
                            merged, removed = _union_label_objects(
                                entry["objs"], objs, iou_threshold=0.95
                            )
                            merged_annotations += 1
                            iou_dedup_removed_boxes += removed
                            entry["objs"] = merged
                            entry["sig"] = _canonical_label_signature(merged)
                            pbar.update(1)
        for entry in dedup_map.values():
            split_name = entry["split_name"]
            dataset_name = entry["dataset_name"]
            image_src = entry["image_src"]
            image_ext = os.path.splitext(image_src)[1]
            stem = _unique_merge_stem(dataset_name, image_src, used_stems[split_name])
            image_dst = os.path.join(target_dir, split_name, "images", f"{stem}{image_ext}")
            label_dst = os.path.join(target_dir, split_name, "labels", f"{stem}.txt")
            if _write_label_objects(label_dst, entry["objs"]):
                shutil.copy2(image_src, image_dst)
                copied_count += 1
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    if args.drop_empty_images:
        pruned = prune_output_empty_label_pairs(target_dir)
        if pruned:
            print(f"[INFO] --drop-empty-images: pairs without objects removed: {pruned}")
            copied_count = max(0, copied_count - pruned)

    if args.strip_unused_classes:
        class_map_pre = {name: idx for idx, name in enumerate(selected_classes)}
        strip_stats = strip_unused_classes(
            target_dir,
            "split",
            {"classes": class_map_pre},
            class_names_map=class_names_map,
        )
        if strip_stats.removed_class_names:
            selected_classes = [
                k for k, _ in sorted(strip_stats.new_class_map.items(), key=lambda kv: kv[1])
            ]
            print(
                f"[INFO] --strip-unused-classes: removed {strip_stats.removed_class_names} "
                f"({strip_stats.classes_before} -> {strip_stats.classes_after} classes)"
            )

    print(f"\n[DEBUG] Total label files: {total_labels}")
    print(f"[DEBUG] Image+label pair processed: {processed_pairs}")
    print(f"[DEBUG] Skipped equivalent takes: {skipped_equivalent}")
    print(f"[DEBUG] Markup merges (same image, different labels): {merged_annotations}")
    print(f"[DEBUG] Removed boxes by IoU-dedup: {iou_dedup_removed_boxes}")
    print(f"[DEBUG] Unique images after dedup: {len(dedup_map)}")
    print(f"[DEBUG] Filtered and copied: {copied_count}")
    pct = (copied_count / total_labels * 100) if total_labels else 0.0
    print(f"[DEBUG] Percentage of files used: {pct:.2f}%")

    print(f"\n[OK] {copied_count} images with filtered annotations copied.")

    yaml_path = os.path.join(target_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n\n")
        f.write(f"nc: {len(selected_classes)}\n")
        f.write(f"names: {selected_classes}\n")

    print(f"[OK] Final YAML created: {yaml_path}")

    if layout is not None:
        out_key = os.path.basename(os.path.normpath(target_dir))
        _update_datasets_sidecar(layout, out_key, selected_classes, target_dir)
        print(f"[OK] Updated {layout.work_datasets_info_path()} and class_names.json in datasets/")
        try:
            source_datasets = []
            for ds_name, info in matching_datasets:
                source_datasets.append(
                    {
                        "name": ds_name,
                        "path": resolve_dataset_root_for_entry(
                            ds_name,
                            info,
                            workspace_root=workspace_root,
                            source_catalog_dir=layout.datasets,
                            legacy_source_parent=source_dir,
                        ),
                        "dataset_hash": info.get("dataset_hash"),
                    }
                )
            passport_path = write_dataset_passport(
                output_dataset_dir=target_dir,
                command="fusion",
                source_datasets=source_datasets,
                parameters=vars(args),
                workspace_root=workspace_root,
                transformations=[
                    {
                        "selected_classes": list(selected_classes),
                        "merge_classes": args.merge_classes or [],
                        "exclude_classes": excluded_classes,
                        "fusion_split": [train_part, val_part, test_part],
                        "include_partial_datasets": bool(args.include_partial_datasets),
                        "common_classes_only": bool(args.common_classes_only),
                        "exclude_test": bool(args.exclude_test),
                        "drop_empty_images": bool(args.drop_empty_images),
                        "strip_unused_classes": bool(args.strip_unused_classes),
                    }
                ],
                random_seed=DEFAULT_RANDOM_SEED,
                stats_before={"total_labels": total_labels},
                stats_after={
                    "copied_images": copied_count,
                    "processed_pairs": processed_pairs,
                    "skipped_equivalent": skipped_equivalent,
                    "merged_annotations": merged_annotations,
                    "iou_dedup_removed_boxes": iou_dedup_removed_boxes,
                    "unique_images_after_dedup": len(dedup_map),
                },
            )
            print(f"[OK] Passport: {passport_path}")
        except Exception as e:
            print(f"[WARNING] Failed to write dataset_passport.json: {e}")
    if replay_cmd:
        print_replay_command("after execution", replay_cmd)


if __name__ == "__main__":
    main()
