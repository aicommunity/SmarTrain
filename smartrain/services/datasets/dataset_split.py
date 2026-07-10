"""CLI: repartition one dataset into train/valid/test splits."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import prompt_prefilled_text, prompt_text
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.services.datasets.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.services.datasets.dataset_cli_catalog import (
    EMPTY_DATASETS_INFO_MESSAGE,
    load_datasets_catalog,
    try_prompt_dataset_interactive,
)
from smartrain.services.datasets.dataset_cli_common import update_datasets_sidecar
from smartrain.services.datasets.dataset_hash import calculate_dataset_hash
from smartrain.services.datasets.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.services.datasets.dataset_split_core import (
    DEFAULT_RANDOM_SEED,
    TRAIN_PART,
    VAL_PART,
    TEST_PART,
    parse_split_ratio_arg,
    split_pairs_by_ratio,
    unique_output_stem,
)
from smartrain.services.datasets.image_label_pairs import collect_label_image_pairs


def build_split_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Repartition one dataset into train/valid/test splits")
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (aka {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Source dataset key from datasets_info.json")
    p.add_argument("--output-name", type=str, default=None, help="Output dataset name (default <dataset>_split)")
    p.add_argument(
        "--split-ratio",
        type=str,
        default=None,
        help=(
            "Shares train,val,test with random repartition of frames within each bucket "
            f"(three numbers separated by commas, sum 1.0). Default {TRAIN_PART},{VAL_PART},{TEST_PART}."
        ),
    )
    p.add_argument(
        "--exclude-test",
        action="store_true",
        help="Exclude test buckets from the source dataset input (does not affect output layout).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned output without copying files.")
    return p


from smartrain.services.datasets.data_yaml_writer import write_data_yaml_from_class_map


def _write_data_yaml(out_dir: str, class_map: dict[str, Any]) -> None:
    write_data_yaml_from_class_map(out_dir, class_map, val_rel="valid/images")


def _interactive_fill(args: argparse.Namespace, dataset_names: list[str]) -> None:
    print("[INFO] Interactive split mode")
    from smartrain.cli_entrypoints.support.cli_prompts import prompt_choice

    args.dataset = prompt_choice("Dataset", dataset_names, default=(args.dataset or dataset_names[0]))
    output_default = str(args.output_name or "").strip()
    if output_default:
        args.output_name = prompt_prefilled_text("Output dataset name", output_default).strip() or None
    else:
        args.output_name = prompt_text("Output dataset name (empty=auto)", default="").strip() or None
    split_default = args.split_ratio or f"{TRAIN_PART},{VAL_PART},{TEST_PART}"
    args.split_ratio = prompt_prefilled_text("Split ratio train,val,test", split_default).strip() or split_default


def _ensure_split_dirs(out_dir: str) -> None:
    for split in ("train", "valid", "test"):
        os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, split, "labels"), exist_ok=True)


def _copy_split_pairs(
    splits_data: dict[str, list[tuple[str, str]]],
    out_dir: str,
) -> dict[str, int]:
    used_stems = {split: set() for split in ("train", "valid", "test")}
    counts = {"train": 0, "valid": 0, "test": 0}
    for split_name, split_pairs in splits_data.items():
        for image_src, label_src in split_pairs:
            image_ext = os.path.splitext(image_src)[1]
            stem = unique_output_stem(image_src, used_stems[split_name])
            image_dst = os.path.join(out_dir, split_name, "images", f"{stem}{image_ext}")
            label_dst = os.path.join(out_dir, split_name, "labels", f"{stem}.txt")
            shutil.copy2(image_src, image_dst)
            if os.path.isfile(label_src):
                shutil.copy2(label_src, label_dst)
            counts[split_name] += 1
    return counts


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_split_arg_parser()
    args = parser.parse_args(argv)
    interactive_allowed = is_interactive_allowed(argv)

    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    catalog = load_datasets_catalog(layout)
    if not catalog:
        print(EMPTY_DATASETS_INFO_MESSAGE)
        return

    interactive_used = False
    if try_prompt_dataset_interactive(
        args=args,
        argv=argv,
        fill=lambda: _interactive_fill(args, sorted(catalog.keys())),
    ):
        interactive_used = True

    if not args.dataset:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return
    if args.dataset not in catalog:
        print(f"[ERROR] Unknown dataset: {args.dataset}")
        return

    try:
        train_part, val_part, test_part = parse_split_ratio_arg(args.split_ratio)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    entry = catalog[args.dataset]
    entry_dict = entry if isinstance(entry, dict) else {}
    structure = str(entry_dict.get("structure", "split"))
    class_map = entry_dict.get("classes", {})
    if not isinstance(class_map, dict):
        class_map = {}

    src_root = resolve_dataset_root_for_entry(
        args.dataset,
        entry_dict,
        workspace_root=layout.root,
        source_catalog_dir=layout.datasets,
        legacy_source_parent=layout.datasets,
    )

    out_base = args.output_name or f"{args.dataset}_split"
    out_name = next_dataset_name(layout.datasets, out_base)
    out_dir = os.path.join(layout.datasets, out_name)

    temp_ctx: tempfile.TemporaryDirectory[str] | None = None
    temp_root = ""
    if structure == "cvat11":
        temp_ctx = tempfile.TemporaryDirectory(prefix="smartrain_split_")
        temp_root = temp_ctx.name

    random.seed(DEFAULT_RANDOM_SEED)
    bucket_counts: dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    total_pairs = 0
    try:
        buckets = iter_image_label_buckets(
            src_root,
            structure,
            entry_dict,
            dataset_name=args.dataset,
            temp_root=temp_root,
            exclude_test=bool(args.exclude_test),
        )
        splits_accum: dict[str, list[tuple[str, str]]] = {"train": [], "valid": [], "test": []}
        for images_path, labels_path in buckets:
            pairs = collect_label_image_pairs(images_path, labels_path)
            if not pairs:
                continue
            splits_data = split_pairs_by_ratio(pairs, train_part, val_part, test_part, rng=random)
            for split_name, split_pairs in splits_data.items():
                splits_accum[split_name].extend(split_pairs)
                bucket_counts[split_name] += len(split_pairs)
            total_pairs += len(pairs)
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    if args.split_ratio and str(args.split_ratio).strip():
        print(
            f"[INFO] --split-ratio: train={train_part}, val={val_part}, test={test_part} "
            "(repartitioning within each bucket of the source dataset)"
        )

    replay_cmd = None
    if interactive_used:
        replay_cmd = build_non_interactive_command("split", parser, args)
        print_replay_command("before launch", replay_cmd)

    if args.dry_run:
        print(
            f"[OK] dry-run: dataset={args.dataset}, output={out_name}, "
            f"pairs={total_pairs}, train={bucket_counts['train']}, "
            f"valid={bucket_counts['valid']}, test={bucket_counts['test']}"
        )
        if replay_cmd:
            print_replay_command("after execution", replay_cmd)
        return

    _ensure_split_dirs(out_dir)
    copied_counts = _copy_split_pairs(splits_accum, out_dir)
    if class_map:
        _write_data_yaml(out_dir, class_map)

    out_hash = calculate_dataset_hash(out_dir)
    if class_map:
        update_datasets_sidecar(
            layout=layout,
            output_key=out_name,
            class_map=class_map,
            target_dir=out_dir,
            output_hash=out_hash,
            structure="split",
        )

    passport_path = write_dataset_passport(
        output_dataset_dir=out_dir,
        command="split",
        source_datasets=[
            {
                "name": args.dataset,
                "path": src_root,
                "dataset_hash": entry_dict.get("dataset_hash"),
            }
        ],
        parameters=vars(args),
        workspace_root=layout.root,
        transformations=[
            {
                "split_ratio": [train_part, val_part, test_part],
                "exclude_test": bool(args.exclude_test),
                "source_dataset": args.dataset,
            }
        ],
        random_seed=DEFAULT_RANDOM_SEED,
        stats_before={"total_pairs": total_pairs},
        stats_after={
            "copied_train": copied_counts["train"],
            "copied_valid": copied_counts["valid"],
            "copied_test": copied_counts["test"],
            "output_hash": out_hash,
        },
    )
    print(f"[OK] Dataset created: {out_dir}")
    print(
        f"[OK] Split counts: train={copied_counts['train']}, "
        f"valid={copied_counts['valid']}, test={copied_counts['test']}"
    )
    print(f"[OK] Passport: {passport_path}")
    if replay_cmd:
        print_replay_command("after execution", replay_cmd)


if __name__ == "__main__":
    main()
