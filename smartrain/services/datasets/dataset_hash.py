import argparse
import hashlib
import json
import os
import sys

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.core.runtime.workspace_paths import (
    DATASETS_INFO_FILE,
    WorkspaceLayout,
    resolve_dataset_root,
    resolve_or_extract_dataset_root,
    resolve_path_under_workspace,
    resolve_workspace_root,
)


def calculate_dataset_hash(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    if not os.path.isdir(dataset_path):
        raise ValueError(f"The specified path is not a folder: {dataset_path}")

    hasher = hashlib.md5()
    ignored_files = {".DS_Store", "Thumbs.db", ".gitkeep", ".gitignore"}
    dataset_path = os.path.abspath(dataset_path)
    dataset_path_len = len(dataset_path) + 1
    items = []

    for root, dirs, files in os.walk(dataset_path):
        dirs.sort()
        files.sort()
        rel_root = root[dataset_path_len:] if len(root) > dataset_path_len else ""
        for dir_name in dirs:
            rel_path = os.path.join(rel_root, dir_name) if rel_root else dir_name
            items.append(("dir", rel_path))
        for file_name in files:
            if file_name in ignored_files:
                continue
            rel_path = os.path.join(rel_root, file_name) if rel_root else file_name
            file_path = os.path.join(root, file_name)
            try:
                file_size = os.path.getsize(file_path)
                items.append(("file", rel_path, file_size))
            except (OSError, IOError):
                continue

    items.sort()
    for item in items:
        if item[0] == "dir":
            hasher.update(b"dir:")
            hasher.update(item[1].encode("utf-8"))
            hasher.update(b"\n")
        elif item[0] == "file":
            hasher.update(b"file:")
            hasher.update(item[1].encode("utf-8"))
            hasher.update(b":")
            hasher.update(str(item[2]).encode("utf-8"))
            hasher.update(b"\n")
    return hasher.hexdigest()[:8]


def calculate_zip_metadata_hash(zip_path: str) -> str:
    ap = os.path.abspath(zip_path)
    st = os.stat(ap)
    hasher = hashlib.md5()
    hasher.update(ap.encode("utf-8"))
    hasher.update(b":")
    hasher.update(str(st.st_size).encode("utf-8"))
    hasher.update(b":")
    hasher.update(str(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))).encode("utf-8"))
    return hasher.hexdigest()[:8]


def resolve_hash_dataset_root(
    workspace_cli: str | None,
    dataset_path_pos: str | None,
    dataset_name: str | None,
    raw_dataset: str | None,
    *,
    hash_zip_metadata: bool,
) -> tuple[str, bool]:
    if raw_dataset is not None and str(raw_dataset).strip():
        name = str(raw_dataset).strip()
        root_ws = resolve_workspace_root(workspace_cli)
        layout = WorkspaceLayout(root_ws)
        info_path = layout.work_datasets_info_path()
        if not os.path.isfile(info_path):
            raise FileNotFoundError(f"{info_path} not found.")
        with open(info_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        if not isinstance(catalog, dict):
            raise ValueError(f"{info_path}: JSON object expected.")
        if name not in catalog:
            raise KeyError(f"The name {name!r} is missing from datasets/{DATASETS_INFO_FILE}.")
        entry = catalog[name]
        if not isinstance(entry, dict):
            raise TypeError(f"The {name!r} entry must be a JSON object.")
        raw_dp = entry.get("data_path")
        if isinstance(raw_dp, str) and raw_dp.strip():
            resolved = resolve_path_under_workspace(root_ws, raw_dp)
        else:
            resolved = os.path.join(layout.datasets, name)
        if hash_zip_metadata and resolved.lower().endswith(".zip") and os.path.isfile(resolved):
            return resolved, True
        extracted = resolve_or_extract_dataset_root(root_ws, name, entry, layout.datasets)
        return extracted, False

    if dataset_name is not None and str(dataset_name).strip():
        name = str(dataset_name).strip()
        root = resolve_workspace_root(workspace_cli)
        layout = WorkspaceLayout(root)
        expanded = os.path.abspath(os.path.expanduser(name))
        yaml_here = os.path.join(expanded, "data.yaml")
        if os.path.isdir(expanded) and os.path.isfile(yaml_here):
            return expanded, False
        info_path = layout.work_datasets_info_path()
        if not os.path.isfile(info_path):
            raise FileNotFoundError(
                f"The directory with data.yaml for {name!r} was not found and {info_path} is missing."
            )
        with open(info_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        if not isinstance(catalog, dict):
            raise ValueError(f"{info_path}: JSON object expected.")
        if name not in catalog:
            raise KeyError(f"The name {name!r} is missing from datasets/{DATASETS_INFO_FILE}.")
        entry = catalog[name]
        if not isinstance(entry, dict):
            raise TypeError(f"The {name!r} entry must be a JSON object.")
        root = resolve_dataset_root(layout.root, name, entry, layout.datasets)
        if hash_zip_metadata and root.lower().endswith(".zip") and os.path.isfile(root):
            return root, True
        if root.lower().endswith(".zip") and os.path.isfile(root):
            extracted = resolve_or_extract_dataset_root(layout.root, name, entry, layout.datasets)
            return extracted, False
        return root, False

    if dataset_path_pos is None or not str(dataset_path_pos).strip():
        raise ValueError(
            "Specify the path to the dataset folder, or --dataset, or --raw-dataset "
            "with --workspace (or SMART_TRAIN_WORKSPACE)."
        )
    p = os.path.abspath(os.path.expanduser(str(dataset_path_pos).strip()))
    if p.lower().endswith(".zip") and os.path.isfile(p):
        if hash_zip_metadata:
            return p, True
        raise ValueError(
            "For .zip by positional path, specify --hash-zip-metadata or use "
            "--raw-dataset with workspace (unpacking into cache)."
        )
    return p, False


def build_hash_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(description="Calculating a dataset hash based on the structure, file names and their sizes")
    parser.add_argument("dataset_path", type=str, nargs="?", default=None, help="Path to dataset folder")
    parser.add_argument("--workspace", type=str, default=None, help="Workspace root for --dataset/--raw-dataset")
    parser.add_argument("--dataset", type=str, default=None, metavar="NAME")
    parser.add_argument("--raw-dataset", type=str, default=None, metavar="NAME")
    parser.add_argument("--hash-zip-metadata", action="store_true")
    parser.add_argument("--validate", type=str, default=None, help="Expected hash value for validation")
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = build_hash_arg_parser().parse_args(argv)
    sel = [
        bool(args.dataset_path and str(args.dataset_path).strip()),
        bool(args.dataset and str(args.dataset).strip()),
        bool(args.raw_dataset and str(args.raw_dataset).strip()),
    ]
    if sum(sel) > 1:
        print("[ERROR] Specify exactly one of: path to dataset, --dataset, --raw-dataset.", file=sys.stderr)
        sys.exit(2)

    try:
        root, zip_meta = resolve_hash_dataset_root(
            args.workspace,
            args.dataset_path,
            args.dataset,
            args.raw_dataset,
            hash_zip_metadata=bool(args.hash_zip_metadata),
        )
        computed_hash = calculate_zip_metadata_hash(root) if zip_meta else calculate_dataset_hash(root)
        if args.validate:
            if computed_hash.lower() == args.validate.lower():
                print(f"Validation successful. Hash matches: {computed_hash}")
                sys.exit(0)
            print("Validation failed.")
            print(f"Expected: {args.validate}")
            print(f"Received: {computed_hash}")
            sys.exit(1)
        print(computed_hash)
        sys.exit(0)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

