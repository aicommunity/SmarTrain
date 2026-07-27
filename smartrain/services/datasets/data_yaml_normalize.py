"""CLI compatibility module for dataset ``data.yaml`` normalization."""
from __future__ import annotations

import argparse
import json
import os
import sys

from smartrain.core.runtime.data_yaml_normalize import (
    _rewrite_split_string,
    iter_dataset_roots_with_data_yaml,
    normalize_data_yaml_file,
    normalize_data_yaml_mapping,
)

__all__ = [
    "iter_dataset_roots_with_data_yaml",
    "normalize_data_yaml_file",
    "normalize_data_yaml_mapping",
    "run_normalize",
]


def run_normalize(
    datasets_dir: str,
    *,
    dry_run: bool = False,
    as_json: bool = False,
) -> int:
    results: list[dict[str, str]] = []
    datasets_abs = os.path.abspath(os.path.expanduser(datasets_dir))
    for d in iter_dataset_roots_with_data_yaml(datasets_dir):
        changed, msg = normalize_data_yaml_file(d, dry_run=dry_run)
        label = os.path.relpath(d, datasets_abs)
        results.append({"dir": label, "changed": str(changed), "detail": msg})
    if as_json:
        print(json.dumps({"datasets_dir": datasets_dir, "results": results}, ensure_ascii=False, indent=2))
    else:
        for r in results:
            flag = "YES" if r["changed"] == "True" else "no"
            print(f"[{flag}] {r['dir']}: {r['detail']}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Normalize data.yaml under a datasets directory (portable Ultralytics).")
    p.add_argument(
        "--datasets-dir",
        type=str,
        default=None,
        help="Directory containing dataset subfolders (default: <workspace>/datasets).",
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root (dataset dir defaults to <workspace>/datasets).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions without writing files.")
    p.add_argument("--json", action="store_true", dest="as_json", help="Machine-readable output.")
    return p


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_arg_parser().parse_args(argv)
    from smartrain.core.runtime.workspace_paths import WorkspaceLayout, resolve_workspace_root

    if args.datasets_dir:
        ddir = os.path.abspath(os.path.expanduser(args.datasets_dir))
    else:
        root = resolve_workspace_root(args.workspace)
        ddir = WorkspaceLayout(root).datasets
    raise SystemExit(run_normalize(ddir, dry_run=bool(args.dry_run), as_json=bool(args.as_json)))


if __name__ == "__main__":
    main()

