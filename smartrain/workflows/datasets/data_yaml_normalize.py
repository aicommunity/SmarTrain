"""
Normalize dataset data.yaml for portable Ultralytics layout:
- drop `path` (dataset root = directory containing this file)
- strip leading `./` from train/val/test
- rewrite absolute split paths that point inside the dataset directory as relative
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


def _strip_dot_slash(s: str) -> str:
    t = " ".join(((s or "").replace("\\", "/")).split())
    while t.startswith("./"):
        t = t[2:]
    return t


def _abs_under_root(root: str, candidate: str) -> str | None:
    try:
        ap = os.path.abspath(candidate)
        root_abs = os.path.abspath(root)
        if ap == root_abs or ap.startswith(root_abs + os.sep):
            return ap
    except (OSError, ValueError):
        return None
    return None


def _to_rel_split(root: str, raw: str) -> str:
    s = _strip_dot_slash(raw)
    if not s:
        return s
    root_abs = os.path.abspath(root)
    joined = os.path.abspath(os.path.join(root_abs, s.replace("/", os.sep)))
    try:
        rel = os.path.relpath(joined, root_abs)
        if not rel.startswith(".."):
            return rel.replace("\\", "/")
    except ValueError:
        pass
    abs_s = os.path.abspath(s.replace("/", os.sep))
    try:
        rel = os.path.relpath(abs_s, root_abs)
        if not rel.startswith(".."):
            return rel.replace("\\", "/")
    except ValueError:
        pass
    return s.replace("\\", "/")


def _foreign_absolute_to_split_relative(dataset_root: str, t: str) -> str | None:
    root = Path(dataset_root)
    sp = t.replace("\\", "/").strip()
    if not sp:
        return None
    is_abs = sp.startswith("/") or (len(sp) > 2 and sp[1] == ":")
    if not is_abs:
        return None
    for rel in ("train/images", "val/images", "valid/images", "test/images"):
        cand = root / rel.replace("/", os.sep)
        if not cand.is_dir():
            continue
        if sp.endswith("/" + rel):
            return rel
    if (root / "images").is_dir():
        for tail in ("/train/images", "/val/images", "/valid/images", "/test/images"):
            if sp.endswith(tail):
                return None
        if sp.endswith("/images"):
            return "images"
    return None


def _rewrite_split_string(dataset_root: str, raw: str) -> str:
    t = _to_rel_split(dataset_root, raw)
    mapped = _foreign_absolute_to_split_relative(dataset_root, t)
    if mapped is not None:
        return mapped
    return t


def _normalize_split_field(root: str, value: Any) -> Any:
    if isinstance(value, list):
        return [_rewrite_split_string(root, str(x)) for x in value if isinstance(x, str)]
    if isinstance(value, str):
        return _rewrite_split_string(root, value)
    return value


def normalize_data_yaml_mapping(dataset_root: str, data: Mapping[str, Any]) -> dict[str, Any]:
    root = os.path.abspath(os.path.expanduser(dataset_root))
    out: dict[str, Any] = dict(data)
    out.pop("path", None)
    for k in ("train", "val", "test", "minival"):
        if k not in out or out[k] in (None, ""):
            continue
        out[k] = _normalize_split_field(root, out[k])
    return out


def _canonical_dump(data: dict[str, Any]) -> str:
    order_first = ("train", "val", "test", "minival")
    ordered: dict[str, Any] = {}
    for k in order_first:
        if k in data:
            ordered[k] = data[k]
    for k, v in data.items():
        if k not in ordered:
            ordered[k] = v
    return yaml.safe_dump(ordered, allow_unicode=True, default_flow_style=False, sort_keys=False)


def normalize_data_yaml_file(dataset_dir: str, *, dry_run: bool = False) -> tuple[bool, str]:
    p = Path(dataset_dir) / "data.yaml"
    if not p.is_file():
        return False, "no data.yaml"
    try:
        text = p.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
    except Exception as e:
        return False, f"read error: {e}"
    if not isinstance(raw, dict):
        return False, "not a mapping"
    new_data = normalize_data_yaml_mapping(str(p.parent), raw)
    new_dump = _canonical_dump(new_data)
    try:
        unchanged = yaml.safe_load(new_dump) == yaml.safe_load(text)
    except Exception:
        unchanged = False
    if unchanged:
        return False, "already normalized"
    if dry_run:
        return True, "would update"
    p.write_text(new_dump, encoding="utf-8")
    return True, "updated"


def iter_dataset_roots_with_data_yaml(datasets_root: str) -> list[str]:
    root = Path(os.path.abspath(os.path.expanduser(datasets_root)))
    if not root.is_dir():
        return []
    found: set[str] = set()
    for p in root.rglob("data.yaml"):
        if not p.is_file():
            continue
        if p.parent.resolve() == root.resolve():
            continue
        found.add(str(p.parent.resolve()))
    return sorted(found)


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

