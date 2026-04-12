"""
Rewrite absolute paths under a workspace root to portable relative strings in JSON/YAML artifacts.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smartrain.data_yaml_normalize import iter_dataset_roots_with_data_yaml, normalize_data_yaml_file
from smartrain.path_portable import relativize_abs_paths_in_obj, relativize_if_under
from smartrain.workspace_paths import WorkspaceLayout


@dataclass
class RepairReport:
    data_yaml_updated: int = 0
    data_yaml_unchanged: int = 0
    passports_updated: int = 0
    datasets_info_updated: int = 0
    training_metadata_updated: int = 0
    zip_meta_updated: int = 0
    datasets_list_updated: int = 0
    messages: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.messages.append(msg)


def _is_under(workspace_root: str, path: str) -> bool:
    try:
        a = os.path.abspath(os.path.expanduser(path))
        b = os.path.abspath(os.path.expanduser(workspace_root))
        return a == b or a.startswith(b + os.sep)
    except (OSError, ValueError):
        return False


def _repair_training_metadata_file(path: Path, workspace_root: str) -> tuple[bool, dict[str, Any]]:
    wr = os.path.abspath(os.path.expanduser(workspace_root))
    with open(path, encoding="utf-8") as f:
        meta: dict[str, Any] = json.load(f)
    if not isinstance(meta, dict):
        return False, meta
    changed = False
    ws = meta.get("workspace")
    if isinstance(ws, dict):
        root_val = ws.get("root")
        if isinstance(root_val, str) and root_val.strip():
            rv = root_val.strip()
            if os.path.isabs(rv) and os.path.abspath(rv) == wr:
                ws["root"] = "."
                changed = True
        for key in ("dataset_path_relative", "run_directory_relative"):
            v = ws.get(key)
            if isinstance(v, str) and os.path.isabs(v.strip()):
                rel = relativize_if_under(workspace_root, v.strip())
                if rel != v:
                    ws[key] = rel
                    changed = True
    ti = meta.get("training_info")
    if isinstance(ti, dict):
        ds = ti.get("dataset")
        if isinstance(ds, dict):
            pa = ds.get("path_absolute")
            if isinstance(pa, str) and pa.strip() and _is_under(workspace_root, pa):
                rel = relativize_if_under(workspace_root, pa.strip())
                if rel is not None:
                    ds["path_under_workspace"] = rel
                    del ds["path_absolute"]
                    changed = True
    return changed, meta


def _repair_zip_meta_file(path: Path, workspace_root: str) -> tuple[bool, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        meta: dict[str, Any] = json.load(f)
    if not isinstance(meta, dict):
        return False, meta
    raw = meta.get("zip_path")
    if not isinstance(raw, str) or not raw.strip():
        return False, meta
    if not os.path.isabs(raw.strip()):
        return False, meta
    rel = relativize_if_under(workspace_root, raw.strip())
    if rel == raw.strip() or os.path.isabs(str(rel)):
        return False, meta
    meta = dict(meta)
    meta["zip_path"] = rel
    return True, meta


def _repair_datasets_info(data: dict[str, Any], workspace_root: str) -> tuple[dict[str, Any], bool]:
    wr = os.path.abspath(os.path.expanduser(workspace_root))
    out = dict(data)
    any_changed = False
    for name, entry in list(out.items()):
        if not isinstance(entry, dict):
            continue
        e = dict(entry)
        row_changed = False
        for key in ("data_path", "source_ref"):
            v = e.get(key)
            if isinstance(v, str) and v.strip() and os.path.isabs(v.strip()):
                rel = relativize_if_under(wr, v.strip())
                if rel != v:
                    e[key] = rel
                    row_changed = True
        if row_changed:
            out[name] = e
            any_changed = True
    return out, any_changed


def repair_workspace_paths(
    workspace_root: str,
    *,
    dry_run: bool = False,
    include_datasets_list: bool = False,
) -> RepairReport:
    """
    Normalize portable paths under ``workspace_root`` (datasets yaml/passports, index JSON,
    run ``training_metadata.json``, zip cache ``__meta__.json``).
    """
    report = RepairReport()
    wr = os.path.abspath(os.path.expanduser(workspace_root))
    layout = WorkspaceLayout(wr)
    datasets_dir = layout.datasets

    for d in iter_dataset_roots_with_data_yaml(datasets_dir):
        changed, msg = normalize_data_yaml_file(d, dry_run=dry_run)
        if changed:
            report.data_yaml_updated += 1
            report.log(f"data.yaml: {d}: {msg}")
        else:
            report.data_yaml_unchanged += 1

    for passport_path in Path(datasets_dir).rglob("dataset_passport.json"):
        if not passport_path.is_file():
            continue
        with open(passport_path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            continue
        repaired = relativize_abs_paths_in_obj(raw, wr)
        if repaired != raw:
            report.passports_updated += 1
            report.log(f"passport: {passport_path}")
            if not dry_run:
                with open(passport_path, "w", encoding="utf-8") as f:
                    json.dump(repaired, f, ensure_ascii=False, indent=2)

    info_path = layout.work_datasets_info_path()
    if os.path.isfile(info_path):
        with open(info_path, encoding="utf-8") as f:
            idx = json.load(f)
        if isinstance(idx, dict):
            new_idx, ch = _repair_datasets_info(idx, wr)
            if ch:
                report.datasets_info_updated = 1
                report.log(f"datasets_info: {info_path}")
                if not dry_run:
                    with open(info_path, "w", encoding="utf-8") as f:
                        json.dump(new_idx, f, ensure_ascii=False, indent=4)

    runs_root = Path(layout.runs)
    if runs_root.is_dir():
        for meta_path in runs_root.rglob("training_metadata.json"):
            ch, new_meta = _repair_training_metadata_file(meta_path, wr)
            if ch:
                report.training_metadata_updated += 1
                report.log(f"training_metadata: {meta_path}")
                if not dry_run:
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(new_meta, f, ensure_ascii=False, indent=2)

    cache_root = Path(layout.extracted_datasets)
    if cache_root.is_dir():
        for meta_path in cache_root.rglob("__meta__.json"):
            ch, new_meta = _repair_zip_meta_file(meta_path, wr)
            if ch:
                report.zip_meta_updated += 1
                report.log(f"zip __meta__: {meta_path}")
                if not dry_run:
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(new_meta, f, ensure_ascii=False, indent=2)

    if include_datasets_list:
        list_path = os.path.join(layout.raw_data, "datasets_list.txt")
        if os.path.isfile(list_path):
            with open(list_path, encoding="utf-8") as f:
                lines = f.read().splitlines()
            new_lines: list[str] = []
            list_changed = False
            for line in lines:
                s = line.strip()
                if s and not s.startswith("#") and os.path.isabs(s) and _is_under(wr, s):
                    rel = relativize_if_under(wr, s)
                    new_lines.append(rel if rel is not None else line)
                    if rel != s:
                        list_changed = True
                else:
                    new_lines.append(line)
            if list_changed:
                report.datasets_list_updated = 1
                report.log(f"datasets_list: {list_path}")
                if not dry_run:
                    with open(list_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    return report


def print_repair_report(report: RepairReport, *, dry_run: bool) -> None:
    prefix = "[dry-run] " if dry_run else ""
    print(f"\n{prefix}[INFO] Path repair: data_yaml touched={report.data_yaml_updated}, unchanged_yaml={report.data_yaml_unchanged}")
    print(
        f"{prefix}[INFO] passports={report.passports_updated}, "
        f"datasets_info={report.datasets_info_updated}, "
        f"training_metadata={report.training_metadata_updated}, "
        f"zip_meta={report.zip_meta_updated}, "
        f"datasets_list={report.datasets_list_updated}"
    )
    for m in report.messages[:50]:
        print(f"{prefix}[INFO] {m}")
    if len(report.messages) > 50:
        print(f"{prefix}[INFO] ... and {len(report.messages) - 50} more")
