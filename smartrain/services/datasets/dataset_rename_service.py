from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from smartrain.core.runtime.path_portable import relativize_if_under
from smartrain.core.runtime.workspace_paths import (
    WORKSPACE_QUEUE_BASENAME,
    WorkspaceLayout,
    resolve_dataset_root,
    resolve_path_under_workspace,
)
from smartrain.services.datasets.dataset_cli_common import load_dataset_catalog


class DatasetRenameError(ValueError):
    pass


@dataclass(frozen=True)
class DirRenameOperation:
    src: Path
    dst: Path
    label: str


@dataclass(frozen=True)
class FileUpdateOperation:
    path: Path
    label: str


@dataclass
class DatasetRenamePlan:
    layout: WorkspaceLayout
    old_name: str
    new_name: str
    catalog_entry: dict[str, Any]
    dir_renames: list[DirRenameOperation] = field(default_factory=list)
    file_updates: list[FileUpdateOperation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    move_data_path: bool = False


@dataclass(frozen=True)
class DatasetRenameResult:
    old_name: str
    new_name: str
    renamed_dirs: tuple[Path, ...]
    updated_files: tuple[Path, ...]
    skipped: bool = False
    reason: str = ""


_DATASET_NAME_RE = re.compile(r"[^\w.\-+]+", re.UNICODE)


def sanitize_dataset_name(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        raise DatasetRenameError("new dataset name is empty")
    if "/" in raw or "\\" in raw or ".." in raw:
        raise DatasetRenameError(f"invalid dataset name: {raw!r}")
    sanitized = _DATASET_NAME_RE.sub("_", raw).strip("._")
    if not sanitized:
        raise DatasetRenameError("new dataset name is empty after sanitization")
    return sanitized[:180]


def _default_dataset_dir(layout: WorkspaceLayout, name: str) -> Path:
    return Path(layout.datasets) / name


def _is_standard_dataset_root(layout: WorkspaceLayout, old_name: str, resolved_root: Path) -> bool:
    default = _default_dataset_dir(layout, old_name).resolve()
    return resolved_root.resolve() == default


def _rel_workspace(path: Path, workspace_root: Path) -> str:
    rel = relativize_if_under(str(workspace_root), str(path.resolve()))
    return rel if rel else str(path.resolve())


def _replace_dataset_token_in_path(path_str: str, old_name: str, new_name: str) -> str:
    if path_str == old_name:
        return new_name
    for sep in ("/", "\\"):
        old_seg = f"{sep}{old_name}{sep}"
        new_seg = f"{sep}{new_name}{sep}"
        if old_seg in path_str:
            path_str = path_str.replace(old_seg, new_seg)
        if path_str.endswith(f"{sep}{old_name}"):
            path_str = path_str[: -len(old_name)] + new_name
        if path_str.startswith(f"{old_name}{sep}"):
            path_str = new_name + path_str[len(old_name) :]
    return path_str.replace(f"datasets/{old_name}", f"datasets/{new_name}").replace(
        f"runs/{old_name}", f"runs/{new_name}"
    ).replace(f"models/{old_name}", f"models/{new_name}")


def _replace_in_text(text: str, old_name: str, new_name: str) -> str:
    patterns = [
        (f"--dataset {old_name}", f"--dataset {new_name}"),
        (f"--data {old_name}", f"--data {new_name}"),
        (f"datasets/{old_name}", f"datasets/{new_name}"),
        (f"runs/{old_name}", f"runs/{new_name}"),
        (f"models/{old_name}", f"models/{new_name}"),
    ]
    out = text
    for old_pat, new_pat in patterns:
        out = out.replace(old_pat, new_pat)
    return out


def _patch_json_dataset_refs(obj: Any, old_name: str, new_name: str, workspace_root: Path) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key == "name" and isinstance(value, str) and value == old_name:
                out[key] = new_name
            elif key in ("path", "data_path", "data", "source_path") and isinstance(value, str):
                out[key] = _replace_dataset_token_in_path(value, old_name, new_name)
            elif key == "dataset" and isinstance(value, str) and value == old_name:
                out[key] = new_name
            elif key == "dataset" and isinstance(value, dict):
                ds = dict(value)
                if isinstance(ds.get("name"), str) and ds["name"] == old_name:
                    ds["name"] = new_name
                out[key] = _patch_json_dataset_refs(ds, old_name, new_name, workspace_root)
            else:
                out[key] = _patch_json_dataset_refs(value, old_name, new_name, workspace_root)
        return out
    if isinstance(obj, list):
        return [_patch_json_dataset_refs(item, old_name, new_name, workspace_root) for item in obj]
    if isinstance(obj, str) and obj == old_name:
        return new_name
    return obj


def _collect_analytics_renames(layout: WorkspaceLayout, old_name: str, new_name: str) -> list[DirRenameOperation]:
    ops: list[DirRenameOperation] = []
    root = Path(layout.analytics)
    if not root.is_dir():
        return ops

    reports = root / "datasets-reports"
    if reports.is_dir():
        prefix = f"{old_name}_"
        for child in sorted(reports.iterdir()):
            if child.is_dir() and child.name.startswith(prefix):
                dst = reports / (new_name + child.name[len(old_name) :])
                ops.append(DirRenameOperation(src=child.resolve(), dst=dst.resolve(), label="analytics report dir"))

    for sub, exact in (("pr_curves", f"pr_all_classes_{old_name}.png"), ("inference_tests", f"{old_name}.csv")):
        parent = root / sub
        if not parent.is_dir():
            continue
        src = parent / exact
        if src.is_file():
            new_exact = exact.replace(old_name, new_name)
            dst = parent / new_exact
            ops.append(DirRenameOperation(src=src.resolve(), dst=dst.resolve(), label=f"analytics {sub}"))

    return ops


def _collect_passport_updates(layout: WorkspaceLayout, old_name: str, new_name: str) -> list[FileUpdateOperation]:
    ops: list[FileUpdateOperation] = []
    datasets_root = Path(layout.datasets)
    if not datasets_root.is_dir():
        return ops
    for passport in sorted(datasets_root.rglob("dataset_passport.json")):
        try:
            payload = json.loads(passport.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        patched = _patch_json_dataset_refs(payload, old_name, new_name, Path(layout.root))
        if patched != payload:
            ops.append(FileUpdateOperation(path=passport.resolve(), label="dataset_passport.json"))
    return ops


def _collect_run_artifact_updates(
    layout: WorkspaceLayout, old_name: str, new_name: str, runs_dir: Path
) -> list[FileUpdateOperation]:
    ops: list[FileUpdateOperation] = []
    if not runs_dir.is_dir():
        return ops
    for meta in sorted(runs_dir.rglob("training_metadata.json")):
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        patched = _patch_json_dataset_refs(payload, old_name, new_name, Path(layout.root))
        if patched != payload:
            ops.append(FileUpdateOperation(path=meta.resolve(), label="training_metadata.json"))
    for args_yaml in sorted(runs_dir.rglob("args.yaml")):
        try:
            payload = yaml.safe_load(args_yaml.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        patched = _patch_json_dataset_refs(payload, old_name, new_name, Path(layout.root))
        if patched != payload:
            ops.append(FileUpdateOperation(path=args_yaml.resolve(), label="args.yaml"))
    return ops


def _collect_model_metadata_updates(
    layout: WorkspaceLayout, old_name: str, new_name: str, models_dir: Path
) -> list[FileUpdateOperation]:
    ops: list[FileUpdateOperation] = []
    if not models_dir.is_dir():
        return ops
    for json_path in sorted(models_dir.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        patched = _patch_json_dataset_refs(payload, old_name, new_name, Path(layout.root))
        if patched != payload:
            ops.append(FileUpdateOperation(path=json_path.resolve(), label="model metadata"))
    return ops


def _queue_path(layout: WorkspaceLayout) -> Path:
    return Path(layout.root) / WORKSPACE_QUEUE_BASENAME


def _collect_queue_update(layout: WorkspaceLayout, old_name: str, new_name: str) -> FileUpdateOperation | None:
    qp = _queue_path(layout)
    if not qp.is_file():
        return None
    text = qp.read_text(encoding="utf-8")
    if _replace_in_text(text, old_name, new_name) == text:
        return None
    return FileUpdateOperation(path=qp.resolve(), label="queue.txt")


def _extracted_cache_cleanup(layout: WorkspaceLayout, old_name: str) -> list[DirRenameOperation]:
    cache_root = Path(layout.extracted_datasets)
    if not cache_root.is_dir():
        return []
    ops: list[DirRenameOperation] = []
    for child in sorted(cache_root.iterdir()):
        if child.is_dir() and old_name in child.name:
            trash = cache_root / f".trash_{child.name}"
            ops.append(DirRenameOperation(src=child.resolve(), dst=trash.resolve(), label="extracted cache cleanup"))
    return ops


def build_rename_plan(
    layout: WorkspaceLayout,
    old_name: str,
    new_name: str,
    *,
    move_data_path: bool = False,
) -> DatasetRenamePlan:
    old_key = (old_name or "").strip()
    if not old_key:
        raise DatasetRenameError("dataset name is empty")

    sanitized = sanitize_dataset_name(new_name)
    catalog = load_dataset_catalog(layout)
    if old_key not in catalog:
        raise DatasetRenameError(f"dataset not found in catalog: {old_key!r}")
    entry = catalog[old_key]
    if not isinstance(entry, dict):
        raise DatasetRenameError(f"invalid catalog entry for {old_key!r}")

    plan = DatasetRenamePlan(
        layout=layout,
        old_name=old_key,
        new_name=sanitized,
        catalog_entry=dict(entry),
        move_data_path=move_data_path,
    )

    if sanitized == old_key:
        return plan

    if sanitized in catalog:
        raise DatasetRenameError(f"dataset name already exists: {sanitized!r}")

    workspace_root = Path(layout.root).resolve()
    resolved_root = Path(
        resolve_dataset_root(layout.root, old_key, entry, layout.datasets)
    ).resolve()

    default_dir = _default_dataset_dir(layout, old_key)
    new_default_dir = _default_dataset_dir(layout, sanitized)
    standard_root = _is_standard_dataset_root(layout, old_key, resolved_root)

    if not standard_root and not move_data_path:
        raise DatasetRenameError(
            f"dataset {old_key!r} uses custom data_path {entry.get('data_path')!r}; "
            "pass --move-data-path to relocate data"
        )

    def _add_dir_rename(src: Path, dst: Path, label: str) -> None:
        if not src.exists():
            return
        if dst.exists():
            raise DatasetRenameError(f"target already exists: {dst}")
        plan.dir_renames.append(DirRenameOperation(src=src.resolve(), dst=dst.resolve(), label=label))

    if standard_root:
        _add_dir_rename(default_dir, new_default_dir, "dataset directory")
    elif move_data_path:
        new_data_path = _replace_dataset_token_in_path(
            str(entry.get("data_path") or _rel_workspace(resolved_root, workspace_root)),
            old_key,
            sanitized,
        )
        new_resolved = resolve_path_under_workspace(layout.root, new_data_path)
        _add_dir_rename(resolved_root, Path(new_resolved), "custom data_path directory")

    runs_old = Path(layout.runs) / old_key
    runs_new = Path(layout.runs) / sanitized
    _add_dir_rename(runs_old, runs_new, "runs directory")

    models_old = Path(layout.models) / old_key
    models_new = Path(layout.models) / sanitized
    _add_dir_rename(models_old, models_new, "models directory")

    plan.dir_renames.extend(_collect_analytics_renames(layout, old_key, sanitized))
    plan.dir_renames.extend(_extracted_cache_cleanup(layout, old_key))

    plan.file_updates.append(FileUpdateOperation(path=Path(layout.work_datasets_info_path()).resolve(), label="datasets_info.json"))
    plan.file_updates.extend(_collect_passport_updates(layout, old_key, sanitized))

    runs_scan = runs_old if runs_old.exists() else runs_new
    models_scan = models_old if models_old.exists() else models_new
    plan.file_updates.extend(_collect_run_artifact_updates(layout, old_key, sanitized, runs_scan))
    plan.file_updates.extend(_collect_model_metadata_updates(layout, old_key, sanitized, models_scan))

    queue_op = _collect_queue_update(layout, old_key, sanitized)
    if queue_op:
        plan.file_updates.append(queue_op)

    if entry.get("modified") is True:
        plan.warnings.append("dataset has modified=true; scan may overwrite metadata on next refresh")

    return plan


def _write_catalog_rename(layout: WorkspaceLayout, plan: DatasetRenamePlan) -> None:
    info_path = Path(layout.work_datasets_info_path())
    with open(info_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    if not isinstance(catalog, dict):
        raise DatasetRenameError("datasets_info.json is not a dict")

    entry = dict(catalog.pop(plan.old_name))
    if "data_path" in entry and isinstance(entry["data_path"], str):
        entry["data_path"] = _replace_dataset_token_in_path(entry["data_path"], plan.old_name, plan.new_name)
    elif _is_standard_dataset_root(
        layout,
        plan.old_name,
        Path(resolve_dataset_root(layout.root, plan.old_name, plan.catalog_entry, layout.datasets)),
    ):
        entry["data_path"] = _rel_workspace(_default_dataset_dir(layout, plan.new_name), Path(layout.root))

    catalog[plan.new_name] = entry

    tmp = info_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(catalog, ensure_ascii=False, indent=4), encoding="utf-8")
    tmp.replace(info_path)


def _apply_file_update(path: Path, old_name: str, new_name: str) -> None:
    if not path.is_file():
        return
    if path.name == "args.yaml":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        patched = _patch_json_dataset_refs(payload, old_name, new_name, path.parent)
        path.write_text(yaml.safe_dump(patched, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        patched = _patch_json_dataset_refs(payload, old_name, new_name, path.parent)
        path.write_text(json.dumps(patched, ensure_ascii=False, indent=2), encoding="utf-8")
        return
  # queue.txt
    text = path.read_text(encoding="utf-8")
    path.write_text(_replace_in_text(text, old_name, new_name), encoding="utf-8")


def format_plan_report(plan: DatasetRenamePlan) -> str:
    lines = [f"[dataset rename] {plan.old_name} -> {plan.new_name}"]
    for w in plan.warnings:
        lines.append(f"  warning: {w}")
    for op in plan.dir_renames:
        rel_src = _rel_workspace(op.src, Path(plan.layout.root))
        rel_dst = _rel_workspace(op.dst, Path(plan.layout.root))
        lines.append(f"  rename ({op.label}): {rel_src} -> {rel_dst}")
    seen: set[Path] = set()
    for op in plan.file_updates:
        if op.path in seen:
            continue
        seen.add(op.path)
        rel = _rel_workspace(op.path, Path(plan.layout.root))
        lines.append(f"  update ({op.label}): {rel}")
    return "\n".join(lines)


def _map_path_after_renames(path: Path, dir_renames: list[DirRenameOperation]) -> Path:
    resolved = path.resolve()
    for op in dir_renames:
        try:
            rel = resolved.relative_to(op.src.resolve())
        except ValueError:
            continue
        return (op.dst / rel).resolve()
    return resolved


def apply_dataset_rename(plan: DatasetRenamePlan, *, dry_run: bool = False) -> DatasetRenameResult:
    if plan.old_name == plan.new_name:
        return DatasetRenameResult(
            old_name=plan.old_name,
            new_name=plan.new_name,
            renamed_dirs=(),
            updated_files=(),
            skipped=True,
            reason="new name is the same as current name",
        )

    if dry_run:
        return DatasetRenameResult(
            old_name=plan.old_name,
            new_name=plan.new_name,
            renamed_dirs=tuple(op.dst for op in plan.dir_renames),
            updated_files=tuple(op.path for op in plan.file_updates),
            skipped=False,
        )

    renamed_dirs: list[Path] = []
    for op in plan.dir_renames:
        op.src.rename(op.dst)
        renamed_dirs.append(op.dst)

    _write_catalog_rename(plan.layout, plan)

    updated_files: list[Path] = []
    seen: set[Path] = set()
    for op in plan.file_updates:
        target = _map_path_after_renames(op.path, plan.dir_renames)
        if target in seen:
            continue
        seen.add(target)
        if op.label == "datasets_info.json":
            continue
        _apply_file_update(target, plan.old_name, plan.new_name)
        updated_files.append(target)

    for trash in Path(plan.layout.extracted_datasets).glob(".trash_*"):
        if trash.is_dir():
            shutil.rmtree(trash, ignore_errors=True)

    return DatasetRenameResult(
        old_name=plan.old_name,
        new_name=plan.new_name,
        renamed_dirs=tuple(renamed_dirs),
        updated_files=tuple(updated_files),
        skipped=False,
    )
