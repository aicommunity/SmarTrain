from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.services.models.release_model_naming import (
    is_registry_bundle_path,
    is_release_metadata,
    load_release_metadata,
    parse_canonical_release_stem,
    release_json_path_for_pt,
    sanitize_release_stem,
    task_type_from_release_stem_task,
)


@dataclass(frozen=True)
class ReleaseModelEntry:
    pt_path: Path
    stem: str
    dataset_dir: Path
    rel_path: str
    release_json: Path


@dataclass(frozen=True)
class RenameOperation:
    src: Path
    dst: Path


@dataclass
class RenamePlan:
    entry: ReleaseModelEntry
    old_stem: str
    new_stem: str
    operations: list[RenameOperation] = field(default_factory=list)
    old_pt_path: Path = field(default_factory=Path)
    new_pt_path: Path = field(default_factory=Path)
    new_json_path: Path = field(default_factory=Path)
    new_release_dir: Path = field(default_factory=Path)


@dataclass(frozen=True)
class RenameResult:
    old_stem: str
    new_stem: str
    renamed_paths: tuple[Path, ...]
    skipped: bool = False
    reason: str = ""


class ReleaseRenameError(ValueError):
    pass


def discover_release_models(layout: WorkspaceLayout) -> list[ReleaseModelEntry]:
    models_dir = Path(layout.models)
    if not models_dir.is_dir():
        return []

    found: list[ReleaseModelEntry] = []
    root = Path(layout.root).resolve()
    for pt_path in sorted(models_dir.rglob("*.pt")):
        if not pt_path.is_file():
            continue
        if is_registry_bundle_path(pt_path):
            continue
        json_path = release_json_path_for_pt(pt_path)
        if not load_release_metadata(json_path):
            continue
        rel = str(pt_path.relative_to(root)) if pt_path.is_relative_to(root) else str(pt_path)
        found.append(
            ReleaseModelEntry(
                pt_path=pt_path.resolve(),
                stem=pt_path.stem,
                dataset_dir=pt_path.parent.resolve(),
                rel_path=rel,
                release_json=json_path.resolve(),
            )
        )
    return found


def collect_stem_siblings(parent_dir: Path, old_stem: str) -> list[Path]:
    if not parent_dir.is_dir():
        return []
    out: list[Path] = []
    for child in parent_dir.iterdir():
        name = child.name
        if name == old_stem and child.is_dir():
            out.append(child.resolve())
        elif name.startswith(old_stem + ".") or name.startswith(old_stem + "_"):
            out.append(child.resolve())
    return out


def _replace_stem_in_name(name: str, old_stem: str, new_stem: str) -> str:
    if name == old_stem:
        return new_stem
    if name.startswith(old_stem + "."):
        return new_stem + name[len(old_stem) :]
    if name.startswith(old_stem + "_"):
        return new_stem + name[len(old_stem) :]
    return name


def build_rename_plan(entry: ReleaseModelEntry, new_stem: str) -> RenamePlan:
    sanitized = sanitize_release_stem(new_stem)
    if not sanitized or sanitized == "unknown":
        raise ReleaseRenameError("new release name is empty or invalid")

    old_stem = entry.stem
    if sanitized == old_stem:
        plan = RenamePlan(entry=entry, old_stem=old_stem, new_stem=sanitized)
        plan.old_pt_path = entry.pt_path
        plan.new_pt_path = entry.pt_path
        plan.new_json_path = entry.release_json
        plan.new_release_dir = entry.dataset_dir / sanitized
        return plan

    siblings = collect_stem_siblings(entry.dataset_dir, old_stem)
    if entry.pt_path.resolve() not in siblings:
        siblings.append(entry.pt_path.resolve())

    operations: list[RenameOperation] = []
    for src in siblings:
        dst_name = _replace_stem_in_name(src.name, old_stem, sanitized)
        dst = (entry.dataset_dir / dst_name).resolve()
        if dst.exists() and dst != src:
            raise ReleaseRenameError(f"target already exists: {dst}")
        operations.append(RenameOperation(src=src, dst=dst))

    operations.sort(key=lambda op: len(op.src.name), reverse=True)

    new_pt = entry.dataset_dir / f"{sanitized}.pt"
    new_json = entry.dataset_dir / f"{sanitized}.json"
    plan = RenamePlan(
        entry=entry,
        old_stem=old_stem,
        new_stem=sanitized,
        operations=operations,
        old_pt_path=entry.pt_path,
        new_pt_path=new_pt.resolve(),
        new_json_path=new_json.resolve(),
        new_release_dir=(entry.dataset_dir / sanitized).resolve(),
    )
    return plan


def _patch_path_value(value: Any, old_pt: Path, new_pt: Path) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    old_s = str(old_pt)
    new_s = str(new_pt)
    if value == old_s:
        return new_s
    if value == old_pt.name:
        return new_pt.name
    old_stem = old_pt.stem
    new_stem = new_pt.stem
    if old_stem in value:
        return value.replace(old_stem, new_stem)
    return value


def _update_training_info_from_stem(payload: dict[str, Any], new_stem: str) -> bool:
    parsed = parse_canonical_release_stem(new_stem)
    if parsed is None:
        return False
    training = payload.get("training")
    if not isinstance(training, dict):
        return False
    ti = training.get("training_info")
    if not isinstance(ti, dict):
        return False
    ti["model"] = parsed.model
    ti["task_type"] = task_type_from_release_stem_task(parsed.task)
    return True


def _patch_release_json(payload: dict[str, Any], plan: RenamePlan) -> None:
    old_pt = plan.old_pt_path
    new_pt = plan.new_pt_path
    new_json = plan.new_json_path
    new_release_dir = plan.new_release_dir

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts["model_path"] = str(new_pt)
        artifacts["json_path"] = str(new_json)
        artifacts["release_dir"] = str(new_release_dir)
        artifacts["train_copy_dir"] = str(new_release_dir / "train")
        artifacts["test_copy_dir"] = str(new_release_dir / "test")

    _update_training_info_from_stem(payload, plan.new_stem)


def _patch_training_metadata(meta_path: Path, new_stem: str) -> None:
    if not meta_path.is_file():
        return
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    parsed = parse_canonical_release_stem(new_stem)
    if parsed is None:
        return
    ti = payload.get("training_info")
    if not isinstance(ti, dict):
        return
    ti["model"] = parsed.model
    ti["task_type"] = task_type_from_release_stem_task(parsed.task)
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _patch_sidecar_meta(sidecar_path: Path, old_pt: Path, new_pt: Path) -> None:
    if not sidecar_path.is_file():
        return
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return

    model_path = payload.get("path")
    if isinstance(model_path, str):
        payload["path"] = _patch_path_value(model_path, old_pt, new_pt)

    filename = payload.get("filename")
    if isinstance(filename, str):
        payload["filename"] = _replace_stem_in_name(filename, old_pt.stem, new_pt.stem)

    source_path = payload.get("source_path")
    if isinstance(source_path, str):
        payload["source_path"] = _patch_path_value(source_path, old_pt, new_pt)

    sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_release_rename(plan: RenamePlan) -> RenameResult:
    if plan.old_stem == plan.new_stem:
        return RenameResult(
            old_stem=plan.old_stem,
            new_stem=plan.new_stem,
            renamed_paths=(),
            skipped=True,
            reason="new name is the same as current name",
        )

    renamed: list[Path] = []
    op_by_dst: dict[Path, RenameOperation] = {}

    for op in plan.operations:
        op.src.rename(op.dst)
        renamed.append(op.dst)
        op_by_dst[op.dst.resolve()] = op

    release_json_dst = plan.new_json_path
    if release_json_dst.is_file():
        try:
            payload = json.loads(release_json_dst.read_text(encoding="utf-8"))
        except Exception as e:
            raise ReleaseRenameError(f"failed to read release metadata: {release_json_dst}: {e}") from e
        if not is_release_metadata(payload):
            raise ReleaseRenameError(f"invalid release metadata after rename: {release_json_dst}")
        _patch_release_json(payload, plan)
        release_json_dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    training_meta = plan.new_release_dir / "training_metadata.json"
    _patch_training_metadata(training_meta, plan.new_stem)

    old_pt = plan.old_pt_path
    new_pt = plan.new_pt_path
    for dst in renamed:
        if dst.name.endswith(".meta.json"):
            _patch_sidecar_meta(dst, old_pt, new_pt)

    return RenameResult(
        old_stem=plan.old_stem,
        new_stem=plan.new_stem,
        renamed_paths=tuple(renamed),
        skipped=False,
    )
