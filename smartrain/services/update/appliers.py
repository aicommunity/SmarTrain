"""Apply UpdatePlan steps to bring a workspace toward canonical layout."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from smartrain.core.runtime.run_artifacts import (
    ensure_run_layout,
    materialize_preferred_run_model,
    normalize_model_references_in_metadata,
    preferred_run_model_path,
)
from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.services.datasets.data_yaml_normalize import (
    normalize_data_yaml_file,
    normalize_data_yaml_mapping,
)
from smartrain.services.models.release_model_naming import load_release_metadata, release_json_path_for_pt
from smartrain.services.models.release_models_manifest import (
    entry_key_for_pt,
    load_manifest,
    remove_entry,
    save_manifest,
    upsert_entry,
)
from smartrain.services.update.path_norm import (
    needs_path_rewrite,
    normalize_stored_path,
    to_posix_rel,
    workspace_rel_posix,
)
from smartrain.services.update.plan import UpdateRisk, UpdateStatus, UpdateStep


def _workspace_rel(layout: WorkspaceLayout, path: Path) -> str:
    return workspace_rel_posix(layout, path)


def _rewrite_sidecar_paths(layout: WorkspaceLayout, json_path: Path) -> tuple[bool, str]:
    """
    Rewrite abs / backslash paths in a release sidecar to workspace-relative POSIX.

    Returns ``(ok, message)``. ``ok`` is False when a field still needs rewrite but
    could not be remapped.
    """
    payload = load_release_metadata(json_path)
    if not payload:
        return False, "unreadable sidecar"
    release_dir: Path | None = None
    try:
        # Sidecar lives at <release>/models/<stem>.json
        if json_path.parent.name == "models":
            release_dir = json_path.parent.parent
        else:
            release_dir = json_path.parent
    except Exception:
        release_dir = None

    changed = False
    failed: list[str] = []

    def _rel_or_fail(field: str, value: str) -> str:
        nonlocal changed
        if not needs_path_rewrite(value) and "\\" not in value:
            # Still canonicalize legacy train/test tails when possible
            remapped = normalize_stored_path(layout, value, release_dir=release_dir)
            if remapped is not None and remapped != to_posix_rel(value):
                changed = True
                return remapped
            posix = to_posix_rel(value)
            if posix != value:
                changed = True
            return posix
        remapped = normalize_stored_path(layout, value, release_dir=release_dir)
        if remapped is None:
            failed.append(field)
            return value
        if remapped != value:
            changed = True
        return remapped

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ("model_path", "json_path", "release_dir", "train_copy_dir", "test_copy_dir"):
            val = artifacts.get(key)
            if isinstance(val, str) and val.strip():
                artifacts[key] = _rel_or_fail(f"artifacts.{key}", val)
    source = payload.get("source")
    if isinstance(source, dict):
        for key in ("source_run", "source_run_relative"):
            sr = source.get(key)
            if isinstance(sr, str) and sr.strip():
                if key == "source_run_relative" and not needs_path_rewrite(sr):
                    posix = to_posix_rel(sr)
                    if posix != sr:
                        changed = True
                        source[key] = posix
                else:
                    source[key] = _rel_or_fail(f"source.{key}", sr)

    if changed:
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed:
        return False, f"could not remap: {', '.join(failed)}"
    return True, "sidecar paths rewritten" if changed else "already relative posix"


def _normalize_runtime_data_yaml(layout: WorkspaceLayout, yaml_path: Path) -> tuple[bool, str]:
    """Strip abs ``path:`` and force POSIX relative splits in a runtime data yaml."""
    if not yaml_path.is_file():
        return False, "missing"
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"read error: {e}"
    if not isinstance(raw, dict):
        return False, "not a mapping"

    # Prefer dataset root from path: if remappable under workspace datasets/
    path_val = raw.get("path")
    dataset_root = yaml_path.parent
    if isinstance(path_val, str) and path_val.strip():
        remapped = normalize_stored_path(layout, path_val, must_exist=False)
        if remapped and remapped.startswith("datasets/"):
            dataset_root = Path(layout.root) / remapped
        elif (yaml_path.parent.parent.name in {"runs", "models"} or True):
            # Fall back: use remapped path only for normalize root when it exists
            if remapped:
                cand = Path(layout.root) / remapped
                if cand.is_dir():
                    dataset_root = cand

    new_data = normalize_data_yaml_mapping(str(dataset_root), raw)
    # Force posix on split strings
    for k in ("train", "val", "test", "minival"):
        v = new_data.get(k)
        if isinstance(v, str):
            new_data[k] = to_posix_rel(v)
        elif isinstance(v, list):
            new_data[k] = [to_posix_rel(str(x)) for x in v]

    order_first = ("train", "val", "test", "minival")
    ordered: dict = {}
    for k in order_first:
        if k in new_data:
            ordered[k] = new_data[k]
    for k, v in new_data.items():
        if k not in ordered:
            ordered[k] = v
    new_dump = yaml.safe_dump(ordered, allow_unicode=True, default_flow_style=False, sort_keys=False)
    try:
        old = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        unchanged = old == yaml.safe_load(new_dump)
    except Exception:
        unchanged = False
    if unchanged:
        return False, "already normalized"
    yaml_path.write_text(new_dump, encoding="utf-8")
    return True, "runtime yaml normalized"


def _unify_root_release(layout: WorkspaceLayout, pt_path: Path) -> None:
    release_dir = pt_path.parent
    models_sub = release_dir / "models"
    models_sub.mkdir(parents=True, exist_ok=True)
    stem = pt_path.stem
    for child in list(release_dir.iterdir()):
        if not child.is_file():
            continue
        if child.name.startswith(f"{stem}.") or child.name.startswith(f"{stem}_") or child.name == f"{stem}.pt":
            dest = models_sub / child.name
            if dest.exists() and dest.resolve() != child.resolve():
                continue
            if child.resolve() != dest.resolve():
                shutil.move(str(child), str(dest))
    new_pt = models_sub / f"{stem}.pt"
    new_json = models_sub / f"{stem}.json"
    if new_json.is_file():
        _rewrite_sidecar_paths(layout, new_json)
        payload = load_release_metadata(new_json)
        if payload:
            artifacts = payload.setdefault("artifacts", {})
            if isinstance(artifacts, dict):
                artifacts["model_path"] = _workspace_rel(layout, new_pt)
                artifacts["json_path"] = _workspace_rel(layout, new_json)
                artifacts["release_dir"] = _workspace_rel(layout, release_dir)
            new_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if new_pt.is_file():
        key = entry_key_for_pt(new_pt)
        comment = ""
        manifest = load_manifest(layout)
        entries = manifest.get("entries") or {}
        if isinstance(entries, dict):
            folder_key = f"{release_dir.parent.name}/{release_dir.name}"
            for cand in (key, folder_key):
                ent = entries.get(cand)
                if isinstance(ent, dict):
                    comment = str(ent.get("comment") or "")
                    break
            if folder_key in entries and folder_key != key:
                remove_entry(layout, folder_key)
        upsert_entry(layout, entry_key=key, model_path=new_pt, comment=comment)


def _unify_r2_release(layout: WorkspaceLayout, sibling_pt: Path, release_dir: Path) -> None:
    models_sub = release_dir / "models"
    models_sub.mkdir(parents=True, exist_ok=True)
    stem = sibling_pt.stem
    dest_pt = models_sub / sibling_pt.name
    if not dest_pt.exists():
        shutil.move(str(sibling_pt), str(dest_pt))
    old_json = sibling_pt.parent / f"{stem}.json"
    dest_json = models_sub / f"{stem}.json"
    if old_json.is_file() and not dest_json.exists():
        shutil.move(str(old_json), str(dest_json))
    for child in list(sibling_pt.parent.iterdir()):
        if not child.is_file():
            continue
        if child.name.startswith(f"{stem}.") or child.name.startswith(f"{stem}_"):
            dest = models_sub / child.name
            if not dest.exists():
                shutil.move(str(child), str(dest))
    if dest_json.is_file():
        _rewrite_sidecar_paths(layout, dest_json)
        payload = load_release_metadata(dest_json)
        if payload:
            artifacts = payload.setdefault("artifacts", {})
            if isinstance(artifacts, dict):
                artifacts["model_path"] = _workspace_rel(layout, dest_pt)
                artifacts["json_path"] = _workspace_rel(layout, dest_json)
                artifacts["release_dir"] = _workspace_rel(layout, release_dir)
            dest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if dest_pt.is_file():
        upsert_entry(layout, entry_key=entry_key_for_pt(dest_pt), model_path=dest_pt)


def apply_step(layout: WorkspaceLayout, step: UpdateStep, *, dry_run: bool) -> UpdateStep:
    if dry_run:
        step.status = UpdateStatus.DRY_RUN
        step.message = "would apply"
        return step
    if step.risk == UpdateRisk.SKIP or step.action == "skip":
        step.status = UpdateStatus.SKIPPED
        step.message = "skipped"
        return step
    try:
        action = step.action
        if action in {"migrate_train", "migrate_tests", "migrate_layout"}:
            root = Path(layout.root) / step.detail if step.detail and not step.detail.startswith("/") else Path(step.detail or layout.root)
            if step.paths:
                p0 = Path(step.paths[0])
                if not p0.is_absolute():
                    p0 = Path(layout.root) / p0
                if p0.name in {"train", "test", "test-ultralytics"} or p0.name.startswith("test_"):
                    root = p0.parent
                elif p0.is_file():
                    root = p0.parent
                elif (p0 / "training_metadata.json").is_file() or (p0 / "train").is_dir():
                    root = p0
            ensure_run_layout(str(root))
            # If root runtime yaml still present with identical tmp, ensure_run_layout removes it.
            # Differing content is handled by drop_root_runtime_yaml ASK steps.
            step.status = UpdateStatus.APPLIED
            step.message = "layout migrated"
        elif action == "drop_root_runtime_yaml":
            src = Path(step.paths[0])
            if not src.is_absolute():
                src = Path(layout.root) / src
            if src.is_file():
                src.unlink()
            step.status = UpdateStatus.APPLIED
            step.message = "root runtime yaml removed (kept tmp/)"
        elif action == "materialize_weights":
            root = Path(layout.root)
            if len(step.paths) >= 2:
                preferred = Path(step.paths[1])
                if not preferred.is_absolute():
                    preferred = root / preferred
                run_root = preferred.parent.parent if preferred.parent.name == "models" else preferred.parent
            else:
                run_root = root
            materialize_preferred_run_model(str(run_root), ext=".pt", move=True, normalize_metadata=True)
            step.status = UpdateStatus.APPLIED
            step.message = "weights materialized"
        elif action == "unify_root_release":
            pt = Path(step.paths[0])
            if not pt.is_absolute():
                pt = Path(layout.root) / pt
            _unify_root_release(layout, pt)
            step.status = UpdateStatus.APPLIED
            step.message = "release unified"
        elif action == "unify_r2_release":
            sibling = Path(step.paths[0])
            release_dir = Path(step.paths[1])
            if not sibling.is_absolute():
                sibling = Path(layout.root) / sibling
            if not release_dir.is_absolute():
                release_dir = Path(layout.root) / release_dir
            _unify_r2_release(layout, sibling, release_dir)
            step.status = UpdateStatus.APPLIED
            step.message = "r2 release unified"
        elif action == "rewrite_sidecar_paths":
            jp = Path(step.paths[0])
            if not jp.is_absolute():
                jp = Path(layout.root) / jp
            ok, msg = _rewrite_sidecar_paths(layout, jp)
            if ok:
                step.status = UpdateStatus.APPLIED
                step.message = msg
            else:
                step.status = UpdateStatus.FAILED
                step.message = msg
        elif action == "rekey_manifest":
            old_key, new_key = step.paths[0], step.paths[1]
            manifest = load_manifest(layout)
            entries = manifest.get("entries") or {}
            if isinstance(entries, dict) and old_key in entries:
                old = entries.pop(old_key)
                if new_key not in entries:
                    entries[new_key] = old
                save_manifest(layout, manifest)
            step.status = UpdateStatus.APPLIED
            step.message = f"rekeyed {old_key} → {new_key}"
        elif action == "relativize_manifest_path":
            key = step.detail
            manifest = load_manifest(layout)
            entries = manifest.get("entries") or {}
            if isinstance(entries, dict) and key in entries and isinstance(entries[key], dict):
                mp = entries[key].get("model_path")
                if isinstance(mp, str) and mp.strip():
                    remapped = normalize_stored_path(layout, mp)
                    if remapped is None:
                        step.status = UpdateStatus.FAILED
                        step.message = f"could not remap model_path: {mp}"
                        return step
                    entries[key]["model_path"] = remapped
                    save_manifest(layout, manifest)
            step.status = UpdateStatus.APPLIED
            step.message = "manifest path relativized"
        elif action == "drop_manifest_entry":
            remove_entry(layout, step.detail)
            step.status = UpdateStatus.APPLIED
            step.message = "stale entry removed"
        elif action == "normalize_yaml":
            yaml_path = Path(step.paths[0])
            if not yaml_path.is_absolute():
                yaml_path = Path(layout.root) / yaml_path
            changed, msg = normalize_data_yaml_file(str(yaml_path.parent), dry_run=False)
            step.status = UpdateStatus.APPLIED if changed else UpdateStatus.SKIPPED
            step.message = msg
        elif action == "normalize_runtime_yaml":
            yaml_path = Path(step.paths[0])
            if not yaml_path.is_absolute():
                yaml_path = Path(layout.root) / yaml_path
            changed, msg = _normalize_runtime_data_yaml(layout, yaml_path)
            step.status = UpdateStatus.APPLIED if changed else UpdateStatus.SKIPPED
            step.message = msg
        elif action == "normalize_metadata":
            meta = Path(step.paths[0])
            if not meta.is_absolute():
                meta = Path(layout.root) / meta
            run_root = meta.parent
            preferred = preferred_run_model_path(str(run_root), ".pt")
            normalize_model_references_in_metadata(str(run_root), Path(preferred).name)
            step.status = UpdateStatus.APPLIED
            step.message = "metadata normalized"
        else:
            step.status = UpdateStatus.SKIPPED
            step.message = f"unknown action: {action}"
    except Exception as e:
        step.status = UpdateStatus.FAILED
        step.message = str(e)
    return step


def apply_plan(
    layout: WorkspaceLayout,
    steps: list[UpdateStep],
    *,
    dry_run: bool = False,
    include_ask: bool = False,
    ask_callback=None,
) -> list[UpdateStep]:
    """Apply steps. SAFE always; ASK when include_ask or ask_callback returns True."""
    results: list[UpdateStep] = []
    for step in steps:
        if step.risk == UpdateRisk.SKIP:
            step.status = UpdateStatus.SKIPPED
            step.message = "skip category"
            results.append(step)
            continue
        if step.risk == UpdateRisk.ASK:
            if include_ask:
                allow = True
            elif ask_callback is not None:
                allow = bool(ask_callback(step))
            else:
                step.status = UpdateStatus.SKIPPED
                step.message = "ask step skipped (use --apply-all or interactive)"
                results.append(step)
                continue
            if not allow:
                step.status = UpdateStatus.SKIPPED
                step.message = "declined by user"
                results.append(step)
                continue
        results.append(apply_step(layout, step, dry_run=dry_run))
    return results
