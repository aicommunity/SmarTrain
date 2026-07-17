from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.services.models.release_model_naming import (
    is_release_metadata,
    load_release_metadata,
    release_json_path_for_pt,
)

MANIFEST_FILENAME = "releases_manifest.json"
MANIFEST_VERSION = 1


def manifest_path(layout: WorkspaceLayout) -> Path:
    return Path(layout.models) / MANIFEST_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_manifest() -> dict[str, Any]:
    return {"version": MANIFEST_VERSION, "entries": {}}


def load_manifest(layout: WorkspaceLayout) -> dict[str, Any]:
    path = manifest_path(layout)
    if not path.is_file():
        return _empty_manifest()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_manifest()
    if not isinstance(payload, dict):
        return _empty_manifest()
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        payload["entries"] = {}
    payload.setdefault("version", MANIFEST_VERSION)
    return payload


def save_manifest(layout: WorkspaceLayout, payload: dict[str, Any]) -> None:
    path = manifest_path(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    out["version"] = MANIFEST_VERSION
    entries = out.get("entries")
    if not isinstance(entries, dict):
        out["entries"] = {}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def _artifact_release_dir(pt_path: Path) -> Path | None:
    meta = load_release_metadata(release_json_path_for_pt(pt_path))
    if not meta:
        return None
    artifacts = meta.get("artifacts") or {}
    release_dir = artifacts.get("release_dir")
    if not isinstance(release_dir, str) or not release_dir.strip():
        return None
    try:
        return Path(release_dir).expanduser().resolve()
    except Exception:
        return None


def release_dir_for_pt(pt_path: Path) -> Path:
    artifact_dir = _artifact_release_dir(pt_path)
    if artifact_dir is not None:
        return artifact_dir
    # Unified layout: weights under ``<bundle>/models/<stem>.pt``.
    if pt_path.parent.name == "models":
        parent = pt_path.parent.parent
        if parent.is_dir() and _path_has_ancestor_named(parent, "models"):
            return parent.resolve()
    # Legacy R1: bundle folder name matched the weight stem.
    if pt_path.parent.name == pt_path.stem:
        return pt_path.parent.resolve()
    # R2 / legacy flat sibling bundle dir.
    return (pt_path.parent / pt_path.stem).resolve()


def is_nested_release_layout(pt_path: Path) -> bool:
    if not pt_path.is_file() or pt_path.suffix.lower() != ".pt":
        return False
    bundle_root = release_dir_for_pt(pt_path)
    parent = pt_path.parent.resolve()
    if parent == bundle_root.resolve():
        return True
    return parent.name == "models" and parent.parent.resolve() == bundle_root.resolve()


def is_unified_release_bundle(release_dir: Path) -> bool:
    """True when release keeps run-like tree with canonical weights under ``models/``."""
    d = release_dir.expanduser().resolve()
    if not d.is_dir():
        return False
    models_sub = d / "models"
    if not models_sub.is_dir():
        return False
    for pt in sorted(models_sub.glob("*.pt")):
        if load_release_metadata(release_json_path_for_pt(pt)):
            return True
    return False


def find_release_pt_in_dir(directory: Path) -> Path | None:
    """Locate a released ``.pt`` inside a release/bundle directory."""
    d = directory.expanduser().resolve()
    if not d.is_dir():
        return None
    models_sub = d / "models"
    if models_sub.is_dir():
        for pt in sorted(models_sub.glob("*.pt")):
            if pt.is_file() and load_release_metadata(release_json_path_for_pt(pt)):
                return pt
    legacy = d / f"{d.name}.pt"
    if legacy.is_file() and load_release_metadata(release_json_path_for_pt(legacy)):
        return legacy
    for pt in sorted(d.glob("*.pt")):
        if pt.is_file() and load_release_metadata(release_json_path_for_pt(pt)):
            return pt
    if legacy.is_file():
        return legacy
    pts = sorted(p for p in d.glob("*.pt") if p.is_file())
    return pts[0] if len(pts) == 1 else None


def _path_has_ancestor_named(path: Path, name: str) -> bool:
    return any(part == name for part in path.parts)


def is_workspace_release_bundle(path: Path) -> bool:
    """True if ``path`` is inside a workspace release catalog entry (flat or nested).

    Used to keep ``model convert`` from treating release dirs as training runs
    (they also contain ``training_metadata.json``).
    """
    p = path.expanduser().resolve()
    if p.is_file() and p.suffix.lower() == ".pt":
        if load_release_metadata(release_json_path_for_pt(p)):
            return True
        if is_nested_release_layout(p) and _path_has_ancestor_named(p, "models") and not _path_has_ancestor_named(
            p, "runs"
        ):
            return True

    start = p if p.is_dir() else p.parent
    for cand in [start, *start.parents]:
        if _path_has_ancestor_named(cand, "runs"):
            continue
        nested_pt = find_release_pt_in_dir(cand)
        if nested_pt is not None:
            if load_release_metadata(release_json_path_for_pt(nested_pt)):
                return True
            if (
                is_nested_release_layout(nested_pt)
                and _path_has_ancestor_named(cand, "models")
                and not _path_has_ancestor_named(cand, "runs")
            ):
                return True
        sibling_pt = cand.parent / f"{cand.name}.pt"
        if sibling_pt.is_file() and load_release_metadata(release_json_path_for_pt(sibling_pt)):
            return True
    return False


def dataset_name_for_pt(pt_path: Path) -> str:
    return release_dir_for_pt(pt_path).parent.name


def entry_key_for_pt(pt_path: Path) -> str:
    dataset = dataset_name_for_pt(pt_path)
    return f"{dataset}/{pt_path.stem}"


def resolve_entry_key_from_pt_or_dir(path: Path, *, layout: WorkspaceLayout | None = None) -> str | None:
    p = path.expanduser().resolve()
    if p.is_dir():
        pt = find_release_pt_in_dir(p)
        if pt is None:
            sibling = p.parent / f"{p.name}.pt"
            if sibling.is_file():
                pt = sibling
            else:
                return None
        return entry_key_for_pt(pt)
    if p.suffix.lower() == ".pt" and p.is_file():
        return entry_key_for_pt(p)
    if layout is not None and p.suffix.lower() == ".json":
        payload = load_release_metadata(p)
        if payload:
            artifacts = payload.get("artifacts") or {}
            model_path = artifacts.get("model_path")
            if isinstance(model_path, str) and model_path.strip():
                mp = Path(model_path)
                if not mp.is_absolute() and layout is not None:
                    mp = (Path(layout.root) / mp).resolve()
                if mp.is_file():
                    return entry_key_for_pt(mp)
    return None


def _comment_from_sidecar(pt_path: Path) -> str:
    payload = load_release_metadata(release_json_path_for_pt(pt_path))
    if not isinstance(payload, dict):
        return ""
    comment = payload.get("comment")
    return str(comment).strip() if comment is not None else ""


def get_comment(layout: WorkspaceLayout, entry_key: str) -> str:
    manifest = load_manifest(layout)
    entries = manifest.get("entries") or {}
    entry = entries.get(entry_key)
    if isinstance(entry, dict):
        comment = entry.get("comment")
        if comment is not None and str(comment).strip():
            return str(comment).strip()
    return ""


def get_comment_for_pt(layout: WorkspaceLayout, pt_path: Path) -> str:
    key = entry_key_for_pt(pt_path)
    comment = get_comment(layout, key)
    if comment:
        return comment
    return _comment_from_sidecar(pt_path)


def _comment_from_release_sidecars(directory: Path) -> str:
    """Read ``comment`` from any release metadata ``*.json`` in the bundle."""
    models_sub = directory / "models"
    if models_sub.is_dir():
        for jp in sorted(models_sub.glob("*.json")):
            payload = load_release_metadata(jp)
            if not payload:
                continue
            comment = payload.get("comment")
            if comment is not None and str(comment).strip():
                return str(comment).strip()
    for jp in sorted(directory.glob("*.json")):
        payload = load_release_metadata(jp)
        if not payload:
            continue
        comment = payload.get("comment")
        if comment is not None and str(comment).strip():
            return str(comment).strip()
    return ""


def _manifest_comment_for_models_relative(layout: WorkspaceLayout, release_dir: Path) -> str:
    """Lookup ``releases_manifest`` by ``<dataset>/<release_folder>`` (R3 folder ≠ weight stem)."""
    try:
        models_root = Path(layout.models).expanduser().resolve()
        rel = release_dir.resolve().relative_to(models_root)
    except Exception:
        return ""
    parts = rel.parts
    if len(parts) < 2:
        return ""
    # models/<dataset>/<release_dir>/...
    key = f"{parts[0]}/{parts[1]}"
    return get_comment(layout, key)


def get_comment_for_run_dir(layout: WorkspaceLayout, run_dir: str) -> str:
    p = Path(run_dir).expanduser().resolve()
    if not p.is_dir():
        return ""
    pt = find_release_pt_in_dir(p)
    if pt is not None:
        comment = get_comment_for_pt(layout, pt)
        if comment:
            return comment
    sibling = p.parent / f"{p.name}.pt"
    if sibling.is_file():
        comment = get_comment_for_pt(layout, sibling)
        if comment:
            return comment
    # R3 / polluted release dirs: weight may live only under nested models/<run_id>.pt
    # while comment lives on detect_*.json sidecar or manifests keyed by folder name.
    manifest_comment = _manifest_comment_for_models_relative(layout, p)
    if manifest_comment:
        return manifest_comment
    return _comment_from_release_sidecars(p)


def layout_for_run_dir(run_dir: str) -> WorkspaceLayout | None:
    p = Path(run_dir).expanduser().resolve()
    parts = p.parts
    for i, part in enumerate(parts):
        if part in {"models", "runs"} and i > 0:
            root = Path(*parts[:i])
            if (root / "models").is_dir() or (root / "runs").is_dir():
                return WorkspaceLayout(str(root))
    try:
        from smartrain.core.runtime.workspace_paths import resolve_workspace_root

        return WorkspaceLayout(resolve_workspace_root(None))
    except ValueError:
        return None


def release_comment_for_run_dir(run_dir: str) -> str:
    layout = layout_for_run_dir(run_dir)
    if layout is None:
        return ""
    return get_comment_for_run_dir(layout, run_dir)


def upsert_entry(
    layout: WorkspaceLayout,
    *,
    entry_key: str,
    model_path: Path,
    comment: str = "",
    released_at: str | None = None,
) -> None:
    manifest = load_manifest(layout)
    entries = manifest.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        manifest["entries"] = entries
    root = Path(layout.root).resolve()
    rel_model = (
        str(model_path.resolve().relative_to(root))
        if model_path.resolve().is_relative_to(root)
        else str(model_path.resolve())
    )
    now = _now_iso()
    existing = entries.get(entry_key) if isinstance(entries.get(entry_key), dict) else {}
    entries[entry_key] = {
        "comment": str(comment or ""),
        "model_path": rel_model,
        "released_at": released_at or (existing.get("released_at") if isinstance(existing, dict) else None) or now,
        "comment_updated_at": now,
    }
    save_manifest(layout, manifest)


def set_comment(layout: WorkspaceLayout, entry_key: str, comment: str) -> None:
    manifest = load_manifest(layout)
    entries = manifest.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        manifest["entries"] = entries
    existing = entries.get(entry_key)
    if not isinstance(existing, dict):
        existing = {}
    existing["comment"] = str(comment or "")
    existing["comment_updated_at"] = _now_iso()
    entries[entry_key] = existing
    save_manifest(layout, manifest)


def remove_entry(layout: WorkspaceLayout, entry_key: str) -> bool:
    manifest = load_manifest(layout)
    entries = manifest.get("entries")
    if not isinstance(entries, dict) or entry_key not in entries:
        return False
    entries.pop(entry_key, None)
    save_manifest(layout, manifest)
    return True


def rename_entry(
    layout: WorkspaceLayout,
    old_key: str,
    new_key: str,
    *,
    model_path: Path,
) -> None:
    manifest = load_manifest(layout)
    entries = manifest.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        manifest["entries"] = entries
    old_entry = entries.pop(old_key, None)
    root = Path(layout.root).resolve()
    rel_model = (
        str(model_path.resolve().relative_to(root))
        if model_path.resolve().is_relative_to(root)
        else str(model_path.resolve())
    )
    now = _now_iso()
    if isinstance(old_entry, dict):
        new_entry = dict(old_entry)
    else:
        new_entry = {}
    new_entry["model_path"] = rel_model
    new_entry.setdefault("released_at", now)
    new_entry["comment_updated_at"] = now
    entries[new_key] = new_entry
    save_manifest(layout, manifest)


def sync_sidecar_comment(json_path: Path, comment: str) -> None:
    if not json_path.is_file():
        return
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not is_release_metadata(payload):
        return
    payload["comment"] = str(comment or "")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_release_entry_label(rel_path: str, comment: str, *, max_comment: int = 48) -> str:
    text = rel_path
    c = str(comment or "").strip()
    if c:
        if len(c) > max_comment:
            c = c[: max_comment - 1] + "…"
        text = f"{rel_path}  —  {c}"
    return text
