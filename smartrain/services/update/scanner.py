"""Scan workspace for legacy on-disk shapes and build an UpdatePlan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from smartrain.core.runtime.run_artifacts import preferred_run_model_path
from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.services.models.release_model_naming import (
    is_registry_bundle_path,
    load_release_metadata,
    release_json_path_for_pt,
)
from smartrain.services.models.release_models_manifest import (
    entry_key_for_pt,
    is_unified_release_bundle,
    load_manifest,
)
from smartrain.services.update.path_norm import needs_path_rewrite, to_posix_rel, workspace_rel_posix
from smartrain.services.update.plan import (
    UpdateCategory,
    UpdatePlan,
    UpdateRisk,
    UpdateStep,
)
from smartrain.workflows.analyze.results_analyzer import find_run_directories


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _rel(layout: WorkspaceLayout, path: Path) -> str:
    return workspace_rel_posix(layout, path)


def _iter_run_like_dirs(layout: WorkspaceLayout) -> list[Path]:
    found: list[Path] = []
    for base in (Path(layout.runs), Path(layout.models)):
        if not base.is_dir():
            continue
        for p in find_run_directories(str(base)):
            found.append(Path(p).resolve())
    models = Path(layout.models)
    if models.is_dir():
        for meta in models.rglob("training_metadata.json"):
            if is_registry_bundle_path(meta):
                continue
            root = meta.parent.resolve()
            if root not in found:
                found.append(root)
    return sorted(set(found), key=lambda p: str(p))


def _scan_layout(layout: WorkspaceLayout, root: Path, steps: list[UpdateStep]) -> None:
    rel = _rel(layout, root)
    if (root / "train").is_dir() and not (root / "train-ultralytics").is_dir():
        steps.append(
            UpdateStep(
                id=f"layout-train:{rel}",
                category=UpdateCategory.LAYOUT,
                risk=UpdateRisk.SAFE,
                title="Migrate train/ → train-ultralytics/",
                detail=rel,
                paths=[_rel(layout, root / "train")],
                action="migrate_train",
            )
        )
    elif (root / "train").is_dir() and (root / "train-ultralytics").is_dir():
        steps.append(
            UpdateStep(
                id=f"layout-train-merge:{rel}",
                category=UpdateCategory.LAYOUT,
                risk=UpdateRisk.ASK,
                title="Merge legacy train/ into existing train-ultralytics/",
                detail="Both directories exist; may overwrite or drop files",
                paths=[_rel(layout, root / "train"), _rel(layout, root / "train-ultralytics")],
                action="migrate_train",
            )
        )

    legacy_root_tests = [
        "test",
        "test-ultralytics",
        "test_onnx",
        "test_engine",
        "test_trt",
        "test_pt_uni",
    ]
    for name in legacy_root_tests:
        src = root / name
        if not src.exists():
            continue
        steps.append(
            UpdateStep(
                id=f"layout-test:{rel}:{name}",
                category=UpdateCategory.LAYOUT,
                risk=UpdateRisk.SAFE if not (root / "tests" / name.replace("test", "test-ultralytics", 1)).exists() else UpdateRisk.ASK,
                title=f"Relocate root {name} into tests/",
                detail=rel,
                paths=[_rel(layout, src)],
                action="migrate_tests",
            )
        )

    for pattern in ("test_metrics*.csv", "val_metrics*.csv", "confidence_recommendations_*.json"):
        for src in sorted(root.glob(pattern)):
            if not src.is_file():
                continue
            dst = root / "tests" / src.name
            risk = UpdateRisk.SAFE
            if dst.is_file() and _sha256_file(src) != _sha256_file(dst):
                risk = UpdateRisk.ASK
            steps.append(
                UpdateStep(
                    id=f"layout-root-file:{rel}:{src.name}",
                    category=UpdateCategory.LAYOUT,
                    risk=risk,
                    title=f"Move root {src.name} → tests/",
                    detail=rel,
                    paths=[_rel(layout, src)],
                    action="migrate_tests",
                )
            )

    for name in ("_runtime_data_train.yaml", "_runtime_data_test.yaml"):
        src = root / name
        if not src.is_file():
            continue
        dst = root / "tmp" / name
        if not dst.is_file():
            steps.append(
                UpdateStep(
                    id=f"layout-runtime-yaml:{rel}:{name}",
                    category=UpdateCategory.LAYOUT,
                    risk=UpdateRisk.SAFE,
                    title=f"Move {name} → tmp/",
                    detail=rel,
                    paths=[_rel(layout, src)],
                    action="migrate_layout",
                )
            )
        elif _sha256_file(src) == _sha256_file(dst):
            steps.append(
                UpdateStep(
                    id=f"layout-runtime-yaml-dup:{rel}:{name}",
                    category=UpdateCategory.LAYOUT,
                    risk=UpdateRisk.SAFE,
                    title=f"Remove duplicate root {name} (identical tmp/)",
                    detail=rel,
                    paths=[_rel(layout, src)],
                    action="migrate_layout",
                )
            )
        else:
            steps.append(
                UpdateStep(
                    id=f"layout-runtime-yaml-conflict:{rel}:{name}",
                    category=UpdateCategory.LAYOUT,
                    risk=UpdateRisk.ASK,
                    title=f"Remove root {name} conflicting with tmp/ (keep tmp)",
                    detail=rel,
                    paths=[_rel(layout, src), _rel(layout, dst)],
                    action="drop_root_runtime_yaml",
                )
            )


def _scan_weights(layout: WorkspaceLayout, root: Path, steps: list[UpdateStep]) -> None:
    if is_registry_bundle_path(root):
        return
    preferred = Path(preferred_run_model_path(str(root), ".pt"))
    if preferred.is_file():
        return
    candidates: list[Path] = []
    models = root / "models"
    if models.is_dir():
        candidates.extend(sorted(p for p in models.glob("*.pt") if p.is_file()))
    candidates.extend(sorted(p for p in root.glob("*.pt") if p.is_file()))
    for rel in (
        "train-ultralytics/weights/best.pt",
        "train-ultralytics/weights/last.pt",
        "train/weights/best.pt",
        "train/weights/last.pt",
    ):
        cand = root / rel
        if cand.is_file():
            candidates.append(cand)
    if not candidates:
        return
    resolved = candidates[0]
    risk = UpdateRisk.ASK if len(candidates) > 1 else UpdateRisk.SAFE
    steps.append(
        UpdateStep(
            id=f"weights-materialize:{_rel(layout, root)}",
            category=UpdateCategory.WEIGHTS,
            risk=risk,
            title="Materialize preferred models/<stem>.pt",
            detail=f"from {resolved.name}",
            paths=[_rel(layout, resolved), _rel(layout, preferred)],
            action="materialize_weights",
        )
    )


def _sidecar_needs_rewrite(meta: dict) -> bool:
    artifacts = meta.get("artifacts") or {}
    if isinstance(artifacts, dict):
        for key in ("model_path", "json_path", "release_dir", "train_copy_dir", "test_copy_dir"):
            val = artifacts.get(key)
            if isinstance(val, str) and needs_path_rewrite(val):
                return True
    source = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    if isinstance(source, dict):
        for key in ("source_run", "source_run_relative"):
            val = source.get(key)
            if isinstance(val, str) and needs_path_rewrite(val):
                return True
    return False


def _scan_release_bundle(layout: WorkspaceLayout, release_dir: Path, steps: list[UpdateStep]) -> None:
    if is_registry_bundle_path(release_dir):
        return
    if is_unified_release_bundle(release_dir):
        models_sub = release_dir / "models"
        for jp in sorted(models_sub.glob("*.json")):
            meta = load_release_metadata(jp)
            if not meta:
                continue
            if _sidecar_needs_rewrite(meta):
                steps.append(
                    UpdateStep(
                        id=f"manifest-relpaths:{_rel(layout, jp)}",
                        category=UpdateCategory.MANIFEST,
                        risk=UpdateRisk.SAFE,
                        title="Rewrite release sidecar paths to workspace-relative POSIX",
                        detail=_rel(layout, jp),
                        paths=[_rel(layout, jp)],
                        action="rewrite_sidecar_paths",
                    )
                )
        return

    root_pts = sorted(p for p in release_dir.glob("*.pt") if p.is_file())
    for pt in root_pts:
        if not load_release_metadata(release_json_path_for_pt(pt)):
            continue
        steps.append(
            UpdateStep(
                id=f"release-unify-root:{_rel(layout, pt)}",
                category=UpdateCategory.RELEASES,
                risk=UpdateRisk.SAFE,
                title="Unify root-level release PT/convert into models/",
                detail=_rel(layout, release_dir),
                paths=[_rel(layout, pt)],
                action="unify_root_release",
            )
        )

    sibling = release_dir.parent / f"{release_dir.name}.pt"
    if sibling.is_file() and load_release_metadata(release_json_path_for_pt(sibling)):
        steps.append(
            UpdateStep(
                id=f"release-unify-r2:{_rel(layout, sibling)}",
                category=UpdateCategory.RELEASES,
                risk=UpdateRisk.ASK,
                title="Unify R2 sibling .pt into release folder models/",
                detail=_rel(layout, release_dir),
                paths=[_rel(layout, sibling), _rel(layout, release_dir)],
                action="unify_r2_release",
            )
        )


def _scan_manifest_keys(layout: WorkspaceLayout, steps: list[UpdateStep]) -> None:
    manifest = load_manifest(layout)
    entries = manifest.get("entries") or {}
    if not isinstance(entries, dict):
        return
    models_root = Path(layout.models).resolve()
    for key, entry in list(entries.items()):
        if not isinstance(entry, dict):
            continue
        model_path = entry.get("model_path")
        if not isinstance(model_path, str) or not model_path.strip():
            continue
        mp = Path(to_posix_rel(model_path))
        if not mp.is_absolute() and not (len(model_path) > 2 and model_path[1] == ":"):
            mp = (Path(layout.root) / to_posix_rel(model_path)).resolve()
        else:
            try:
                mp = Path(model_path).resolve()
            except Exception:
                mp = Path(layout.root) / to_posix_rel(model_path)
        if not mp.is_file():
            steps.append(
                UpdateStep(
                    id=f"manifest-stale:{key}",
                    category=UpdateCategory.MANIFEST,
                    risk=UpdateRisk.ASK,
                    title="Stale releases_manifest entry (missing model file)",
                    detail=key,
                    paths=[model_path],
                    action="drop_manifest_entry",
                )
            )
            continue
        try:
            expected = entry_key_for_pt(mp)
        except Exception:
            continue
        if expected != key:
            parts = key.split("/", 1)
            if len(parts) == 2 and parts[1] != mp.stem:
                steps.append(
                    UpdateStep(
                        id=f"manifest-rekey:{key}",
                        category=UpdateCategory.MANIFEST,
                        risk=UpdateRisk.SAFE,
                        title=f"Rekey manifest entry {key} → {expected}",
                        detail=_rel(layout, mp),
                        paths=[key, expected],
                        action="rekey_manifest",
                    )
                )
        if needs_path_rewrite(model_path):
            try:
                under_ws = mp.is_relative_to(models_root) or mp.is_relative_to(Path(layout.root).resolve())
            except Exception:
                under_ws = True
            if under_ws or "\\" in model_path:
                steps.append(
                    UpdateStep(
                        id=f"manifest-abspath:{key}",
                        category=UpdateCategory.MANIFEST,
                        risk=UpdateRisk.SAFE,
                        title="Make releases_manifest model_path workspace-relative POSIX",
                        detail=key,
                        paths=[model_path],
                        action="relativize_manifest_path",
                    )
                )


def _runtime_yaml_needs_norm(payload: dict) -> bool:
    if "path" in payload:
        return True
    for k in ("train", "val", "test", "minival"):
        v = payload.get(k)
        if isinstance(v, str) and (needs_path_rewrite(v) or v.startswith("./")):
            return True
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str) and (needs_path_rewrite(item) or item.startswith("./")):
                    return True
    return False


def _scan_yaml(layout: WorkspaceLayout, steps: list[UpdateStep]) -> None:
    datasets = Path(layout.datasets)
    if datasets.is_dir():
        for data_yaml in sorted(datasets.rglob("data.yaml")):
            if data_yaml.parent.resolve() == datasets.resolve():
                continue
            try:
                import yaml

                payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
            except Exception:
                steps.append(
                    UpdateStep(
                        id=f"yaml-bad:{_rel(layout, data_yaml)}",
                        category=UpdateCategory.YAML,
                        risk=UpdateRisk.ASK,
                        title="Unreadable data.yaml",
                        detail=_rel(layout, data_yaml),
                        paths=[_rel(layout, data_yaml)],
                        action="normalize_yaml",
                    )
                )
                continue
            if not isinstance(payload, dict):
                continue
            needs = "path" in payload
            if not needs:
                for k in ("train", "val", "test"):
                    v = payload.get(k)
                    if isinstance(v, str) and (v.startswith("/") or v.startswith("./") or "\\" in v):
                        needs = True
                        break
            if needs:
                steps.append(
                    UpdateStep(
                        id=f"yaml-norm:{_rel(layout, data_yaml)}",
                        category=UpdateCategory.YAML,
                        risk=UpdateRisk.SAFE,
                        title="Normalize data.yaml (drop path, relative splits)",
                        detail=_rel(layout, data_yaml.parent),
                        paths=[_rel(layout, data_yaml)],
                        action="normalize_yaml",
                    )
                )

    # Runtime data yaml under runs/ and models/
    for base in (Path(layout.runs), Path(layout.models)):
        if not base.is_dir():
            continue
        for name in ("_runtime_data_train.yaml", "_runtime_data_test.yaml"):
            for yp in sorted(base.rglob(name)):
                if not yp.is_file():
                    continue
                try:
                    import yaml

                    payload = yaml.safe_load(yp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                if not _runtime_yaml_needs_norm(payload):
                    continue
                steps.append(
                    UpdateStep(
                        id=f"yaml-runtime:{_rel(layout, yp)}",
                        category=UpdateCategory.YAML,
                        risk=UpdateRisk.SAFE,
                        title="Normalize runtime _runtime_data_*.yaml (drop path, POSIX splits)",
                        detail=_rel(layout, yp),
                        paths=[_rel(layout, yp)],
                        action="normalize_runtime_yaml",
                    )
                )


def _scan_metadata(layout: WorkspaceLayout, root: Path, steps: list[UpdateStep]) -> None:
    meta = root / "training_metadata.json"
    if not meta.is_file():
        try:
            if root.resolve().is_relative_to(Path(layout.runs).resolve()):
                steps.append(
                    UpdateStep(
                        id=f"metadata-missing:{_rel(layout, root)}",
                        category=UpdateCategory.METADATA,
                        risk=UpdateRisk.SKIP,
                        title="Missing training_metadata.json (use migrate-models)",
                        detail=_rel(layout, root),
                        paths=[_rel(layout, root)],
                        action="skip",
                    )
                )
        except Exception:
            pass
        return
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        steps.append(
            UpdateStep(
                id=f"metadata-bad:{_rel(layout, meta)}",
                category=UpdateCategory.METADATA,
                risk=UpdateRisk.ASK,
                title="Unreadable training_metadata.json",
                detail=_rel(layout, meta),
                paths=[_rel(layout, meta)],
                action="skip",
            )
        )
        return
    if not isinstance(payload, dict):
        return
    paths = payload.get("paths") if isinstance(payload.get("paths"), dict) else {}
    best = str(paths.get("best_model") or "")
    if best and ("/" in best or "\\" in best or best.endswith("best.pt") or "weights" in best):
        steps.append(
            UpdateStep(
                id=f"metadata-best:{_rel(layout, root)}",
                category=UpdateCategory.METADATA,
                risk=UpdateRisk.SAFE,
                title="Normalize training_metadata model path references",
                detail=best,
                paths=[_rel(layout, meta)],
                action="normalize_metadata",
            )
        )


def scan_workspace(layout: WorkspaceLayout) -> UpdatePlan:
    steps: list[UpdateStep] = []
    for root in _iter_run_like_dirs(layout):
        _scan_layout(layout, root, steps)
        _scan_weights(layout, root, steps)
        _scan_metadata(layout, root, steps)
        try:
            if root.resolve().is_relative_to(Path(layout.models).resolve()):
                _scan_release_bundle(layout, root, steps)
        except Exception:
            pass

    _scan_manifest_keys(layout, steps)
    _scan_yaml(layout, steps)

    uniq: dict[str, UpdateStep] = {}
    for s in steps:
        uniq[s.id] = s
    return UpdatePlan(workspace_root=str(Path(layout.root).resolve()), steps=list(uniq.values()))


def residual_after(layout: WorkspaceLayout, categories: frozenset | None = None) -> UpdatePlan:
    plan = scan_workspace(layout)
    if categories is not None:
        plan = plan.filtered(categories)
    plan.residual = list(plan.steps)
    return plan
