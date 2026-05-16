#!/usr/bin/env python3
"""Backfill model_manifest.json task_type for promoted models without provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smartrain.adapters.canonical.read.model_adapter import _infer_task_from_model_name, _load_json
from smartrain.canonical.refs import canonical_target_from_model_dir

SUPPORTED_WEIGHT_EXTS = {".pt", ".onnx", ".engine", ".trt"}


def _resolve_weights_path(model_dir: Path) -> Path | None:
    try:
        target = canonical_target_from_model_dir(model_dir)
        if target.model_path.is_file():
            return target.model_path
    except Exception:
        pass
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_WEIGHT_EXTS:
            return path
    return None


def _has_task_provenance(model_dir: Path) -> bool:
    manifest = _load_json(model_dir / "model_manifest.json")
    if str(manifest.get("task_type") or "").strip():
        return True
    training_metadata = _load_json(model_dir / "training_metadata.json")
    training_info = training_metadata.get("training_info", {})
    if not isinstance(training_info, dict):
        return False
    if str(training_info.get("task_type") or "").strip():
        return True
    utrain = training_info.get("ultralytics_train")
    if isinstance(utrain, dict) and str(utrain.get("task") or "").strip():
        return True
    weights = _resolve_weights_path(model_dir)
    if weights is not None and _infer_task_from_model_name(weights.name):
        return True
    return False


def migrate_models_root(models_root: Path, *, dry_run: bool) -> tuple[int, int]:
    updated = 0
    skipped = 0
    if not models_root.is_dir():
        raise FileNotFoundError(f"Models root not found: {models_root}")

    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        if _has_task_provenance(model_dir):
            skipped += 1
            continue
        weights = _resolve_weights_path(model_dir)
        if weights is None:
            print(f"[SKIP] {model_dir}: no weights file")
            skipped += 1
            continue
        inferred = _infer_task_from_model_name(weights.name) or "detection"
        manifest_path = model_dir / "model_manifest.json"
        payload: dict = {}
        if manifest_path.is_file():
            raw = _load_json(manifest_path)
            if isinstance(raw, dict):
                payload = dict(raw)
        payload["task_type"] = inferred
        payload.setdefault("provenance_migration", "scripts/migrate_model_task_provenance.py")
        if dry_run:
            print(f"[DRY-RUN] {manifest_path}: task_type={inferred!r}")
        else:
            manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[OK] {manifest_path}: task_type={inferred!r}")
        updated += 1
    return updated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True, help="Workspace models/ directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files.")
    args = parser.parse_args()
    updated, skipped = migrate_models_root(args.models_root.expanduser().resolve(), dry_run=bool(args.dry_run))
    print(f"Done: updated={updated}, skipped={skipped}")


if __name__ == "__main__":
    main()
