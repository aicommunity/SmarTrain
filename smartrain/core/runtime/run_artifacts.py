from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

LEGACY_RUN_ROOT_VAL_RECS_PREFIX = "val-recs-"

_PROTECTED_RUN_ROOT_DIR_NAMES = frozenset({"models", "tmp", ".smartrain", ".smartrain_cache"})


def reject_documentation_placeholder_path(path: str | Path, *, kind: str = "path") -> None:
    """Reject literal ``...`` used as a docs/CLI placeholder (not a filesystem parent dir)."""
    raw = str(path or "").strip()
    if not raw:
        return
    if raw == "..." or "..." in raw.split("/") or "..." in raw.split("\\"):
        raise ValueError(
            f"Invalid {kind}: {raw!r} contains '...' as a path segment. "
            "Replace with the full real directory path (see docs), not an ellipsis placeholder."
        )
    if Path(raw).expanduser().name == "...":
        raise ValueError(
            f"Invalid {kind}: {raw!r} resolves to a directory named '...'. "
            "Use the full run or workspace path."
        )


def _normalize_run_root(run_dir: str | Path) -> Path:
    reject_documentation_placeholder_path(run_dir, kind="run_dir")
    root = Path(run_dir).expanduser().resolve()
    # Defensive normalization: callers may accidentally pass runs/<run>/models
    # or runs/<run>/tmp/tests/train-* instead of runs/<run>.
    # Collapse such paths to run root.
    while root.name in {"models", "tmp", "tests"} or root.name.startswith("train-") or root.name.startswith("test-"):
        parent = root.parent
        if parent == root:
            break
        if (
            (parent / "training_metadata.json").is_file()
            or (parent / "train").is_dir()
            or (parent / "models").is_dir()
            or (parent / "tests").is_dir()
        ):
            root = parent
            continue
        break
    return root


def run_models_dir(run_dir: str) -> Path:
    root = _normalize_run_root(run_dir)
    return root / "models"


def run_tmp_dir(run_dir: str) -> Path:
    root = _normalize_run_root(run_dir)
    return root / "tmp"


def ensure_runtime_tmp_dir(run_dir: str) -> Path:
    """Ensure a writable tmp dir for runtime data.yaml without polluting release bundles.

    Training runs get full ``ensure_run_layout``. Release bundles only get ``tmp/``
    (no empty ``tests/`` / layout migration).
    """
    root = _normalize_run_root(run_dir)
    if _looks_like_release_bundle_dir(root):
        tmp = root / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp
    ensure_run_layout(str(root))
    return run_tmp_dir(str(root))


def ensure_runtime_layout_for_yaml(run_dir: str) -> tuple[Path, Path]:
    """Callbacks for ``build_runtime_data_yaml``: release-safe layout + tmp path."""
    root = _normalize_run_root(run_dir)
    if _looks_like_release_bundle_dir(root):
        tmp = ensure_runtime_tmp_dir(str(root))
        models = root / "models"
        return models, tmp
    return ensure_run_layout(str(root))


def run_tests_dir(run_dir: str) -> Path:
    root = _normalize_run_root(run_dir)
    return root / "tests"


def run_train_backend_dir(run_dir: str, backend: str = "ultralytics") -> Path:
    root = _normalize_run_root(run_dir)
    safe_backend = str(backend or "ultralytics").strip().lower().replace("/", "-").replace("\\", "-")
    return root / f"train-{safe_backend}"


def run_test_backend_dir(run_dir: str, backend: str = "ultralytics") -> Path:
    safe_backend = str(backend or "ultralytics").strip().lower().replace("/", "-").replace("\\", "-")
    return run_tests_dir(run_dir) / f"test-{safe_backend}"


def run_test_format_dir(run_dir: str, format_name: str) -> Path:
    safe_fmt = str(format_name or "pt").strip().lower().replace("/", "-").replace("\\", "-")
    return run_tests_dir(run_dir) / f"test_{safe_fmt}"


def subtree_has_files(path: Path) -> bool:
    """True if path is a file or a directory tree containing at least one file."""
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    for _root, _dirs, files in os.walk(path, followlinks=False):
        if files:
            return True
    return False


def _is_ultralytics_suffix_dir(name: str, canonical_name: str) -> bool:
    if name == canonical_name:
        return True
    if not name.startswith(canonical_name):
        return False
    tail = name[len(canonical_name) :]
    if not tail:
        return False
    return bool(re.fullmatch(r"-?\d+", tail))


def _score_ultralytics_artifact_dir(path: Path, *, kind: str) -> tuple[int, float]:
    priority = 0
    if kind == "train":
        if (path / "results.csv").is_file():
            priority = 100
        elif (path / "weights" / "best.pt").is_file():
            priority = 90
        elif (path / "weights" / "last.pt").is_file():
            priority = 85
        elif (path / "args.yaml").is_file():
            priority = 70
    elif kind == "test":
        if (path / "pr.csv").is_file():
            priority = 80
        elif (path / "args.yaml").is_file():
            priority = 70
        elif (path / "BoxPR_curve.png").is_file():
            priority = 65
    if priority == 0 and subtree_has_files(path):
        priority = 10
    try:
        mtime = path.stat().st_mtime if path.exists() else 0.0
    except OSError:
        mtime = 0.0
    return priority, mtime


def _consolidate_ultralytics_named_dir(parent: Path, canonical_name: str, *, kind: str) -> None:
    if not parent.is_dir():
        return
    candidates = [
        p for p in parent.iterdir() if p.is_dir() and _is_ultralytics_suffix_dir(p.name, canonical_name)
    ]
    if not candidates:
        return
    canonical = parent / canonical_name
    canonical.mkdir(parents=True, exist_ok=True)
    for src in sorted(candidates, key=lambda p: _score_ultralytics_artifact_dir(p, kind=kind), reverse=True):
        if src.resolve() == canonical.resolve():
            continue
        _move_tree_merge_to_canonical(src, canonical)
    for entry in list(parent.iterdir()):
        if not entry.is_dir():
            continue
        if not _is_ultralytics_suffix_dir(entry.name, canonical_name):
            continue
        if entry.resolve() == canonical.resolve():
            continue
        if not subtree_has_files(entry):
            shutil.rmtree(entry, ignore_errors=True)


def consolidate_train_backend_dir(run_dir: str, backend: str = "ultralytics") -> None:
    root = _normalize_run_root(run_dir)
    safe = str(backend or "ultralytics").strip().lower().replace("/", "-").replace("\\", "-")
    _consolidate_ultralytics_named_dir(root, f"train-{safe}", kind="train")


def consolidate_test_backend_dir(run_dir: str, backend: str = "ultralytics") -> None:
    tests = run_tests_dir(run_dir)
    safe = str(backend or "ultralytics").strip().lower().replace("/", "-").replace("\\", "-")
    _consolidate_ultralytics_named_dir(tests, f"test-{safe}", kind="test")


def consolidate_val_recs_dirs(run_dir: str) -> None:
    tests = run_tests_dir(run_dir)
    if not tests.is_dir():
        return
    for entry in list(tests.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name != "val-ultralytics-recs" and not entry.name.startswith("val-recs-"):
            continue
        _consolidate_ultralytics_named_dir(tests, entry.name, kind="val")
        if entry.is_dir() and not subtree_has_files(entry):
            shutil.rmtree(entry, ignore_errors=True)


def relocate_or_remove_legacy_val_recs_at_run_root(run_dir: str) -> None:
    root = _normalize_run_root(run_dir)
    tests = run_tests_dir(str(root))
    tests.mkdir(parents=True, exist_ok=True)
    for entry in list(root.iterdir()):
        if not entry.is_dir():
            continue
        if not entry.name.startswith(LEGACY_RUN_ROOT_VAL_RECS_PREFIX):
            continue
        dest = tests / entry.name
        if not subtree_has_files(entry):
            shutil.rmtree(entry, ignore_errors=True)
            continue
        _move_tree_merge_to_canonical(entry, dest)


def prune_empty_subdirs(run_dir: str) -> None:
    root = _normalize_run_root(run_dir)
    if not root.is_dir():
        return

    def _prune_children(container: Path) -> None:
        if not container.is_dir():
            return
        for child in list(container.iterdir()):
            if not child.is_dir():
                continue
            if child.name in _PROTECTED_RUN_ROOT_DIR_NAMES:
                continue
            if not subtree_has_files(child):
                shutil.rmtree(child, ignore_errors=True)

    _prune_children(root)
    tests = root / "tests"
    if tests.is_dir():
        _prune_children(tests)


def normalize_ultralytics_run_layout(run_dir: str) -> None:
    """Idempotent post-process: merge Ultralytics suffix dirs, drop empty legacy shells."""
    root = _normalize_run_root(run_dir)
    if not root.is_dir():
        return
    consolidate_train_backend_dir(str(root), backend="ultralytics")
    consolidate_test_backend_dir(str(root), backend="ultralytics")
    consolidate_val_recs_dirs(str(root))
    relocate_or_remove_legacy_val_recs_at_run_root(str(root))
    prune_empty_subdirs(str(root))


def remove_empty_train_ultralytics_dir(model_dir: str, backend: str = "ultralytics") -> None:
    p = run_train_backend_dir(model_dir, backend)
    if p.is_dir() and not subtree_has_files(p):
        shutil.rmtree(p, ignore_errors=True)


def _migrate_legacy_run_layout(root: Path) -> None:
    # Lazy migration with cleanup: move legacy artifacts into canonical
    # layout and remove legacy folders/files after successful relocation.
    run_tests_dir(str(root)).mkdir(parents=True, exist_ok=True)
    _migrate_legacy_train_artifacts(root)
    _migrate_legacy_test_artifacts(root)


def _move_file_to_canonical(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            # Canonical copy already exists; drop legacy duplicate.
            src.unlink(missing_ok=True)
            return
        src.replace(dst)
    except Exception:
        return


def _move_tree_merge_to_canonical(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    for item in sorted(src.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            try:
                item.rmdir()
            except OSError:
                pass
            continue
        _move_file_to_canonical(item, target)
    try:
        src.rmdir()
    except OSError:
        pass


def _migrate_legacy_train_artifacts(root: Path) -> None:
    legacy = root / "train"
    canonical = run_train_backend_dir(str(root), "ultralytics")
    _move_tree_merge_to_canonical(legacy, canonical)


def _migrate_legacy_test_artifacts(root: Path) -> None:
    tests_root = run_tests_dir(str(root))
    tests_root.mkdir(parents=True, exist_ok=True)

    legacy_to_new_dirs = {
        "test": "test-ultralytics",
        "test_onnx": "test_onnx",
        "test_engine": "test_engine",
        "test_trt": "test_trt",
        "test_pt_uni": "test_pt_uni",
    }
    for legacy_name, new_name in legacy_to_new_dirs.items():
        _move_tree_merge_to_canonical(root / legacy_name, tests_root / new_name)

    root_ultra = root / "test-ultralytics"
    canonical_ultra = tests_root / "test-ultralytics"
    if root_ultra.is_dir():
        _move_tree_merge_to_canonical(root_ultra, canonical_ultra)

    legacy_patterns = (
        "test_metrics*.csv",
        "val_metrics*.csv",
        "confidence_recommendations_*.json",
        "test_artifacts_manifest.json",
    )
    for pattern in legacy_patterns:
        for src in root.glob(pattern):
            if src.is_file():
                _move_file_to_canonical(src, tests_root / src.name)


def ensure_run_layout(run_dir: str) -> tuple[Path, Path]:
    root = _normalize_run_root(run_dir)
    models = run_models_dir(str(root))
    tmp = run_tmp_dir(str(root))
    tests = run_tests_dir(str(root))
    models.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_run_layout(root)
    normalize_ultralytics_run_layout(str(root))
    for runtime_name in ("_runtime_data_train.yaml", "_runtime_data_test.yaml"):
        src = root / runtime_name
        dst = tmp / runtime_name
        if src.is_file() and not dst.exists():
            try:
                src.replace(dst)
            except Exception:
                pass
    return models, tmp


def preferred_run_model_path(run_dir: str, ext: str = ".pt") -> str:
    """Return canonical weights path under ``models/`` (does not create directories)."""
    root = _normalize_run_root(run_dir)
    models = root / "models"
    suffix = ext if str(ext).startswith(".") else f".{ext}"
    stem = resolve_run_weights_stem(str(root))
    return str(models / f"{stem}{suffix}")


def resolve_run_weights_stem(run_dir: str) -> str:
    """Canonical weight basename stem for a run (independent of folder name when possible)."""
    root = _normalize_run_root(run_dir)
    meta_path = root / "training_metadata.json"
    payload: dict[str, Any] | None = None
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = None

    if payload is not None:
        paths = payload.get("paths")
        if isinstance(paths, dict):
            best = paths.get("best_model")
            if isinstance(best, str) and best.strip():
                raw = best.strip().replace("\\", "/")
                # Ignore legacy relative paths (train/weights/best.pt); only trust top-level basenames.
                if "/" not in raw and not _looks_like_legacy_model_reference(raw, ".pt"):
                    name = Path(raw).name
                    if name.lower().endswith(".pt"):
                        return name[:-3]
                    return Path(name).stem

        from smartrain.services.models.release_model_naming import build_model_weights_stem_from_metadata

        computed = build_model_weights_stem_from_metadata(payload)
        if computed:
            return computed

    models = root / "models"
    legacy = models / f"{root.name}.pt"
    if legacy.is_file():
        return root.name
    if models.is_dir():
        pts = sorted(p for p in models.glob("*.pt") if p.is_file())
        if len(pts) == 1:
            return pts[0].stem
        detect_like = [
            p
            for p in pts
            if p.stem.startswith(
                ("detect_", "segment_", "segmentation_", "classify_", "classification_")
            )
            or "_epochs_b" in p.stem
        ]
        if len(detect_like) == 1:
            return detect_like[0].stem

    # Legacy: weight basename matched the run folder name.
    return root.name


_TASK_WEIGHT_STEM_PREFIXES = (
    "detect_",
    "segment_",
    "segmentation_",
    "classify_",
    "classification_",
)


def _task_like_weight_paths(candidates: list[Path]) -> list[Path]:
    return [
        p
        for p in candidates
        if p.stem.startswith(_TASK_WEIGHT_STEM_PREFIXES) or "_epochs_b" in p.stem
    ]


def _looks_like_release_bundle_dir(root: Path) -> bool:
    """True when ``root`` looks like a published release bundle (do not mkdir run layout)."""
    for json_path in root.glob("*.json"):
        if json_path.name in {
            "training_metadata.json",
            "model_manifest.json",
            "test_artifacts_manifest.json",
            "releases_manifest.json",
        }:
            continue
        if (root / f"{json_path.stem}.pt").is_file():
            return True
    for prefix in ("detect_", "segment_", "classify_"):
        if any(root.glob(f"{prefix}*.pt")):
            return True
    models = root / "models"
    if models.is_dir():
        for pt in sorted(models.glob("*.pt")):
            sidecar = models / f"{pt.stem}.json"
            if sidecar.is_file():
                try:
                    payload = json.loads(sidecar.read_text(encoding="utf-8"))
                except Exception:
                    payload = None
                if isinstance(payload, dict) and payload.get("source") and payload.get("artifacts"):
                    return True
    if models.is_dir() and root.name.startswith(("detect_", "segment_", "classify_")):
        nested = models / f"{root.name}.pt"
        if nested.is_file():
            return True
    return False


def _resolve_run_model_existing(root: Path, suffix: str) -> Path | None:
    """Locate existing weight files without mutating the directory tree."""
    models = root / "models"
    stem = resolve_run_weights_stem(str(root))
    preferred = models / f"{stem}{suffix}"
    if preferred.is_file():
        return preferred
    legacy = models / f"{root.name}{suffix}"
    if legacy.is_file():
        return legacy
    if models.is_dir():
        candidates = sorted(p for p in models.glob(f"*{suffix}") if p.is_file())
        if len(candidates) == 1:
            return candidates[0]
        task_like = _task_like_weight_paths(candidates)
        if len(task_like) == 1:
            return task_like[0]
    root_named = root / f"{root.name}{suffix}"
    if root_named.is_file():
        return root_named
    # R3 / nested release: ``models/<ds>/<run_id>/detect_*.pt`` (folder ≠ stem).
    root_pts = sorted(p for p in root.glob(f"*{suffix}") if p.is_file())
    if len(root_pts) == 1:
        return root_pts[0]
    task_like = _task_like_weight_paths(root_pts)
    if len(task_like) == 1:
        return task_like[0]
    for pt in root_pts:
        if (root / f"{pt.stem}.json").is_file():
            return pt
    # R2: sibling ``models/<ds>/<stem>.pt`` next to ``models/<ds>/<stem>/``.
    sibling = root.parent / f"{root.name}{suffix}"
    if sibling.is_file():
        return sibling
    for rel in (
        f"train-ultralytics/weights/best{suffix}",
        f"train-ultralytics/weights/last{suffix}",
        f"train-ultralytics/best{suffix}",
        f"train/weights/best{suffix}",
        f"train/weights/last{suffix}",
        f"train/best{suffix}",
    ):
        cand = root / rel
        if cand.is_file():
            return cand
    return None


def resolve_run_model(run_dir: str, ext: str = ".pt") -> Path | None:
    """Resolve weights under run or release layouts (additive; preserves legacy paths)."""
    root = _normalize_run_root(run_dir)
    suffix = ext if str(ext).startswith(".") else f".{ext}"
    found = _resolve_run_model_existing(root, suffix)
    if found is not None:
        return found
    if _looks_like_release_bundle_dir(root):
        return None
    # Training runs may still need layout migration (legacy train/ → models/).
    ensure_run_layout(str(root))
    return _resolve_run_model_existing(root, suffix)


def _looks_like_legacy_model_reference(value: str, ext: str) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    lowered = raw.replace("\\", "/").lower()
    suffix = ext if str(ext).startswith(".") else f".{ext}"
    suffix_l = suffix.lower()
    legacy_tail = f"train/weights/best{suffix_l}"
    return lowered.endswith(legacy_tail) or lowered.endswith(f"/best{suffix_l}") or lowered == f"best{suffix_l}"


def normalize_model_references_in_metadata(metadata_path: Path, run_dir: str, ext: str = ".pt") -> bool:
    if not metadata_path.is_file():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    suffix = ext if str(ext).startswith(".") else f".{ext}"
    canonical_name = Path(preferred_run_model_path(run_dir, suffix)).name
    changed = False

    paths = payload.get("paths")
    if isinstance(paths, dict):
        best_model = paths.get("best_model")
        if isinstance(best_model, str) and _looks_like_legacy_model_reference(best_model, suffix):
            paths["best_model"] = canonical_name
            changed = True

    source = payload.get("source")
    if isinstance(source, dict):
        src_weights = source.get("source_weights")
        if isinstance(src_weights, str) and _looks_like_legacy_model_reference(src_weights, suffix):
            source["source_weights"] = canonical_name
            changed = True

    if not changed:
        return False
    tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(metadata_path)
    return True


def materialize_preferred_run_model(
    run_dir: str,
    *,
    ext: str = ".pt",
    source_path: str | None = None,
    move: bool = True,
    normalize_metadata: bool = True,
) -> Path | None:
    root = _normalize_run_root(run_dir)
    if not root.is_dir():
        return None
    ensure_run_layout(str(root))
    canonical = Path(preferred_run_model_path(run_dir, ext))
    canonical.parent.mkdir(parents=True, exist_ok=True)
    if canonical.is_file():
        if normalize_metadata:
            normalize_model_references_in_metadata(root / "training_metadata.json", run_dir, ext)
        return canonical

    source: Path | None = None
    if source_path:
        cand = Path(source_path).expanduser().resolve()
        if cand.is_file():
            source = cand
    if source is None:
        source = resolve_run_model(run_dir, ext)
    if source is None:
        return None
    if source.resolve() == canonical.resolve():
        return canonical
    if move:
        source.rename(canonical)
    else:
        canonical.write_bytes(source.read_bytes())
    if normalize_metadata:
        normalize_model_references_in_metadata(root / "training_metadata.json", run_dir, ext)
    return canonical


def model_sidecar_metadata_path(model_path: str | Path) -> Path:
    p = Path(model_path)
    return p.with_name(f"{p.name}.meta.json")


def is_internal_conversion_artifact(model_path: str | Path) -> bool:
    p = Path(model_path)
    name = p.name.lower()
    # Internal ONNX cache used only as an intermediate for TRT conversion.
    return p.suffix.lower() == ".onnx" and ("_nms0_trtprep" in name or "_nms1_trtprep" in name)


def _fingerprint_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def write_model_sidecar_metadata(
    model_path: str | Path,
    *,
    format_name: str,
    run_dir: str | None = None,
    source_path: str | None = None,
    tool: str | None = None,
    params: dict[str, Any] | None = None,
    status: str = "ok",
    error: str | None = None,
) -> Path:
    p = Path(model_path).expanduser().resolve()
    sidecar = model_sidecar_metadata_path(p)
    payload: dict[str, Any] = {
        "format": str(format_name),
        "path": str(p),
        "filename": p.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_path": source_path,
        "tool": tool,
        "params": params or {},
        "status": status,
        "error": error,
        "fingerprint_sha256": _fingerprint_file(p) if p.is_file() else None,
        "size_bytes": p.stat().st_size if p.is_file() else None,
        "run_path": str(Path(run_dir).expanduser().resolve()) if run_dir else None,
    }
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(sidecar)
    return sidecar


def read_model_sidecar_metadata(model_path: str | Path) -> dict[str, Any] | None:
    sidecar = model_sidecar_metadata_path(model_path)
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def scan_run_models(run_dir: str) -> list[dict[str, Any]]:
    root = _normalize_run_root(run_dir)
    models, _tmp = ensure_run_layout(str(root))
    out: list[dict[str, Any]] = []
    exts = {".pt", ".onnx", ".engine", ".trt"}
    for p in sorted(models.glob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        meta = read_model_sidecar_metadata(p) or {}
        fmt = str(meta.get("format") or p.suffix.lower().lstrip(".")).lower()
        if fmt == "tensorrt-engine":
            fmt = "engine"
        if fmt == "tensorrt-trt":
            fmt = "trt"
        out.append(
            {
                "format": fmt,
                "path": str(p),
                "name": p.name,
                "metadata": meta,
            }
        )
    return out

