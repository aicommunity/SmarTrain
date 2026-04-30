from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _normalize_run_root(run_dir: str | Path) -> Path:
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


def _migrate_legacy_run_layout(root: Path) -> None:
    # Lazy migration with cleanup: move legacy artifacts into canonical
    # layout and remove legacy folders/files after successful relocation.
    run_tests_dir(str(root)).mkdir(parents=True, exist_ok=True)
    run_train_backend_dir(str(root), "ultralytics").mkdir(parents=True, exist_ok=True)
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
    for runtime_name in ("_runtime_data_train.yaml", "_runtime_data_test.yaml"):
        src = root / runtime_name
        dst = tmp / runtime_name
        if src.is_file() and not dst.exists():
            try:
                src.replace(dst)
            except Exception:
                pass
    return models, tmp


def canonical_run_model_path(run_dir: str, ext: str = ".pt") -> str:
    root = _normalize_run_root(run_dir)
    models, _tmp = ensure_run_layout(str(root))
    suffix = ext if str(ext).startswith(".") else f".{ext}"
    return str(models / f"{root.name}{suffix}")


def _legacy_run_model_candidates(run_dir: str, ext: str = ".pt") -> list[Path]:
    root = _normalize_run_root(run_dir)
    suffix = ext if str(ext).startswith(".") else f".{ext}"
    suffix_l = suffix.lower()
    candidates = [
        root / "train-ultralytics" / "weights" / f"best{suffix}",
        root / "train-ultralytics" / f"best{suffix}",
        root / "train" / "weights" / f"best{suffix}",
        root / "weights" / f"best{suffix}",
        root / "train" / f"best{suffix}",
        root / f"best{suffix}",
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for cand in candidates:
        rc = cand.resolve()
        if rc in seen:
            continue
        seen.add(rc)
        if cand.is_file():
            out.append(cand)
    for cand in root.rglob(f"best{suffix}"):
        if not cand.is_file():
            continue
        if cand.suffix.lower() != suffix_l:
            continue
        rc = cand.resolve()
        if rc in seen:
            continue
        seen.add(rc)
        out.append(cand)
    return out


def resolve_run_model_with_legacy_fallback(run_dir: str, ext: str = ".pt") -> Path | None:
    canonical = Path(canonical_run_model_path(run_dir, ext))
    if canonical.is_file():
        return canonical
    root = _normalize_run_root(run_dir)
    suffix = ext if str(ext).startswith(".") else f".{ext}"
    legacy_canonical = root / f"{root.name}{suffix}"
    if legacy_canonical.is_file():
        return legacy_canonical
    for cand in _legacy_run_model_candidates(run_dir, ext):
        if cand.is_file():
            return cand
    return None


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
    canonical_name = Path(canonical_run_model_path(run_dir, suffix)).name
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


def materialize_canonical_run_model(
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
    canonical = Path(canonical_run_model_path(run_dir, ext))
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
        source = resolve_run_model_with_legacy_fallback(run_dir, ext)
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

