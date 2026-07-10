"""Resolve Ultralytics PT test artifact paths across canonical and fallback directories."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from smartrain.core.runtime.run_artifacts import run_test_backend_dir
from smartrain.core.testing.ultralytics_test_contract import (
    ULTRALYTICS_TEST_COLLECT_IMAGE_NAMES,
    rich_files_required_for_format,
)

Provenance = Literal["test", "legacy", "train_val_fallback"]

PROVENANCE_TEST: Provenance = "test"
PROVENANCE_LEGACY: Provenance = "legacy"
PROVENANCE_TRAIN_VAL: Provenance = "train_val_fallback"

_CSV_NAMES = ("pr.csv", "pr_per_class.csv", "args.yaml")
_IMAGE_NAMES = ULTRALYTICS_TEST_COLLECT_IMAGE_NAMES


@dataclass(frozen=True)
class ArtifactSourceDir:
    path: str
    provenance: Provenance


@dataclass
class ResolvedUltralyticsArtifacts:
    root_dir: str
    source_dirs: list[ArtifactSourceDir] = field(default_factory=list)
    resolved: dict[str, tuple[str, Provenance]] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    completeness: str = "missing"

    @property
    def primary_test_dir(self) -> str | None:
        for src in self.source_dirs:
            if src.provenance == PROVENANCE_TEST and os.path.isdir(src.path):
                return src.path
        return None


def _sorted_train_val_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for pattern in ("train-ultralytics", "train-ultralytics*"):
        for p in sorted(root.glob(pattern)):
            if p.is_dir():
                out.append(p.resolve())
    seen: set[str] = set()
    unique: list[Path] = []
    for p in out:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def iter_ultralytics_artifact_source_dirs(root_dir: str) -> list[ArtifactSourceDir]:
    """Search roots in priority order (test canonical first, then legacy, then train-val)."""
    root = Path(root_dir).expanduser().resolve()
    out: list[ArtifactSourceDir] = []
    seen: set[str] = set()

    def _add(path: Path, provenance: Provenance) -> None:
        if not path.is_dir():
            return
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        out.append(ArtifactSourceDir(path=key, provenance=provenance))

    try:
        canonical = Path(run_test_backend_dir(str(root), "ultralytics")).resolve()
        _add(canonical, PROVENANCE_TEST)
    except Exception:
        pass

    for rel, prov in (
        ("test", PROVENANCE_LEGACY),
        ("test-ultralytics", PROVENANCE_LEGACY),
    ):
        _add(root / rel, prov)

    for train_dir in _sorted_train_val_dirs(root):
        _add(train_dir, PROVENANCE_TRAIN_VAL)

    return out


def resolve_ultralytics_artifacts(root_dir: str, *, format_name: str = "pt") -> ResolvedUltralyticsArtifacts:
    root = os.path.abspath(os.path.expanduser(root_dir))
    source_dirs = iter_ultralytics_artifact_source_dirs(root)
    names = list(_CSV_NAMES) + list(_IMAGE_NAMES)
    resolved: dict[str, tuple[str, Provenance]] = {}

    for name in names:
        for src in source_dirs:
            candidate = os.path.join(src.path, name)
            if os.path.isfile(candidate):
                resolved[name] = (candidate, src.provenance)
                break

    required = rich_files_required_for_format(root, format_name) or ()
    missing_required = [n for n in required if n not in resolved]

    has_test_rich = all(
        n in resolved and resolved[n][1] == PROVENANCE_TEST for n in required if n.endswith((".png", ".yaml"))
    )
    has_any_csv = "pr.csv" in resolved and "pr_per_class.csv" in resolved
    has_train_val_images = any(
        resolved.get(n, (None, None))[1] == PROVENANCE_TRAIN_VAL
        for n in _IMAGE_NAMES
        if n in resolved
    )

    if has_test_rich and not missing_required:
        completeness = "complete"
    elif has_any_csv and not has_train_val_images and missing_required:
        completeness = "partial_csv_only"
    elif resolved and has_train_val_images and not has_test_rich:
        completeness = "train_val_fallback"
    elif resolved:
        completeness = "partial_csv_only" if has_any_csv else "train_val_fallback"
    else:
        completeness = "missing"

    return ResolvedUltralyticsArtifacts(
        root_dir=root,
        source_dirs=source_dirs,
        resolved=resolved,
        missing_required=missing_required,
        completeness=completeness,
    )
