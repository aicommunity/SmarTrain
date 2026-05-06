"""
Copy a training run directory tree into a destination bundle (promoted model dir, release dir, etc.).

Preserves relative layout under the run root. Intentionally excludes ``train/weights`` (heavy checkpoints;
canonical weights live under ``run/models/``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def copy_run_bundle(
    run_dir: Path,
    dest_dir: Path,
    *,
    include_tests: bool = False,
    copy_run_models: bool = False,
) -> None:
    """
    Copy reproducibility artifacts from ``run_dir`` into ``dest_dir``.

    Always copies (when present): ``train/`` (excluding ``weights``), ``test/``,
    ``training_metadata.json``, ``test_metrics*.csv`` in the run root,
    ``_runtime_data_*.yaml`` in the run root; if missing there, tries ``run_dir/tmp/``.

    Optional:
    - ``include_tests``: copy ``tests/`` subtree (canonical test pipeline layout).
    - ``copy_run_models``: copy ``run_dir/models/`` → ``dest_dir/models/`` (merge).
    """
    run_dir = run_dir.resolve()
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _copy_train_variant(src: Path, dst_name: str) -> None:
        if not src.is_dir():
            return
        shutil.copytree(
            str(src),
            str(dest_dir / dst_name),
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("weights"),
        )

    # Legacy ``train/`` and provider dirs such as ``train-ultralytics/`` (after ``ensure_run_layout``).
    _copy_train_variant(run_dir / "train", "train")
    for child in sorted(run_dir.glob("train-*")):
        if child.is_dir():
            _copy_train_variant(child, child.name)

    src_legacy_test = run_dir / "test"
    if src_legacy_test.is_dir():
        shutil.copytree(
            str(src_legacy_test),
            str(dest_dir / "test"),
            dirs_exist_ok=True,
        )

    if include_tests:
        src_tests = run_dir / "tests"
        if src_tests.is_dir():
            shutil.copytree(str(src_tests), str(dest_dir / "tests"), dirs_exist_ok=True)

    if copy_run_models:
        src_models = run_dir / "models"
        if src_models.is_dir():
            shutil.copytree(str(src_models), str(dest_dir / "models"), dirs_exist_ok=True)

    for name in ("training_metadata.json", "_runtime_data_test.yaml", "_runtime_data_train.yaml"):
        src = run_dir / name
        if src.is_file():
            shutil.copy2(str(src), str(dest_dir / name))

    for src in sorted(run_dir.glob("test_metrics*.csv")):
        if src.is_file():
            shutil.copy2(str(src), str(dest_dir / src.name))

    tmp_dir = run_dir / "tmp"
    if tmp_dir.is_dir():
        for name in ("_runtime_data_test.yaml", "_runtime_data_train.yaml"):
            if not (dest_dir / name).is_file():
                src = tmp_dir / name
                if src.is_file():
                    shutil.copy2(str(src), str(dest_dir / name))


def normalize_training_metadata_paths_for_bundle(metadata_path: Path, weights_relative_posix: str) -> bool:
    """
    Point ``paths.best_model`` and ``source.source_weights`` at the bundle-relative weights path
    (e.g. ``models/my_run.pt``).
    """
    if not metadata_path.is_file():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    changed = False
    paths = payload.get("paths")
    if isinstance(paths, dict):
        if paths.get("best_model") != weights_relative_posix:
            paths["best_model"] = weights_relative_posix
            changed = True
    source = payload.get("source")
    if isinstance(source, dict):
        if source.get("source_weights") != weights_relative_posix:
            source["source_weights"] = weights_relative_posix
            changed = True
    if not changed:
        return False
    tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(metadata_path)
    return True
