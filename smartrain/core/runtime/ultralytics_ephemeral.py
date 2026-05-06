"""Ultralytics YOLO: temp project dirs and cleanup of default runs/detect junk."""
from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def ephemeral_ultralytics_project(prefix: str = "smartrain_ultra_") -> Iterator[str]:
    d = tempfile.mkdtemp(prefix=prefix)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def ultralytics_sidecar_dir(base: str, *parts: str) -> str:
    p = os.path.abspath(os.path.join(os.path.expanduser(base), *parts))
    os.makedirs(p, exist_ok=True)
    return p


def _subtree_has_any_file(dir_path: str) -> bool:
    for _root, _dirs, files in os.walk(dir_path, followlinks=False):
        if files:
            return True
    return False


def prune_ultralytics_default_detect_under_runs(runs_root: str) -> None:
    """
    Remove ``runs/detect/<child>`` trees that contain no files anywhere (only empty dirs).
    If ``runs/detect`` becomes empty, remove it. Never removes ``runs`` itself.
    """
    detect_root = os.path.join(os.path.abspath(os.path.expanduser(runs_root)), "detect")
    if not os.path.isdir(detect_root):
        return
    try:
        for name in list(os.listdir(detect_root)):
            sub = os.path.join(detect_root, name)
            if os.path.isfile(sub):
                continue
            if os.path.isdir(sub) and not _subtree_has_any_file(sub):
                shutil.rmtree(sub, ignore_errors=True)
        if not os.listdir(detect_root):
            os.rmdir(detect_root)
    except OSError:
        pass


def best_effort_prune_workspace_runs_detect(workspace_root: str) -> None:
    from smartrain.workspace_paths import WorkspaceLayout

    prune_ultralytics_default_detect_under_runs(WorkspaceLayout(workspace_root).runs)


def best_effort_prune_runs_detect_near_run(run_dir: str) -> None:
    """If ``run_dir`` is under ``.../runs/...``, prune empty ``.../runs/detect`` junk."""
    cur = os.path.abspath(os.path.expanduser(run_dir))
    while True:
        parent = os.path.dirname(cur)
        if parent == cur:
            return
        if os.path.basename(parent) == "runs":
            prune_ultralytics_default_detect_under_runs(parent)
            return
        cur = parent
