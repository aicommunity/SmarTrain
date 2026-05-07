from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Callable


def resolve_compare_artifact_path(path: str, session_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(session_dir, path)


def resolve_compare_png_path(
    out_png_cli: str | None,
    out_csv: str,
    *,
    resolve_workspace_root_cb: Callable[[str | None], str],
    workspace_layout_cls,
    workspace_cli: str | None,
    session_name: str | None,
) -> str:
    if out_png_cli:
        return str(out_png_cli)
    if session_name:
        try:
            ws = resolve_workspace_root_cb(workspace_cli)
            layout = workspace_layout_cls(ws)
            return os.path.join(layout.analytics, str(session_name), "artifacts", "compare", "compare_curves.png")
        except ValueError:
            pass
    return os.path.join(os.path.dirname(os.path.abspath(out_csv)), "compare_curves.png")


def finalize_compare_analytics_session(
    *,
    args,
    baseline: str,
    others: list[str],
    out_csv: str,
    out_png: str,
    bar_path: str | None,
    insights_path: str | None,
    resolve_workspace_root_cb: Callable[[str | None], str],
    workspace_layout_cls,
    workspace_env_var: str,
) -> None:
    session_name = (getattr(args, "analytics_session", None) or "").strip()
    if not session_name:
        return
    try:
        ws = resolve_workspace_root_cb(args.workspace)
    except ValueError:
        print(
            f"[ERROR] --analytics-session requires --workspace or {workspace_env_var}.",
            file=sys.stderr,
        )
        sys.exit(1)
    layout = workspace_layout_cls(ws)
    dest_root = os.path.join(layout.analytics, session_name)
    os.makedirs(dest_root, exist_ok=True)
    artifacts: list[dict[str, str]] = []
    for role, p in (
        ("delta_csv", out_csv),
        ("curves_png", out_png),
        ("bars_png", bar_path),
        ("insights_txt", insights_path),
    ):
        if not p:
            continue
        src = resolve_compare_artifact_path(p, dest_root)
        if not os.path.isfile(src):
            continue
        rel_dir = os.path.join("artifacts", "compare")
        dst_dir = os.path.join(dest_root, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        try:
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
            artifacts.append({"role": role, "path": os.path.join(rel_dir, os.path.basename(src))})
        except Exception:
            pass
    manifest = {
        "session_name": session_name,
        "type": "compare",
        "baseline": baseline,
        "others": others,
        "artifacts": artifacts,
    }
    with open(os.path.join(dest_root, "session.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] Session manifest: {os.path.join(dest_root, 'session.json')}")

