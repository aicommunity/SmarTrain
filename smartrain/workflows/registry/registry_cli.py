#!/usr/bin/env python3
"""
Runs in workspace/runs and the workspace/models directory: list, information, promotion of scales.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.core.runtime.run_artifacts import preferred_run_model_path, materialize_preferred_run_model
from smartrain.core.runtime.run_bundle_copy import copy_run_bundle, normalize_training_metadata_paths_for_bundle
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.core.analyze.run_metrics_discovery import latest_test_metrics_path
from smartrain.core.runtime.run_discovery import find_run_directories
from smartrain.core.runtime.run_refs import resolve_run_ref
from smartrain.services.registry.run_fields import load_run_list_fields, load_run_training_metadata


MANIFEST_NAME = "model_manifest.json"


class RegistryCliContext:
    """Workspace context for registry subcommands."""

    def __init__(self, layout: WorkspaceLayout):
        self.layout = layout
        self.workspace_root = layout.root
        self.runs_dir = layout.runs
        self.models_dir = layout.models


def _ordered_run_dirs(ctx: RegistryCliContext) -> list[str]:
    return find_run_directories(ctx.runs_dir)


def _resolve_run_ref(ctx: RegistryCliContext, ref: str) -> str:
    return resolve_run_ref(ctx.runs_dir, ref)


def _cmd_runs_list(ctx: RegistryCliContext) -> None:
    runs = find_run_directories(ctx.runs_dir)
    if not runs:
        print("(runs with training_metadata.json not found)")
        return
    print(f"{'#':>4}  {'model':<14}  {'dataset':<24}  {'run_dir'}")
    print("-" * 100)
    for i, rd in enumerate(runs, start=1):
        try:
            fields = load_run_list_fields(rd)
            m = fields["model"]
            ds = fields["dataset"]
            print(f"{i:4d}  {str(m)[:14]:<14}  {str(ds)[:24]:<24}  {rd}")
        except (OSError, KeyError, TypeError, ValueError) as e:
            print(f"{i:4d} {'?':<14} {'?':<24} {rd} [error: {e}]")


def _cmd_runs_info(ctx: RegistryCliContext, run_path: str) -> None:
    run_path = _resolve_run_ref(ctx, run_path)
    md = load_run_training_metadata(run_path)
    ti = md["training_info"]
    print(json.dumps({"run_dir": run_path, "training_info": ti, "timestamps": md["timestamps"]}, ensure_ascii=False, indent=2))
    best = preferred_run_model_path(run_path, ".pt")
    if not os.path.isfile(best):
        materialized = materialize_preferred_run_model(run_path, ext=".pt", move=True, normalize_metadata=True)
        if materialized is not None:
            best = str(materialized)
    print(f"run model exists: {os.path.isfile(best)}  path: {best}")
    rc = os.path.join(run_path, "train", "results.csv")
    print(f"results.csv exists: {os.path.isfile(rc)}")


def _cmd_runs_metrics(ctx: RegistryCliContext, run_path: str) -> None:
    run_path = _resolve_run_ref(ctx, run_path)
    tm = latest_test_metrics_path(run_path)
    if not tm:
        print(f"[ERROR] No test_metrics*.csv in {run_path}", file=sys.stderr)
        sys.exit(1)
    print(tm)
    with open(tm, "r", encoding="utf-8") as f:
        print(f.read())


def _sanitize_stem(name: str) -> str:
    s = re.sub(r"[^\w.\-+]+", "_", name, flags=re.UNICODE)
    s = s.strip("._")
    return s[:180] if s else "model"


def _friendly_name_base(md: dict) -> str:
    ti = md["training_info"]
    model = _sanitize_stem(str(ti["model"]))
    ds = _sanitize_stem(str(ti["dataset"]["name"]))
    ds_entry = ti["dataset"]
    h = ds_entry["hash"]
    if h is None:
        h = "nohash"
    else:
        h = str(h)[:8]
    ts = md["timestamps"]["training"]
    end = ts["end"]
    if end is None:
        end = ts["start"]
    if end is None:
        dt_part = "unknown"
    else:
        try:
            dtp = datetime.fromisoformat(end.replace("Z", "+00:00"))
            dt_part = dtp.strftime("%Y%m%d_%H%M")
        except ValueError:
            dt_part = _sanitize_stem(end[:19])
    return f"{dt_part}_{model}_{ds}_{h}"


def _unique_model_dir(models_root: str, base: str) -> str:
    d = os.path.join(models_root, base)
    if not os.path.exists(d):
        return d
    n = 2
    while True:
        cand = os.path.join(models_root, f"{base}_{n}")
        if not os.path.exists(cand):
            return cand
        n += 1


def _cmd_models_add(ctx: RegistryCliContext, run_path: str) -> None:
    run_path = _resolve_run_ref(ctx, run_path)
    meta_path = os.path.join(run_path, "training_metadata.json")
    if not os.path.isfile(meta_path):
        print(f"[ERROR] No training_metadata.json: {run_path}", file=sys.stderr)
        sys.exit(1)
    best = preferred_run_model_path(run_path, ".pt")
    if not os.path.isfile(best):
        materialized = materialize_preferred_run_model(run_path, ext=".pt", move=True, normalize_metadata=True)
        if materialized is not None:
            best = str(materialized)
    if not os.path.isfile(best):
        print(f"[ERROR] No run model: {best}", file=sys.stderr)
        sys.exit(1)
    md = load_run_training_metadata(run_path)
    base = _friendly_name_base(md)
    dest_dir = _unique_model_dir(ctx.models_dir, base)
    friendly = os.path.basename(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    run_root = Path(run_path).resolve()
    best_p = Path(best).resolve()
    try:
        weights_rel = best_p.relative_to(run_root).as_posix()
    except ValueError:
        weights_rel = Path("models") / best_p.name
        weights_rel = weights_rel.as_posix()

    copy_run_bundle(run_root, Path(dest_dir), include_tests=True, copy_run_models=True)

    dest_weights = Path(dest_dir).joinpath(*weights_rel.split("/"))
    if not dest_weights.is_file():
        dest_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best_p), str(dest_weights))

    normalize_training_metadata_paths_for_bundle(Path(dest_dir) / "training_metadata.json", weights_rel)

    promoted = datetime.now(timezone.utc).isoformat()
    ti = md["training_info"]
    ds = ti["dataset"]
    from smartrain.core.training.train_profile import task_to_metadata_task_type

    task_type = ""
    raw_task = ti.get("task_type")
    if not str(raw_task or "").strip():
        utrain = ti.get("ultralytics_train")
        if isinstance(utrain, dict) and utrain.get("task"):
            raw_task = utrain.get("task")
    if str(raw_task or "").strip():
        task_type = task_to_metadata_task_type(str(raw_task))
    manifest = {
        "friendly_name": friendly,
        "weights_file": weights_rel,
        "source_run": run_path,
        "source_run_relative": os.path.relpath(run_path, ctx.workspace_root),
        "training_end": md["timestamps"]["training"]["end"],
        "model": ti["model"],
        "dataset_name": ds["name"],
        "dataset_hash": ds["hash"],
        "promoted_at": promoted,
        "workspace_root": ctx.workspace_root,
        "bundle_layout_version": 2,
    }
    if task_type:
        manifest["task_type"] = task_type
    with open(os.path.join(dest_dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] Model: {dest_weights}")
    print(f"[OK] Manifest: {os.path.join(dest_dir, MANIFEST_NAME)}")


def _cmd_models_list(ctx: RegistryCliContext) -> None:
    if not os.path.isdir(ctx.models_dir):
        print("(models directory is empty or not created)")
        return
    names = sorted(
        d for d in os.listdir(ctx.models_dir)
        if os.path.isdir(os.path.join(ctx.models_dir, d))
    )
    for n in names:
        man = os.path.join(ctx.models_dir, n, MANIFEST_NAME)
        if os.path.isfile(man):
            print(n)
        else:
            print(f"{n} (no {MANIFEST_NAME})")


def _cmd_models_info(ctx: RegistryCliContext, name: str) -> None:
    d = os.path.join(ctx.models_dir, name)
    man = os.path.join(d, MANIFEST_NAME)
    if not os.path.isfile(man):
        print(f"[ERROR] No directory or manifest: {d}", file=sys.stderr)
        sys.exit(1)
    with open(man, "r", encoding="utf-8") as f:
        print(f.read())


def _cmd_models_remove(ctx: RegistryCliContext, name: str) -> None:
    d = os.path.join(ctx.models_dir, name)
    if not os.path.isdir(d):
        print(f"[ERROR] No directory: {d}", file=sys.stderr)
        sys.exit(1)
    shutil.rmtree(d)
    print(f"[OK] Deleted: {d}")


def build_registry_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Register of runs and promoted models")
    p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (aka {WORKSPACE_ENV_VAR})",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_rl = sub.add_parser("runs-list", help="List of runs under runs/")
    p_rl.set_defaults(handler="runs_list")

    p_ri = sub.add_parser("runs-info", help="JSON training_info + paths to weights")
    p_ri.add_argument(
        "run_path",
        type=str,
        help="Run directory or line number as in runs-list",
    )
    p_ri.set_defaults(handler="runs_info")

    p_rm = sub.add_parser("runs-metrics", help="Path and contents of test_metrics*.csv")
    p_rm.add_argument(
        "run_path",
        type=str,
        help="Run directory or line number as in runs-list",
    )
    p_rm.set_defaults(handler="runs_metrics")

    p_ma = sub.add_parser(
        "models-add",
        help="Promote run into models/<friendly>/ (weights under models/, train-*, tests/, train/test roots, metadata)",
    )
    p_ma.add_argument(
        "run_path",
        type=str,
        help="Run directory or line number as in runs-list",
    )
    p_ma.set_defaults(handler="models_add")

    p_ml = sub.add_parser("models-list", help="List of names in models/")
    p_ml.set_defaults(handler="models_list")

    p_mi = sub.add_parser("models-info", help="Output of model_manifest.json")
    p_mi.add_argument("name", type=str)
    p_mi.set_defaults(handler="models_info")

    p_mr = sub.add_parser("models-remove", help="Delete directory models/<name>/")
    p_mr.add_argument("name", type=str)
    p_mr.set_defaults(handler="models_remove")

    return p


def _parse_args(argv=None) -> argparse.Namespace:
    return build_registry_arg_parser().parse_args(argv)


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)
    try:
        root = resolve_workspace_root(args.workspace)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    ctx = RegistryCliContext(WorkspaceLayout(root))
    os.makedirs(ctx.runs_dir, exist_ok=True)
    os.makedirs(ctx.models_dir, exist_ok=True)

    h = args.handler
    if h == "runs_list":
        _cmd_runs_list(ctx)
    elif h == "runs_info":
        _cmd_runs_info(ctx, args.run_path)
    elif h == "runs_metrics":
        _cmd_runs_metrics(ctx, args.run_path)
    elif h == "models_add":
        _cmd_models_add(ctx, args.run_path)
    elif h == "models_list":
        _cmd_models_list(ctx)
    elif h == "models_info":
        _cmd_models_info(ctx, args.name)
    elif h == "models_remove":
        _cmd_models_remove(ctx, args.name)
    else:
        print(f"[ERROR] Unknown command: {h}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
