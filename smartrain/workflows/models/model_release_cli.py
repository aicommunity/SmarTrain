from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.cli_support.cli_prompts import print_numbered_options, prompt_choice, prompt_yes_no
from smartrain.cli_support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.interactive_contract import is_interactive_allowed
from smartrain.results_analyzer import find_run_directories, load_metadata, latest_test_metrics_path
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.run_artifacts import canonical_run_model_path, materialize_canonical_run_model
from smartrain.run_bundle_copy import copy_run_bundle


def build_model_release_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Release canonical run .pt into workspace models catalog (empty call starts interactive mode)"
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR})",
    )
    p.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run directory path or run index from discovered runs list",
    )
    return p


def _sanitize_stem(name: str) -> str:
    s = re.sub(r"[^\w.\-+]+", "_", str(name), flags=re.UNICODE).strip("._")
    return s[:180] if s else "unknown"


def _normalize_task(task: str | None) -> str:
    raw = (task or "").strip().lower()
    mapping = {
        "detection": "detect",
        "det": "detect",
        "classification": "classify",
    }
    return mapping.get(raw, raw or "detect")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_csv_last_row(csv_path: Path) -> dict[str, Any] | None:
    if not csv_path.is_file():
        return None
    last: dict[str, Any] | None = None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            last = {str(k).strip(): v for k, v in row.items()}
    return last


def _timestamp_for_name(md: dict[str, Any]) -> str:
    ts = (((md.get("timestamps") or {}).get("training") or {}).get("end")) or (
        ((md.get("timestamps") or {}).get("training") or {}).get("start")
    )
    if not ts:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return _sanitize_stem(str(ts))[:32]
    return dt.strftime("%Y%m%d_%H%M%S")


def _resolve_run_ref(layout: WorkspaceLayout, ref: str) -> Path:
    s = (ref or "").strip()
    if not s:
        raise ValueError("run reference is empty")
    if s.isdigit():
        runs = find_run_directories(layout.runs)
        idx = int(s)
        if idx < 1 or idx > len(runs):
            raise ValueError(f"run index {idx} is out of range 1..{len(runs)}")
        return Path(runs[idx - 1]).resolve()
    return Path(s).expanduser().resolve()


def _discover_runs(layout: WorkspaceLayout) -> list[Path]:
    return [Path(p).resolve() for p in find_run_directories(layout.runs)]


def _pick_run_interactive(layout: WorkspaceLayout) -> Path:
    runs = _discover_runs(layout)
    if not runs:
        raise RuntimeError(f"no runs with training_metadata.json found in {layout.runs}")
    options: list[str] = []
    printable: list[str] = []
    for run_dir in runs:
        ds = "?"
        model = "?"
        try:
            md = load_metadata(str(run_dir))
            ti = md.get("training_info") or {}
            ds = str((ti.get("dataset") or {}).get("name") or "?")
            model = str(ti.get("model") or "?")
        except Exception:
            pass
        rel = (
            str(run_dir.relative_to(Path(layout.root)))
            if run_dir.is_relative_to(Path(layout.root))
            else str(run_dir)
        )
        label = f"dataset={ds} model={model} run={rel}"
        printable.append(label)
        options.append(str(run_dir))
    print_numbered_options("runs", printable)
    picked = prompt_choice("Select run", options, default=options[0], show_options=False)
    return Path(picked).resolve()


def _resolve_data_yaml(run_dir: Path, md: dict[str, Any], layout: WorkspaceLayout) -> Path | None:
    args_yaml = run_dir / "train" / "args.yaml"
    if args_yaml.is_file():
        try:
            payload = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
            data_val = payload.get("data")
            if isinstance(data_val, str) and data_val.strip():
                p = Path(data_val).expanduser()
                if not p.is_absolute():
                    # Try run-relative then workspace-relative
                    p_run = (run_dir / p).resolve()
                    if p_run.is_file():
                        return p_run
                    p_ws = (Path(layout.root) / p).resolve()
                    if p_ws.is_file():
                        return p_ws
                elif p.is_file():
                    return p.resolve()
        except Exception:
            pass

    ds = (md.get("training_info") or {}).get("dataset") or {}
    puw = ds.get("path_under_workspace")
    if isinstance(puw, str) and puw.strip():
        cand = (Path(layout.root) / puw / "data.yaml").resolve()
        if cand.is_file():
            return cand
    pa = ds.get("path_absolute")
    if isinstance(pa, str) and pa.strip():
        cand = Path(pa).expanduser().resolve() / "data.yaml"
        if cand.is_file():
            return cand
    pr = ds.get("path_relative")
    if isinstance(pr, str) and pr.strip():
        cand = (run_dir / pr / "data.yaml").resolve()
        if cand.is_file():
            return cand
    return None


def _extract_classes(run_dir: Path, md: dict[str, Any], layout: WorkspaceLayout) -> tuple[list[str], dict[str, str]]:
    data_yaml = _resolve_data_yaml(run_dir, md, layout)
    names_obj: Any = None
    if data_yaml and data_yaml.is_file():
        try:
            payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
            names_obj = payload.get("names")
        except Exception:
            names_obj = None

    class_names: list[str] = []
    idx_map: dict[str, str] = {}
    if isinstance(names_obj, list):
        class_names = [str(x) for x in names_obj]
        idx_map = {str(i): v for i, v in enumerate(class_names)}
    elif isinstance(names_obj, dict):
        pairs: list[tuple[int, str]] = []
        for k, v in names_obj.items():
            try:
                ki = int(k)
            except Exception:
                continue
            pairs.append((ki, str(v)))
        pairs.sort(key=lambda x: x[0])
        class_names = [v for _, v in pairs]
        idx_map = {str(i): n for i, n in pairs}
    return class_names, idx_map


def _extract_io_spec(best_pt: Path, img_size: int | None) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    try:
        import torch
        from ultralytics import YOLO
    except Exception as e:
        return {"input": None, "outputs": None}, [f"ultralytics/torch unavailable: {e}"]

    sz = int(img_size or 640)
    io_spec: dict[str, Any] = {
        "input": {
            "shape": [1, 3, sz, sz],
            "dtype": "torch.float32",
        },
        "outputs": None,
    }
    try:
        model = YOLO(str(best_pt))
        x = torch.zeros((1, 3, sz, sz), dtype=torch.float32)
        with torch.no_grad():
            y = model.model(x)
    except Exception as e:
        return io_spec, [f"failed to inspect model outputs: {e}"]

    def pack(obj: Any) -> Any:
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            shape = [int(v) for v in list(obj.shape)]
            return {"shape": shape, "dtype": str(obj.dtype)}
        if isinstance(obj, (list, tuple)):
            return [pack(v) for v in obj]
        if isinstance(obj, dict):
            return {str(k): pack(v) for k, v in obj.items()}
        return str(type(obj))

    io_spec["outputs"] = pack(y)
    return io_spec, warnings


def _build_release_json(
    *,
    run_dir: Path,
    run_rel: str,
    md: dict[str, Any],
    source_best: Path,
    source_sha: str,
    target_pt: Path,
    layout: WorkspaceLayout,
) -> dict[str, Any]:
    ti = md.get("training_info") or {}
    hp = ti.get("hyperparameters") or {}
    class_names, idx_map = _extract_classes(run_dir, md, layout)
    io_spec, io_warnings = _extract_io_spec(source_best, hp.get("image_size"))
    test_csv = latest_test_metrics_path(str(run_dir))
    train_csv = run_dir / "train" / "results.csv"
    metrics: dict[str, Any] = {
        "train_results_last": _read_csv_last_row(train_csv),
        "test_metrics_last": _read_csv_last_row(Path(test_csv)) if test_csv else None,
    }
    return {
        "source": {
            "source_run": str(run_dir),
            "source_run_relative": run_rel,
            "source_weights": f"{run_dir.name}.pt",
            "source_sha256": source_sha,
            "released_at": datetime.now(timezone.utc).isoformat(),
        },
        "training": {
            "training_info": ti,
            "timestamps": md.get("timestamps"),
            "status": md.get("status"),
            "inference": md.get("inference"),
        },
        "metrics": metrics,
        "classes": {
            "list": class_names,
            "index_to_name": idx_map,
        },
        "io_spec": io_spec,
        "warnings": io_warnings,
        "artifacts": {
            "model_path": str(target_pt),
            "json_path": str(target_pt.with_suffix(".json")),
            "release_dir": str(target_pt.parent / target_pt.stem),
            "train_copy_dir": str(target_pt.parent / target_pt.stem / "train"),
            "test_copy_dir": str(target_pt.parent / target_pt.stem / "test"),
        },
    }


def _target_paths(layout: WorkspaceLayout, run_dir: Path, md: dict[str, Any]) -> tuple[Path, Path]:
    ti = md.get("training_info") or {}
    dataset_name = _sanitize_stem(str((ti.get("dataset") or {}).get("name") or "dataset"))
    task = _normalize_task(str(ti.get("task_type") or "detect"))
    model_name = _sanitize_stem(str(ti.get("model") or "model"))
    model_name = re.sub(r"\.(pt|onnx|engine)$", "", model_name, flags=re.IGNORECASE)
    dt = _timestamp_for_name(md)
    out_dir = Path(layout.models) / dataset_name
    fname = f"{task}_{model_name}_{dt}.pt"
    target_pt = (out_dir / fname).resolve()
    return target_pt, target_pt.with_suffix(".json")


def _copy_run_results(run_dir: Path, release_dir: Path) -> None:
    """Delegate to shared bundle copy (release keeps legacy scope: no ``tests/``, no ``models/``)."""
    copy_run_bundle(run_dir, release_dir, include_tests=False, copy_run_models=False)


def _same_release(
    *,
    existing_pt: Path,
    existing_json: Path,
    source_sha: str,
    run_rel: str,
) -> tuple[bool, str]:
    if not existing_pt.is_file():
        return False, "target model does not exist"
    existing_sha = _sha256_file(existing_pt)
    if existing_sha != source_sha:
        return False, "target model hash differs"
    if not existing_json.is_file():
        return False, "target json metadata does not exist"
    try:
        payload = json.loads(existing_json.read_text(encoding="utf-8"))
    except Exception:
        return False, "target json metadata is unreadable"
    src = payload.get("source") or {}
    expected_source_weights = f"{Path(run_rel).name}.pt"
    same_source = (
        str(src.get("source_run_relative") or "").strip() == run_rel
        and str(src.get("source_weights") or "").strip() == expected_source_weights
    )
    if not same_source:
        return False, "source run differs"
    return True, "same source and hash"


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    parser = build_model_release_arg_parser()
    args = parser.parse_args(argv)

    workspace_root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(workspace_root)
    os.makedirs(layout.models, exist_ok=True)
    os.makedirs(layout.runs, exist_ok=True)

    interactive_allowed = is_interactive_allowed(argv)
    interactive_used = False
    run_dir: Path
    if interactive_allowed and len(argv) == 0 and sys.stdin.isatty():
        run_dir = _pick_run_interactive(layout)
        interactive_used = True
    else:
        if not args.run:
            parser.error(
                "incomplete arguments: use --run (or run command without arguments for interactive mode)."
            )
        run_dir = _resolve_run_ref(layout, str(args.run))

    if not run_dir.is_dir():
        print(f"[ERROR] Run directory does not exist: {run_dir}", file=sys.stderr)
        raise SystemExit(1)
    meta_path = run_dir / "training_metadata.json"
    if not meta_path.is_file():
        print(f"[ERROR] training_metadata.json not found: {meta_path}", file=sys.stderr)
        raise SystemExit(1)
    source_best = Path(canonical_run_model_path(str(run_dir), ".pt"))
    if not source_best.is_file():
        materialized = materialize_canonical_run_model(
            str(run_dir),
            ext=".pt",
            move=True,
            normalize_metadata=True,
        )
        if materialized is not None:
            source_best = Path(materialized)
    if not source_best.is_file():
        print(f"[ERROR] run model not found: {source_best}", file=sys.stderr)
        raise SystemExit(1)

    md = load_metadata(str(run_dir))
    run_rel = os.path.relpath(str(run_dir), layout.root)
    target_pt, target_json = _target_paths(layout, run_dir, md)

    if interactive_used:
        print(f"[INFO] Source: {source_best}")
        print(f"[INFO] Target model: {target_pt}")
        print(f"[INFO] Target json: {target_json}")
        if not prompt_yes_no("Proceed with release?", default=True):
            print("[INFO] Release cancelled by user.")
            raise SystemExit(0)

    source_sha = _sha256_file(source_best)
    same, reason = _same_release(
        existing_pt=target_pt,
        existing_json=target_json,
        source_sha=source_sha,
        run_rel=run_rel,
    )
    if same:
        print(f"[OK] Already released, nothing to do: {target_pt} ({reason})")
        if interactive_used:
            replay_cmd = build_non_interactive_command("model release", parser, args)
            print_replay_command("model release", replay_cmd)
        raise SystemExit(0)
    if target_pt.exists():
        print(
            f"[ERROR] Target model already exists but differs: {target_pt} ({reason}).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    target_pt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source_best), str(target_pt))
    release_dir = target_pt.parent / target_pt.stem
    _copy_run_results(run_dir, release_dir)
    payload = _build_release_json(
        run_dir=run_dir,
        run_rel=run_rel,
        md=md,
        source_best=source_best,
        source_sha=source_sha,
        target_pt=target_pt,
        layout=layout,
    )
    target_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Released model: {target_pt}")
    print(f"[OK] Released metadata: {target_json}")
    if interactive_used:
        args.run = str(run_dir)
        replay_cmd = build_non_interactive_command("model release", parser, args)
        print_replay_command("model release", replay_cmd)


if __name__ == "__main__":
    main()
