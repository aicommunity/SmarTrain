from __future__ import annotations

import json
import os
from io import StringIO
from typing import Any, Callable

import pandas as pd


def recompute_status_path(
    run_dir: str,
    fingerprint: str,
    *,
    run_cache_root_cb: Callable[[str], str],
) -> str:
    return os.path.join(run_cache_root_cb(run_dir), "metrics", f"recompute_status_{fingerprint}.json")


def recompute_status_fingerprint(
    run_dir: str,
    data_yaml: str,
    split: str,
    requested_metrics: list[str],
    *,
    compute_fingerprint_cb: Callable[[dict[str, Any]], str],
    data_yaml_hash_cb: Callable[[str], str],
    weights_hash_cb: Callable[[str], str],
) -> str:
    return compute_fingerprint_cb(
        {
            "tool": "analyze-v2",
            "task": "metrics_recompute_status",
            "split": split,
            "data_yaml_hash": data_yaml_hash_cb(data_yaml),
            "weights_hash": weights_hash_cb(run_dir),
            "requested_metrics": sorted([m.strip() for m in requested_metrics if m.strip()]),
        }
    )


def load_recompute_status(
    run_dir: str,
    data_yaml: str,
    split: str,
    requested_metrics: list[str],
    *,
    recompute_status_fingerprint_cb: Callable[[str, str, str, list[str]], str],
    recompute_status_path_cb: Callable[[str, str], str],
) -> dict[str, Any] | None:
    fp = recompute_status_fingerprint_cb(run_dir, data_yaml, split, requested_metrics)
    path = recompute_status_path_cb(run_dir, fp)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def save_recompute_status(
    run_dir: str,
    data_yaml: str,
    split: str,
    requested_metrics: list[str],
    *,
    resolved: list[str],
    unresolved: list[str],
    status: str,
    recompute_status_fingerprint_cb: Callable[[str, str, str, list[str]], str],
    recompute_status_path_cb: Callable[[str, str], str],
    append_cache_entry_cb: Callable[[str, dict[str, Any]], None],
) -> None:
    fp = recompute_status_fingerprint_cb(run_dir, data_yaml, split, requested_metrics)
    path = recompute_status_path_cb(run_dir, fp)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "status": status,
        "resolved_metrics": sorted(set(resolved)),
        "unresolved_metrics": sorted(set(unresolved)),
        "requested_metrics": sorted(set(requested_metrics)),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    append_cache_entry_cb(
        run_dir,
        {
            "artifact": "metrics.recompute_status",
            "fingerprint": fp,
            "path": os.path.relpath(path, run_dir),
            "status": status,
        },
    )


def collect_missing_metrics_recompute_plan(
    run_dirs: list[str],
    requested_metrics: list[str],
    *,
    data_yaml: str | None = None,
    run_data_yaml_map: dict[str, str] | None = None,
    workspace: str | None = None,
    split: str = "test",
    read_test_metrics_for_run_cb: Callable[[str], dict[str, Any]],
    preferred_run_model_path_cb: Callable[[str, str], str],
    resolve_data_yaml_for_run_cb: Callable[[str, str | None], tuple[str | None, str | None]],
    load_recompute_status_cb: Callable[[str, str, str, list[str]], dict[str, Any] | None],
) -> dict[str, list[dict[str, Any]]]:
    recompute: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        row = read_test_metrics_for_run_cb(run_dir)
        recomputed_csv = os.path.join(run_dir, "test_metrics_recomputed.csv")
        if os.path.isfile(recomputed_csv):
            try:
                rdf = pd.read_csv(recomputed_csv)
                if len(rdf) > 0:
                    row.update(rdf.iloc[0].to_dict())
            except Exception:
                pass
        if not row:
            missing_metrics = list(requested_metrics)
        else:
            missing_metrics = [m for m in requested_metrics if pd.isna(pd.to_numeric(row.get(m), errors="coerce"))]
        if missing_metrics:
            resolved_yaml = str((run_data_yaml_map or {}).get(run_dir) or "").strip() or resolve_data_yaml_for_run_cb(
                run_dir, workspace
            )[0]
            if not resolved_yaml:
                resolved_yaml = data_yaml
            best_pt = preferred_run_model_path_cb(run_dir, ".pt")
            if not resolved_yaml:
                print(
                    "[INFO] "
                    + os.path.basename(run_dir.rstrip(os.sep))
                    + ": skip recompute prompt (no resolved data.yaml for run)."
                )
                skipped.append(
                    {
                        "run_dir": run_dir,
                        "missing_metrics": missing_metrics,
                        "reason": "no_data_yaml",
                    }
                )
                continue
            if not os.path.isfile(best_pt):
                print(
                    "[INFO] "
                    + os.path.basename(run_dir.rstrip(os.sep))
                    + ": skip recompute prompt (run model not found)."
                )
                skipped.append(
                    {
                        "run_dir": run_dir,
                        "missing_metrics": missing_metrics,
                        "reason": "missing_best_pt",
                    }
                )
                continue
            status_yaml = resolved_yaml or os.path.join(run_dir, "_missing_data_yaml_")
            status = load_recompute_status_cb(run_dir, status_yaml, split, requested_metrics)
            if status and isinstance(status, dict):
                unresolved = set(status.get("unresolved_metrics") or [])
                if unresolved and set(missing_metrics).issubset(unresolved):
                    print(
                        "[INFO] "
                        + os.path.basename(run_dir.rstrip(os.sep))
                        + ": skip recompute prompt for known unresolved metrics "
                        + f"{sorted(missing_metrics)} (fingerprint match)."
                    )
                    skipped.append(
                        {
                            "run_dir": run_dir,
                            "missing_metrics": missing_metrics,
                            "reason": "known_unresolved",
                        }
                    )
                    continue
            recompute.append(
                {
                    "run_dir": run_dir,
                    "missing_metrics": missing_metrics,
                    "data_yaml": resolved_yaml,
                }
            )
    return {"recompute": recompute, "skipped": skipped}


def runs_with_missing_metrics(
    run_dirs: list[str],
    requested_metrics: list[str],
    *,
    data_yaml: str | None = None,
    workspace: str | None = None,
    split: str = "test",
    collect_missing_metrics_recompute_plan_cb: Callable[..., dict[str, list[dict[str, Any]]]],
) -> list[str]:
    plan = collect_missing_metrics_recompute_plan_cb(
        run_dirs,
        requested_metrics,
        data_yaml=data_yaml,
        run_data_yaml_map=None,
        workspace=workspace,
        split=split,
    )
    return [str(x.get("run_dir")) for x in plan.get("recompute", []) if x.get("run_dir")]


def recompute_run_test_metrics(
    run_dir: str,
    data_yaml: str,
    split: str,
    *,
    val_batch: int = 1,
    val_imgsz: int = 640,
    val_half: bool = True,
    gpu_only: bool = False,
    preferred_run_model_path_cb: Callable[[str, str], str],
    materialize_preferred_run_model_cb: Callable[..., str | None],
    clear_gpu_memory_cb: Callable[[], None],
    resolve_run_val_profile_cb: Callable[..., tuple[int, int, bool]],
    ultralytics_sidecar_dir_cb: Callable[..., str],
    run_val_memory_safe_cb: Callable[..., Any],
) -> dict[str, Any]:
    from ultralytics import YOLO

    best_pt = preferred_run_model_path_cb(run_dir, ".pt")
    if not os.path.isfile(best_pt):
        materialized = materialize_preferred_run_model_cb(run_dir, ext=".pt", move=True, normalize_metadata=True)
        if materialized is not None:
            best_pt = str(materialized)
    if not os.path.isfile(best_pt):
        raise FileNotFoundError(f"run model not found: {best_pt}")
    model = YOLO(best_pt)
    clear_gpu_memory_cb()
    rb, ri, rh = resolve_run_val_profile_cb(
        run_dir,
        default_batch=val_batch,
        default_imgsz=val_imgsz,
        default_half=val_half,
    )
    ultra_proj = ultralytics_sidecar_dir_cb(run_dir, ".ultralytics_scratch")
    result = run_val_memory_safe_cb(
        model,
        data_yaml=data_yaml,
        split=split,
        val_batch=rb,
        val_imgsz=ri,
        val_half=rh,
        gpu_only=gpu_only,
        ultra_project=ultra_proj,
        ultra_name="val-recompute",
    )
    clear_gpu_memory_cb()
    csv_text = result.to_csv()
    rdf = pd.read_csv(StringIO(csv_text))
    if len(rdf) == 0:
        return {}
    row = rdf.iloc[0].to_dict()
    out_csv = os.path.join(run_dir, "test_metrics_recomputed.csv")
    rdf.to_csv(out_csv, index=False, encoding="utf-8")
    return row
