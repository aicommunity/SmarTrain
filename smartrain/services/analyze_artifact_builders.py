from __future__ import annotations

import os
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd


def collect_confidence_recommendation_tables(
    *,
    run_dirs: list[str],
    out_dir: str,
    flat_row_for_run: Callable[[str], dict[str, Any]],
    recommendation_file_path: Callable[[str, str], str],
    read_recommendation_file: Callable[[str], dict[str, Any] | None],
) -> dict[str, str]:
    rows_by_objective: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for run_dir in run_dirs:
        model_name: str | None = None
        dataset_name: str | None = None
        try:
            flat = flat_row_for_run(run_dir)
            model_name = flat.get("model")
            dataset_name = flat.get("dataset_name")
        except Exception:
            model_name = None
            dataset_name = None
        model_name = model_name or os.path.basename(run_dir.rstrip(os.sep))
        dataset_name = dataset_name or os.path.basename(os.path.dirname(run_dir.rstrip(os.sep)))

        for split in ("val", "test"):
            payload = read_recommendation_file(recommendation_file_path(run_dir, split))
            if not isinstance(payload, dict):
                continue
            objectives = payload.get("objectives")
            if not isinstance(objectives, dict):
                continue
            for objective in ("A", "B", "C"):
                item = objectives.get(objective)
                if not isinstance(item, dict):
                    continue
                beta = item.get("beta")
                global_row = item.get("global")
                if isinstance(global_row, dict):
                    rows_by_objective[objective].append(
                        {
                            "run_dir": run_dir,
                            "run_name": os.path.basename(run_dir.rstrip(os.sep)),
                            "model": model_name,
                            "dataset": dataset_name,
                            "split": split,
                            "objective": objective,
                            "beta": beta,
                            "level": "global",
                            "class_id": -1,
                            "class_name": "all",
                            "recommended_conf": global_row.get("threshold"),
                            "target_metric": global_row.get("metric_value"),
                            "precision": global_row.get("precision"),
                            "recall": global_row.get("recall"),
                            "f1": global_row.get("f1"),
                            "support_instances": None,
                            "status": global_row.get("status") or payload.get("status"),
                            "reason": global_row.get("reason") or payload.get("reason"),
                        }
                    )
                per_class = item.get("per_class")
                if isinstance(per_class, list):
                    for row in per_class:
                        if not isinstance(row, dict):
                            continue
                        rows_by_objective[objective].append(
                            {
                                "run_dir": run_dir,
                                "run_name": os.path.basename(run_dir.rstrip(os.sep)),
                                "model": model_name,
                                "dataset": dataset_name,
                                "split": split,
                                "objective": objective,
                                "beta": beta,
                                "level": "class",
                                "class_id": row.get("class_id"),
                                "class_name": row.get("class_name"),
                                "recommended_conf": row.get("threshold"),
                                "target_metric": row.get("metric_value"),
                                "precision": row.get("precision"),
                                "recall": row.get("recall"),
                                "f1": row.get("f1"),
                                "support_instances": row.get("support_instances"),
                                "status": row.get("status") or payload.get("status"),
                                "reason": row.get("reason") or payload.get("reason"),
                            }
                        )

    out: dict[str, str] = {}
    os.makedirs(out_dir, exist_ok=True)
    sort_cols = ["run_name", "split", "level", "class_id"]
    for objective in ("A", "B", "C"):
        rows = rows_by_objective.get(objective) or []
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if set(sort_cols).issubset(df.columns):
            df = df.sort_values(sort_cols, ascending=[True, True, True, True])
        out_path = os.path.join(out_dir, f"confidence_recommendations_{objective}.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        out[objective] = out_path
    return out


def write_speed_quality_artifacts(
    *,
    session_root: str,
    inference_csv: str,
    requested_runs: list[str],
    metric_sources_payload: dict[str, Any] | None,
    scatter_x: str,
    scatter_y: str,
    run_data_yaml_map: dict[str, str] | None,
    read_test_metrics_for_run: Callable[[str], dict[str, Any]],
) -> dict[str, Any] | None:
    if not os.path.isfile(inference_csv):
        return None
    df = pd.read_csv(inference_csv)
    if len(df) == 0 or "run_dir" not in df.columns:
        return None
    source_map: dict[str, dict[str, str]] = {}
    if isinstance(metric_sources_payload, dict):
        source_map = metric_sources_payload.get("sources") or {}
    rows: list[dict[str, Any]] = []
    run_data_yaml_map = run_data_yaml_map or {}
    df_with_name = df.copy()
    if "run_name" not in df_with_name.columns:
        if "run_dir" in df_with_name.columns:
            df_with_name["run_name"] = df_with_name["run_dir"].astype(str).map(
                lambda p: os.path.basename(str(p).rstrip(os.sep))
            )
        else:
            df_with_name["run_name"] = ""
    for run_dir in requested_runs:
        run_name = os.path.basename(run_dir.rstrip(os.sep))
        sub = df_with_name[(df_with_name["run_dir"] == run_dir) | (df_with_name["run_name"] == run_name)].copy()
        if len(sub) == 0:
            continue
        status_score = sub.get("benchmark_status", pd.Series(["ok"] * len(sub))).astype(str).map(
            lambda s: 0 if s == "ok" else 1
        )
        val_score = pd.to_numeric(sub.get(scatter_x), errors="coerce").isna().astype(int)
        sub = sub.assign(_status_score=status_score, _val_score=val_score).sort_values(
            ["_status_score", "_val_score"], ascending=[True, True]
        )
        rec = sub.iloc[0].to_dict()
        base_metrics = read_test_metrics_for_run(run_dir)
        recomputed_csv = os.path.join(run_dir, "test_metrics_recomputed.csv")
        if os.path.isfile(recomputed_csv):
            try:
                rdf = pd.read_csv(recomputed_csv)
                if len(rdf) > 0:
                    base_metrics.update(rdf.iloc[0].to_dict())
            except Exception:
                pass
        quality = base_metrics.get(scatter_y)
        q_num = pd.to_numeric(quality, errors="coerce")
        s_num = pd.to_numeric(rec.get(scatter_x), errors="coerce")
        if pd.isna(q_num) or pd.isna(s_num):
            continue
        q_src = (source_map.get(run_dir) or {}).get(scatter_y, "original")
        rows.append(
            {
                "run_dir": run_dir,
                "model": rec.get("model") or os.path.basename(run_dir.rstrip(os.sep)),
                "scatter_x_metric": scatter_x,
                "scatter_y_metric": scatter_y,
                "scatter_x_value": float(s_num),
                "scatter_y_value": float(q_num),
                "quality_source": q_src,
                "dataset_yaml_used": run_data_yaml_map.get(run_dir, ""),
            }
        )
    if len(rows) < 2:
        return None
    out_dir = os.path.join(session_root, "artifacts", "speed_quality")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "speed_quality.csv")
    png_path = os.path.join(out_dir, "speed_vs_map.png")
    out_df = pd.DataFrame(rows).sort_values("scatter_x_value", ascending=True)
    out_df.to_csv(csv_path, index=False, encoding="utf-8")
    plt.figure(figsize=(9, 6))
    plt.scatter(out_df["scatter_x_value"], out_df["scatter_y_value"], alpha=0.9)
    for _, row in out_df.iterrows():
        plt.text(float(row["scatter_x_value"]), float(row["scatter_y_value"]), str(row["model"]), fontsize=8)
    plt.xlabel(scatter_x)
    plt.ylabel(scatter_y)
    plt.title("Speed vs Quality")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.close()
    return {
        "csv": os.path.relpath(csv_path, session_root),
        "png": os.path.relpath(png_path, session_root),
    }
