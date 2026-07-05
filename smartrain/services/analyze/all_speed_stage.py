from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable

import pandas as pd


from smartrain.services.analyze.report_labels import build_run_display_labels


def _apply_display_labels_to_benchmark_df(df: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    if df is None or len(df) == 0 or not label_map:
        return df
    out = df.copy()

    def _label_for_row(run_dir: Any, model: Any, run_name: Any) -> str:
        rd = str(run_dir or "").strip()
        if rd:
            abs_rd = os.path.abspath(rd.rstrip(os.sep))
            if abs_rd in label_map:
                return label_map[abs_rd]
        rn = str(run_name or "").strip() or (os.path.basename(rd.rstrip(os.sep)) if rd else "")
        if rn and rn in label_map:
            return label_map[rn]
        m = str(model or "").strip()
        if m and m in label_map:
            return label_map[m]
        return rn or m or "?"

    if "run_dir" in out.columns or "model" in out.columns:
        out["display_label"] = [
            _label_for_row(
                row.get("run_dir") if hasattr(row, "get") else None,
                row.get("model") if hasattr(row, "get") else None,
                row.get("run_name") if hasattr(row, "get") else None,
            )
            for _, row in out.iterrows()
        ]
    return out


def run_all_speed_stage(
    *,
    args: Any,
    profile: str,
    baseline: str,
    others: list[str],
    selected_run_dirs: list[str],
    session_root: str,
    runs_group_dir: str,
    run_data_yaml_map: dict[str, str],
    metric_sources_payload: dict[str, Any] | None,
    record_failure_cb: Callable[..., None],
    group_runs_by_data_yaml_cb: Callable[[list[str], dict[str, str]], tuple[dict[str, list[str]], list[str]]],
    cmd_inference_benchmark_cb: Callable[[argparse.Namespace], None],
    cmd_inference_plot_cb: Callable[[argparse.Namespace], None],
    write_speed_quality_artifacts_cb: Callable[..., dict[str, str] | None],
    build_run_display_labels_cb: Callable[[list[str]], dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if profile not in ("speed", "full"):
        return [], []

    artifacts: list[dict[str, str]] = []
    cache_events: list[dict[str, Any]] = []
    compare_runs = [baseline] + others
    if build_run_display_labels_cb is not None:
        label_map = build_run_display_labels_cb(compare_runs)
    else:
        label_map = build_run_display_labels(compare_runs, build_run_record_cb=None)

    run_groups, unresolved_for_speed = group_runs_by_data_yaml_cb(selected_run_dirs, run_data_yaml_map)
    if unresolved_for_speed:
        print("[WARN] Speed stage: skipped runs without resolved data.yaml:")
        for rd in unresolved_for_speed:
            print(f"[WARN]  - {os.path.basename(rd.rstrip(os.sep))}")
            record_failure_cb(
                stage="speed",
                status="skipped",
                reason_code="no_data_yaml",
                reason_detail="run excluded from speed stage due to unresolved data.yaml",
                run_dir=rd,
                split="test",
            )
    if not run_groups:
        print("[WARN] Speed stage skipped: no runs with resolved data.yaml.")

    inf_csv = os.path.join(session_root, "artifacts", "inference", "benchmark.csv")
    inf_png = os.path.join(session_root, "artifacts", "inference", "benchmark_bars.png")
    os.makedirs(os.path.dirname(inf_csv), exist_ok=True)
    inf_parts: list[pd.DataFrame] = []

    for g_idx, (group_yaml, group_runs) in enumerate(sorted(run_groups.items()), start=1):
        inf_part_csv = os.path.join(session_root, "artifacts", "inference", f"benchmark_group_{g_idx}.csv")
        ib_ns = argparse.Namespace(
            runs_group_dir=runs_group_dir,
            selected_run_dirs=group_runs,
            data_yaml=group_yaml,
            split="test",
            frames=100,
            device="cpu",
            half=False,
            out_csv=inf_part_csv,
            workspace=args.workspace,
            models_root=args.models_root,
            analytics_session=args.analytics_session,
            reuse_run_cache=True,
            cache_stats_out=os.path.join(session_root, "artifacts", "inference", f"cache_stats_group_{g_idx}.json"),
        )
        cmd_inference_benchmark_cb(ib_ns)
        if os.path.isfile(inf_part_csv):
            try:
                part_df = pd.read_csv(inf_part_csv)
                part_df["dataset_yaml_used"] = group_yaml
                if "run_name" not in part_df.columns:
                    part_df["run_name"] = part_df.get("run_dir", pd.Series(dtype=str)).astype(str).map(
                        lambda p: os.path.basename(str(p).rstrip(os.sep))
                    )
                if "benchmark_status" not in part_df.columns:
                    part_df["benchmark_status"] = "ok"
                inf_parts.append(part_df)
            except Exception as e:
                record_failure_cb(
                    stage="speed",
                    status="failed",
                    reason_code="benchmark_group_read_failed",
                    reason_detail=str(e),
                    split="test",
                )

    if inf_parts:
        inf_df = pd.concat(inf_parts, ignore_index=True)
        present = {
            os.path.abspath(str(p))
            for p in inf_df.get("run_dir", pd.Series(dtype=str)).astype(str).tolist()
            if str(p).strip()
        }
        present_by_name: dict[str, str] = {}
        for _, row in inf_df.iterrows():
            rname = str(row.get("run_name") or "").strip()
            if not rname:
                continue
            status = str(row.get("benchmark_status") or "ok").strip()
            current = present_by_name.get(rname)
            if current is None or (current != "ok" and status == "ok"):
                present_by_name[rname] = status
        for run_dir in selected_run_dirs:
            rd = os.path.abspath(run_dir)
            run_name = os.path.basename(run_dir.rstrip(os.sep))
            if rd in present:
                continue
            if present_by_name.get(run_name) == "ok":
                record_failure_cb(
                    stage="speed",
                    status="missing",
                    reason_code="run_dir_mismatch",
                    reason_detail="benchmark row matched by run_name but run_dir differs",
                    run_dir=run_dir,
                    split="test",
                )
                continue
            inf_df = pd.concat(
                [
                    inf_df,
                    pd.DataFrame(
                        [
                            {
                                "model": os.path.basename(run_dir.rstrip(os.sep)),
                                "run_name": os.path.basename(run_dir.rstrip(os.sep)),
                                "run_dir": run_dir,
                                "dataset_yaml_used": run_data_yaml_map.get(run_dir, ""),
                                "benchmark_status": "missing_or_failed",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            record_failure_cb(
                stage="speed",
                status="missing",
                reason_code="benchmark_missing_or_failed",
                reason_detail="benchmark row was not produced for selected run",
                run_dir=run_dir,
                split="test",
            )
        inf_df = _apply_display_labels_to_benchmark_df(inf_df, label_map)
        inf_df.to_csv(inf_csv, index=False, encoding="utf-8")
    else:
        inf_df = pd.DataFrame(
            [
                {
                    "model": os.path.basename(run_dir.rstrip(os.sep)),
                    "run_name": os.path.basename(run_dir.rstrip(os.sep)),
                    "run_dir": run_dir,
                    "dataset_yaml_used": run_data_yaml_map.get(run_dir, ""),
                    "benchmark_status": "missing_or_failed",
                }
                for run_dir in selected_run_dirs
            ]
        )
        inf_df = _apply_display_labels_to_benchmark_df(inf_df, label_map)
        inf_df.to_csv(inf_csv, index=False, encoding="utf-8")

    lb_csv = os.path.join(session_root, "artifacts", "leaderboard", "leaderboard.csv")
    if os.path.isfile(lb_csv) and os.path.isfile(inf_csv):
        try:
            lb_df = pd.read_csv(lb_csv)
            inf_df = pd.read_csv(inf_csv)
            if "run_dir" in lb_df.columns and "run_dir" in inf_df.columns:
                speed_series = (
                    inf_df.assign(
                        speed_metric=pd.to_numeric(inf_df.get("avg_inference_fps"), errors="coerce")
                    )
                    .dropna(subset=["run_dir", "speed_metric"])
                    .groupby("run_dir", as_index=True)["speed_metric"]
                    .max()
                )
                if not speed_series.empty:
                    existing_speed = pd.to_numeric(lb_df.get("speed_metric"), errors="coerce")
                    direct_speed = lb_df["run_dir"].map(speed_series)
                    bench_by_name = {
                        os.path.basename(str(k).rstrip(os.sep)): float(v)
                        for k, v in speed_series.items()
                    }
                    by_name_speed = lb_df["run_dir"].astype(str).map(
                        lambda p: bench_by_name.get(os.path.basename(str(p).rstrip(os.sep)))
                    )
                    lb_df["speed_metric"] = direct_speed.combine_first(by_name_speed).combine_first(existing_speed)
                    if "quality_metric" in lb_df.columns:
                        qv = pd.to_numeric(lb_df.get("quality_metric"), errors="coerce")
                        sv = pd.to_numeric(lb_df.get("speed_metric"), errors="coerce")
                        stable = (
                            pd.Series([1.0] * len(lb_df))
                            if "training_ok" not in lb_df.columns and "testing_ok" not in lb_df.columns
                            else (
                                pd.to_numeric(lb_df.get("training_ok"), errors="coerce").fillna(0.0)
                                * pd.to_numeric(lb_df.get("testing_ok"), errors="coerce").fillna(0.0)
                            )
                        )
                        denom = 0.6 + 0.25 + 0.15
                        lb_df["composite_score"] = (
                            (0.6 * qv.fillna(0.0)) + (0.25 * sv.fillna(0.0)) + (0.15 * stable.fillna(0.0))
                        ) / denom
                    lb_df = lb_df.sort_values("composite_score", ascending=False)
                    lb_df.to_csv(lb_csv, index=False, encoding="utf-8")
        except Exception:
            record_failure_cb(
                stage="speed",
                status="failed",
                reason_code="leaderboard_speed_merge_failed",
                reason_detail="failed to merge speed benchmark into leaderboard",
                split="test",
            )

    ip_ns = argparse.Namespace(
        csv=inf_csv,
        metric="avg_inference_ms_per_frame",
        out_png=inf_png,
        workspace=args.workspace,
        models_root=args.models_root,
        analytics_session=args.analytics_session,
    )
    cmd_inference_plot_cb(ip_ns)
    artifacts.extend(
        [
            {"role": "inference_csv", "path": os.path.relpath(inf_csv, session_root)},
            {"role": "inference_png", "path": os.path.relpath(inf_png, session_root)},
        ]
    )

    for g_idx in range(1, len(run_groups) + 1):
        cache_stats_path = os.path.join(session_root, "artifacts", "inference", f"cache_stats_group_{g_idx}.json")
        if os.path.isfile(cache_stats_path):
            try:
                with open(cache_stats_path, "r", encoding="utf-8") as f:
                    cache_events.extend(json.load(f).get("cache", []))
            except Exception as e:
                record_failure_cb(
                    stage="speed",
                    status="failed",
                    reason_code="cache_stats_read_failed",
                    reason_detail=str(e),
                    split="test",
                )

    if os.path.isfile(inf_csv):
        try:
            if os.path.getsize(inf_csv) > 0:
                sq = write_speed_quality_artifacts_cb(
                    session_root,
                    inf_csv,
                    compare_runs,
                    metric_sources_payload,
                    scatter_x=str(getattr(args, "scatter_x", "avg_inference_ms_per_frame")),
                    scatter_y=str(getattr(args, "scatter_y", "mAP50-95")),
                    run_data_yaml_map=run_data_yaml_map,
                    display_labels=label_map,
                )
                if sq:
                    artifacts.extend(
                        [
                            {"role": "speed_quality_csv", "path": sq["csv"]},
                            {"role": "speed_quality_png", "path": sq["png"]},
                        ]
                    )
                    try:
                        sq_abs = os.path.join(session_root, sq["csv"])
                        sq_df = pd.read_csv(sq_abs)
                        expected = {
                            os.path.basename(str(r).rstrip(os.sep))
                            for r in selected_run_dirs
                        }
                        if "run_name" in sq_df.columns:
                            actual = set(sq_df["run_name"].astype(str).tolist())
                        else:
                            actual = set(sq_df.get("model", pd.Series(dtype=str)).astype(str).tolist())
                        if len(actual) < len(expected):
                            record_failure_cb(
                                stage="speed_quality",
                                status="failed",
                                reason_code="png_incomplete_series",
                                reason_detail=f"speed_quality models={sorted(actual)} expected={sorted(expected)}",
                                split="test",
                            )
                    except Exception as e:
                        record_failure_cb(
                            stage="speed_quality",
                            status="failed",
                            reason_code="speed_quality_validation_failed",
                            reason_detail=str(e),
                            split="test",
                        )
        except Exception:
            record_failure_cb(
                stage="speed_quality",
                status="failed",
                reason_code="speed_quality_write_failed",
                reason_detail="failed to build speed-quality artifacts",
                split="test",
            )

    return artifacts, cache_events

