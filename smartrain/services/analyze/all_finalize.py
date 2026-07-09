from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable

import pandas as pd

from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command


from smartrain.services.analyze.ultralytics_test_ensure import ensure_ultralytics_test_for_runs
from smartrain.services.analyze.eval_dataset_test_artifacts import collect_eval_dataset_test_artifacts


def finalize_all_session(
    *,
    args: Any,
    session_root: str,
    profile: str,
    baseline: str,
    others: list[str],
    data_yaml: str,
    report_languages: list[str],
    run_data_yaml_map: dict[str, str],
    unresolved_data_yaml_runs: list[str],
    artifacts: list[dict[str, str]],
    cache_events: list[dict[str, Any]],
    artifact_failures: list[dict[str, Any]],
    metric_sources_payload: dict[str, Any] | None,
    recompute_missing_metrics: bool,
    build_abbreviations_for_report_cb: Callable[[list[str]], dict[str, str]],
    collect_ultralytics_test_artifacts_cb: Callable[..., tuple[list[dict[str, Any]], list[dict[str, str]]]],
    write_format_compare_artifacts_cb: Callable[[str, list[str]], dict[str, Any] | None],
    collect_confidence_recommendation_tables_cb: Callable[[list[str], str], dict[str, str]],
    write_manifest_cb: Callable[[str, dict[str, Any]], None],
    write_analysis_report_cb: Callable[..., dict[str, str]],
    record_failure_cb: Callable[..., None],
    replay_parser: argparse.ArgumentParser | None = None,
) -> None:
    abbreviations = build_abbreviations_for_report_cb([baseline] + others)
    ensure_ultralytics_test_for_runs(
        [baseline] + others,
        args=args,
        profile=profile,
        workspace_cli=getattr(args, "workspace", None),
        run_data_yaml_map=run_data_yaml_map,
        record_failure_cb=record_failure_cb,
    )
    ultralytics_test_rows, ultralytics_test_artifacts = collect_ultralytics_test_artifacts_cb(
        session_root,
        [baseline] + others,
        abbreviations,
    )
    artifacts.extend(ultralytics_test_artifacts)
    eval_dataset_rows, eval_dataset_artifacts = collect_eval_dataset_test_artifacts(
        session_root,
        [baseline] + others,
        abbreviations,
    )
    artifacts.extend(eval_dataset_artifacts)

    format_compare = write_format_compare_artifacts_cb(session_root, [baseline] + others)
    if format_compare and format_compare.get("csv"):
        artifacts.append({"role": "format_compare_csv", "path": str(format_compare["csv"])})
        perf_rel = str(format_compare.get("perf_test_csv") or "")
        if perf_rel:
            perf_abs = os.path.join(session_root, perf_rel)
            if os.path.isfile(perf_abs):
                try:
                    perf_df = pd.read_csv(perf_abs)
                    if "performance_status" in perf_df.columns:
                        bad = perf_df[perf_df["performance_status"].astype(str).str.lower() != "ok"].copy()
                        for _, row in bad.iterrows():
                            record_failure_cb(
                                stage="format_performance",
                                status="missing",
                                reason_code=str(row.get("performance_reason") or "performance_not_collected"),
                                reason_detail="format performance row is incomplete",
                                run_dir=str(row.get("run_dir") or ""),
                                format_name=str(row.get("format") or ""),
                                split=str(row.get("split") or "test"),
                            )
                except Exception as e:
                    record_failure_cb(
                        stage="format_performance",
                        status="failed",
                        reason_code="format_perf_read_failed",
                        reason_detail=str(e),
                        split="test",
                    )

    conf_tables = collect_confidence_recommendation_tables_cb(
        [baseline] + others,
        os.path.join(session_root, "artifacts", "confidence"),
    )
    for objective in ("A", "B", "C"):
        p = conf_tables.get(objective)
        if p and os.path.isfile(p):
            artifacts.append(
                {
                    "role": f"confidence_recommendations_{objective.lower()}_csv",
                    "path": os.path.relpath(p, session_root),
                }
            )

    single_run_mode = not others
    manifest = {
        "session_name": os.path.basename(session_root),
        "profile": profile,
        "baseline": baseline,
        "others": others,
        "single_run_mode": single_run_mode,
        "artifacts": artifacts,
        "images": [a["path"] for a in artifacts if a["path"].endswith(".png")],
        "tables": [a["path"] for a in artifacts if a["path"].endswith(".csv")],
        "artifact_scope": {
            "single_run": ["test_metrics_recomputed.csv", "pr aggregate/per_class", "inference benchmark profile"],
            "cross_run": ["compare", "leaderboard", "speed_quality", "reports", "session manifest"],
        },
        "sections": [
            "executive_summary",
            "comparison_context",
            "quality_analysis",
            "speed_analysis",
            "format_compare",
            "per_class_analysis",
            "conclusion",
        ],
        "abbreviations": abbreviations,
        "run_data_yaml_map": run_data_yaml_map,
        "runs_with_unresolved_data_yaml": unresolved_data_yaml_runs,
    }
    if ultralytics_test_rows:
        manifest["ultralytics_test"] = ultralytics_test_rows
    if eval_dataset_rows:
        manifest["eval_dataset_tests"] = eval_dataset_rows
    if conf_tables:
        manifest["confidence_recommendations"] = {
            key: os.path.relpath(path, session_root) for key, path in conf_tables.items()
        }
    if metric_sources_payload is not None:
        manifest["metric_sources"] = metric_sources_payload
    else:
        record_failure_cb(
            stage="metrics",
            status="missing",
            reason_code="metric_sources_missing",
            reason_detail="metric_sources.json missing or unreadable",
            split="test",
        )
    if artifact_failures:
        manifest["artifact_failures"] = artifact_failures
        by_reason: dict[str, int] = {}
        for item in artifact_failures:
            reason = str(item.get("reason_code") or "unknown")
            by_reason[reason] = by_reason.get(reason, 0) + 1
        manifest["artifact_failures_summary"] = {
            "total": len(artifact_failures),
            "by_reason_code": by_reason,
        }
    if cache_events:
        manifest["cache"] = {
            "events": cache_events,
            "hits": sum(1 for e in cache_events if e.get("status") == "hit"),
            "misses": sum(1 for e in cache_events if e.get("status") == "miss"),
        }
    if profile == "full":
        manifest["pr_per_class"] = {
            "csv": "artifacts/pr/per_class/pr_per_class.csv",
            "dir": "artifacts/pr/per_class",
        }
    if profile in ("speed", "full"):
        manifest["speed_quality"] = {
            "csv": "artifacts/speed_quality/speed_quality.csv",
            "png": "artifacts/speed_quality/speed_vs_map.png",
            "scatter_x": str(getattr(args, "scatter_x", "avg_inference_ms_per_frame")),
            "scatter_y": str(getattr(args, "scatter_y", "mAP50-95")),
        }
    if format_compare:
        manifest["format_comparison"] = format_compare
        test_csv_val = str(format_compare.get("test_csv") or "").strip()
        for key in ("test_csv", "val_csv", "pt_uni_csv", "eval_csv", "csv"):
            rel = str(format_compare.get(key) or "").strip()
            if not rel:
                continue
            if key == "csv" and test_csv_val and rel == test_csv_val:
                continue
            if rel not in manifest["tables"]:
                manifest["tables"].append(rel)

    manifest_path = os.path.join(session_root, "session.json")
    write_manifest_cb(manifest_path, manifest)

    strict_diag = bool(getattr(args, "strict_diagnostics", False))
    if strict_diag:
        critical_missing = []
        if profile in ("quality", "full") and "metric_sources" not in manifest:
            critical_missing.append("metric_sources")
        if profile == "full":
            pr_meta = manifest.get("pr_per_class") if isinstance(manifest.get("pr_per_class"), dict) else {}
            pr_csv_rel = str((pr_meta or {}).get("csv") or "")
            if not pr_csv_rel:
                critical_missing.append("pr_per_class_csv")
            elif not os.path.isfile(os.path.join(session_root, pr_csv_rel)):
                critical_missing.append("pr_per_class_csv")
        if critical_missing:
            print(
                "[ERROR] Strict diagnostics failed: missing critical artifacts: "
                + ", ".join(critical_missing),
                file=sys.stderr,
            )
            sys.exit(1)

    report_files = write_analysis_report_cb(
        session_root,
        manifest,
        no_pdf=bool(args.no_pdf),
        no_odt=bool(args.no_odt),
        languages=report_languages,
    )
    print(f"[OK] Analyze session: {session_root}")
    print(f"[OK] Manifest: {manifest_path}")
    for key, path in report_files.items():
        print(f"[OK] Report {key}: {path}")

    if replay_parser is not None:
        setattr(args, "baseline", baseline)
        setattr(args, "others", others)
        setattr(args, "profile", profile)
        rl_str = (
            ",".join(report_languages)
            if report_languages
            else str(getattr(args, "report_languages", "ru,en") or "ru,en")
        )
        setattr(args, "report_languages", rl_str)
        setattr(
            args,
            "recompute_missing_metrics_choice",
            "yes" if recompute_missing_metrics else "no",
        )
        if data_yaml:
            setattr(args, "data_yaml", data_yaml)
        cmd = build_non_interactive_command("analyze all", replay_parser, args)
        print_replay_command("after execution", cmd)

