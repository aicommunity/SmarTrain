from __future__ import annotations

import os
from typing import Any, Callable


def run_all_command(
    *,
    args: Any,
    prepare_all_selection_cb: Callable[..., tuple[str, list[str], str, bool]],
    resolve_all_data_yaml_context_cb: Callable[..., tuple[list[str], str, list[str], dict[str, str], list[str]]],
    session_root_cb: Callable[[str | None, str | None], str],
    filtered_run_records_cb: Callable[..., Any],
    prompt_int_cb: Callable[..., Any],
    prompt_text_cb: Callable[..., Any],
    prompt_choice_cb: Callable[..., Any],
    build_run_data_yaml_map_cb: Callable[..., Any],
    auto_select_data_yaml_cb: Callable[..., Any],
    run_all_baseline_artifacts_cb: Callable[..., list[dict[str, str]]],
    run_all_quality_stage_cb: Callable[..., tuple[list[dict[str, str]], dict[str, Any], bool]],
    run_all_leaderboard_stage_cb: Callable[..., list[dict[str, str]]],
    run_all_speed_stage_cb: Callable[..., tuple[list[dict[str, str]], list[dict[str, Any]]]],
    run_all_pr_stage_cb: Callable[..., tuple[list[dict[str, str]], list[dict[str, Any]]]],
    finalize_all_session_cb: Callable[..., None],
    default_map_col: str,
    cmd_compare_cb: Callable[..., Any],
    cmd_export_table_cb: Callable[..., Any],
    write_system_profile_compare_csv_cb: Callable[..., Any],
    write_test_system_profile_compare_csv_cb: Callable[..., Any],
    cmd_leaderboard_cb: Callable[..., Any],
    collect_missing_metrics_recompute_plan_cb: Callable[..., Any],
    cmd_test_metrics_plot_cb: Callable[..., Any],
    group_runs_by_data_yaml_cb: Callable[..., Any],
    cmd_inference_benchmark_cb: Callable[..., Any],
    cmd_inference_plot_cb: Callable[..., Any],
    write_speed_quality_artifacts_cb: Callable[..., Any],
    cmd_pr_curves_cb: Callable[..., Any],
    safe_name_cb: Callable[..., Any],
    build_abbreviations_for_report_cb: Callable[..., Any],
    build_report_manifest_labels_cb: Callable[..., Any] | None = None,
    collect_ultralytics_test_artifacts_cb: Callable[..., Any],
    write_format_compare_artifacts_cb: Callable[..., Any],
    collect_confidence_recommendation_tables_cb: Callable[..., Any],
    write_manifest_cb: Callable[..., Any],
    write_analysis_report_cb: Callable[..., Any],
) -> None:
    baseline, others, profile, selection_prompts_used = prepare_all_selection_cb(
        args,
        filtered_run_records_cb=filtered_run_records_cb,
        prompt_int_cb=prompt_int_cb,
        prompt_text_cb=prompt_text_cb,
        prompt_choice_cb=prompt_choice_cb,
    )
    report_languages, data_yaml, selected_run_dirs, run_data_yaml_map, unresolved_data_yaml_runs = (
        resolve_all_data_yaml_context_cb(
            args=args,
            baseline=baseline,
            others=others,
            profile=profile,
            selection_prompts_used=selection_prompts_used,
            build_run_data_yaml_map_cb=build_run_data_yaml_map_cb,
            auto_select_data_yaml_cb=auto_select_data_yaml_cb,
            prompt_choice_cb=prompt_choice_cb,
            prompt_text_cb=prompt_text_cb,
        )
    )
    session_root = session_root_cb(args.workspace, args.analytics_session)
    artifacts: list[dict[str, str]] = []
    cache_events: list[dict[str, Any]] = []
    artifact_failures: list[dict[str, Any]] = []

    def _record_failure(
        *,
        stage: str,
        status: str,
        reason_code: str,
        reason_detail: str = "",
        run_dir: str | None = None,
        format_name: str | None = None,
        split: str | None = None,
    ) -> None:
        artifact_failures.append(
            {
                "stage": stage,
                "status": status,
                "reason_code": reason_code,
                "reason_detail": reason_detail,
                "run_dir": run_dir or "",
                "format": format_name or "",
                "split": split or "",
            }
        )

    selected_labels = [os.path.basename(x.rstrip(os.sep)) for x in selected_run_dirs]
    if not others:
        print("[INFO] Single-run report mode: cross-run compare artifacts will be skipped.")
    print("[INFO] Selected compare runs:")
    for idx, (run_dir, label) in enumerate(zip(selected_run_dirs, selected_labels), start=1):
        role = "baseline" if idx == 1 else "other"
        print(f"[INFO]  - {role}: {label} ({run_dir})")

    baseline_artifacts = run_all_baseline_artifacts_cb(
        baseline=baseline,
        others=others,
        session_root=session_root,
        workspace=args.workspace,
        analytics_session=args.analytics_session,
        models_root=args.models_root,
        default_map_col=default_map_col,
        cmd_compare_cb=cmd_compare_cb,
        cmd_export_table_cb=cmd_export_table_cb,
        write_system_profile_compare_csv_cb=write_system_profile_compare_csv_cb,
        write_test_system_profile_compare_csv_cb=write_test_system_profile_compare_csv_cb,
    )
    artifacts.extend(baseline_artifacts)

    runs_group_dir = os.path.dirname(baseline)
    quality_artifacts, metric_sources_payload, recompute_missing_metrics = run_all_quality_stage_cb(
        args=args,
        profile=profile,
        baseline=baseline,
        others=others,
        selected_run_dirs=selected_run_dirs,
        session_root=session_root,
        runs_group_dir=runs_group_dir,
        data_yaml=data_yaml,
        run_data_yaml_map=run_data_yaml_map,
        collect_missing_metrics_recompute_plan_cb=collect_missing_metrics_recompute_plan_cb,
        cmd_test_metrics_plot_cb=cmd_test_metrics_plot_cb,
        refresh_runs_summary_cb=cmd_export_table_cb,
    )
    artifacts.extend(quality_artifacts)

    leaderboard_artifacts = run_all_leaderboard_stage_cb(
        selected_run_dirs=selected_run_dirs,
        session_root=session_root,
        workspace=args.workspace,
        analytics_session=args.analytics_session,
        models_root=args.models_root,
        cmd_leaderboard_cb=cmd_leaderboard_cb,
        record_failure_cb=_record_failure,
    )
    artifacts.extend(leaderboard_artifacts)

    speed_artifacts, speed_cache_events = run_all_speed_stage_cb(
        args=args,
        profile=profile,
        baseline=baseline,
        others=others,
        selected_run_dirs=selected_run_dirs,
        session_root=session_root,
        runs_group_dir=runs_group_dir,
        run_data_yaml_map=run_data_yaml_map,
        metric_sources_payload=metric_sources_payload,
        record_failure_cb=_record_failure,
        group_runs_by_data_yaml_cb=group_runs_by_data_yaml_cb,
        cmd_inference_benchmark_cb=cmd_inference_benchmark_cb,
        cmd_inference_plot_cb=cmd_inference_plot_cb,
        write_speed_quality_artifacts_cb=write_speed_quality_artifacts_cb,
        build_run_display_labels_cb=build_abbreviations_for_report_cb,
    )
    artifacts.extend(speed_artifacts)
    cache_events.extend(speed_cache_events)

    pr_artifacts, pr_cache_events = run_all_pr_stage_cb(
        args=args,
        profile=profile,
        selected_run_dirs=selected_run_dirs,
        session_root=session_root,
        runs_group_dir=runs_group_dir,
        run_data_yaml_map=run_data_yaml_map,
        record_failure_cb=_record_failure,
        group_runs_by_data_yaml_cb=group_runs_by_data_yaml_cb,
        cmd_pr_curves_cb=cmd_pr_curves_cb,
        safe_name_cb=safe_name_cb,
    )
    artifacts.extend(pr_artifacts)
    cache_events.extend(pr_cache_events)

    finalize_all_session_cb(
        args=args,
        session_root=session_root,
        profile=profile,
        baseline=baseline,
        others=others,
        data_yaml=data_yaml,
        report_languages=report_languages,
        run_data_yaml_map=run_data_yaml_map,
        unresolved_data_yaml_runs=unresolved_data_yaml_runs,
        artifacts=artifacts,
        cache_events=cache_events,
        artifact_failures=artifact_failures,
        metric_sources_payload=metric_sources_payload,
        recompute_missing_metrics=recompute_missing_metrics,
        build_abbreviations_for_report_cb=build_abbreviations_for_report_cb,
        build_report_manifest_labels_cb=build_report_manifest_labels_cb,
        collect_ultralytics_test_artifacts_cb=collect_ultralytics_test_artifacts_cb,
        write_format_compare_artifacts_cb=write_format_compare_artifacts_cb,
        collect_confidence_recommendation_tables_cb=collect_confidence_recommendation_tables_cb,
        write_manifest_cb=write_manifest_cb,
        write_analysis_report_cb=write_analysis_report_cb,
        record_failure_cb=_record_failure,
    )
