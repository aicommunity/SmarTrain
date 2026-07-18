from __future__ import annotations

import argparse
import os
from typing import Any, Callable

from smartrain.core.runtime.path_portable import posix_relpath


def run_all_baseline_artifacts(
    *,
    baseline: str,
    others: list[str],
    session_root: str,
    workspace: str | None,
    analytics_session: str | None,
    models_root: str | None,
    default_map_col: str,
    cmd_compare_cb: Callable[[argparse.Namespace], None],
    cmd_export_table_cb: Callable[[argparse.Namespace], None],
    write_system_profile_compare_csv_cb: Callable[[list[str], str], bool],
    write_test_system_profile_compare_csv_cb: Callable[[list[str], str], bool],
) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []

    if others:
        compare_csv = os.path.join(session_root, "artifacts", "compare", "compare_delta.csv")
        compare_png = os.path.join(session_root, "artifacts", "compare", "compare_curves.png")
        compare_insights = os.path.join(session_root, "artifacts", "compare", "compare_insights.txt")
        cmp_ns = argparse.Namespace(
            baseline=baseline,
            others=others,
            out_csv=compare_csv,
            out_png=compare_png,
            out_insights=compare_insights,
            metric_column=default_map_col,
            workspace=workspace,
            analytics_session=analytics_session,
            models_root=models_root,
        )
        cmd_compare_cb(cmp_ns)
        artifacts.extend(
            [
                {"role": "compare_csv", "path": posix_relpath(compare_csv, session_root)},
                {"role": "compare_png", "path": posix_relpath(compare_png, session_root)},
                {"role": "compare_insights", "path": posix_relpath(compare_insights, session_root)},
            ]
        )
    else:
        print("[INFO] No candidate runs selected: compare artifacts are skipped (single-run report mode).")

    exp_csv = os.path.join(session_root, "artifacts", "table", "runs_summary.csv")
    exp_ns = argparse.Namespace(
        output=exp_csv,
        workspace=workspace,
        models_root=models_root,
        analytics_session=None,
    )
    cmd_export_table_cb(exp_ns)
    artifacts.append({"role": "summary_csv", "path": posix_relpath(exp_csv, session_root)})

    sys_profile_csv = os.path.join(session_root, "artifacts", "table", "system_profile_compare.csv")
    written_sys_profile = write_system_profile_compare_csv_cb([baseline] + others, sys_profile_csv)
    if written_sys_profile:
        artifacts.append(
            {"role": "system_profile_compare_csv", "path": posix_relpath(sys_profile_csv, session_root)}
        )

    test_sys_profile_csv = os.path.join(session_root, "artifacts", "table", "test_system_profile_compare.csv")
    written_test_sys_profile = write_test_system_profile_compare_csv_cb([baseline] + others, test_sys_profile_csv)
    if written_test_sys_profile:
        artifacts.append(
            {"role": "test_system_profile_compare_csv", "path": posix_relpath(test_sys_profile_csv, session_root)}
        )

    return artifacts

