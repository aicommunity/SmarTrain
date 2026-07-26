from __future__ import annotations

import argparse
import os
from typing import Callable

from smartrain.core.runtime.path_portable import posix_relpath


def run_all_leaderboard_stage(
    *,
    selected_run_dirs: list[str],
    session_root: str,
    workspace: str | None,
    analytics_session: str | None,
    models_root: str | None,
    cmd_leaderboard_cb: Callable[[argparse.Namespace], None],
    record_failure_cb: Callable[..., None] | None = None,
) -> list[dict[str, str]]:
    """Build leaderboard after quality metrics are available (and before speed merge)."""
    artifacts: list[dict[str, str]] = []
    lb_csv = os.path.join(session_root, "artifacts", "leaderboard", "leaderboard.csv")
    lb_ns = argparse.Namespace(
        out_csv=lb_csv,
        selected_run_dirs=selected_run_dirs,
        quality_metric="mAP50-95",
        speed_metric="avg_inference_fps",
        weight_quality=0.6,
        weight_speed=0.25,
        weight_stability=0.15,
        workspace=workspace,
        models_root=models_root,
        analytics_session=analytics_session,
        soft_fail=True,
    )
    cmd_leaderboard_cb(lb_ns)
    if os.path.isfile(lb_csv):
        artifacts.append({"role": "leaderboard_csv", "path": posix_relpath(lb_csv, session_root)})
    elif record_failure_cb is not None:
        record_failure_cb(
            stage="leaderboard",
            status="missing",
            reason_code="leaderboard_no_metrics",
            reason_detail="leaderboard.csv was not produced (insufficient quality/speed metrics)",
            split="test",
        )
    return artifacts
