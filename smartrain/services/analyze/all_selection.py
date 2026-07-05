from __future__ import annotations

import os
import sys
from typing import Any, Callable

from smartrain.core.runtime.workspace_paths import resolve_workspace_root


def prepare_all_selection(
    args: Any,
    *,
    filtered_run_records_cb: Callable[[Any], list[tuple[str, Any]]],
    prompt_int_cb: Callable[..., int],
    prompt_text_cb: Callable[..., str],
    prompt_choice_cb: Callable[..., str],
) -> tuple[str, list[str], str, bool]:
    """Fourth return value is True only when baseline/profile were chosen via interactive prompts.

    It must **not** be equated with ``sys.stdin.isatty()``: a TTY plus a complete CLI (baseline,
    profile, others) must not trigger secondary prompts such as data.yaml mode.
    """
    tty = sys.stdin.isatty()
    baseline = str(getattr(args, "baseline", "") or "").strip()
    others = [str(x).strip() for x in (getattr(args, "others", []) or []) if str(x).strip()]
    profile = str(getattr(args, "profile", "") or "").strip().lower()
    selection_prompts_used = False
    if not baseline or profile not in {"quality", "speed", "full"}:
        if not tty:
            print(
                "[ERROR] Non-interactive `smartrain analyze all` requires --baseline and --profile.",
                file=sys.stderr,
            )
            sys.exit(2)
        selection_prompts_used = True
        indexed = filtered_run_records_cb(args)
        if len(indexed) < 1:
            print("[ERROR] Need at least one run or promoted model for full analysis.")
            sys.exit(1)
        workspace_root: str | None = None
        try:
            workspace_root = resolve_workspace_root(getattr(args, "workspace", None))
        except ValueError:
            workspace_root = None

        def _source_label(path: str) -> str:
            parts = [p.lower() for p in os.path.abspath(path).split(os.sep)]
            if "models" in parts:
                return "models"
            if "runs" in parts:
                return "runs"
            return "?"

        def _display_target_dir(path: str) -> str:
            ap = os.path.abspath(path)
            if workspace_root:
                try:
                    rel = os.path.relpath(ap, workspace_root)
                    if not rel.startswith(".."):
                        return rel
                except Exception:
                    pass
            return ap

        print(f"{'#':>4}  {'src':<7}  {'model':<14}  {'dataset':<24}  {'path (relative to workspace)'}")
        print("-" * 130)
        for i, (rd, rec) in enumerate(indexed, start=1):
            print(
                f"{i:4d}  {_source_label(rd):<7}  {str(rec.model or '?')[:14]:<14}  "
                + f"{str(rec.dataset_name or '?')[:24]:<24}  {_display_target_dir(rd)}"
            )
        if len(indexed) == 1:
            baseline = indexed[0][0]
            others = []
            print("[INFO] Single target in workspace: using it as baseline (baseline-only report).")
        else:
            baseline_idx = prompt_int_cb("Baseline run number", default=1)
            others_raw = prompt_text_cb(
                "Other run numbers (comma-separated, leave empty for baseline-only report)",
                default="",
            ).strip()
            try:
                others_idx = [int(x.strip()) for x in others_raw.split(",") if x.strip()]
            except ValueError:
                print("[ERROR] Invalid run numbers.")
                sys.exit(1)
            if baseline_idx < 1 or baseline_idx > len(indexed):
                print("[ERROR] Baseline index out of range.")
                sys.exit(1)
            baseline = indexed[baseline_idx - 1][0]
            others = [
                indexed[i - 1][0]
                for i in others_idx
                if 1 <= i <= len(indexed) and indexed[i - 1][0] != baseline
            ]
        profile = prompt_choice_cb("Profile", ["quality", "speed", "full"], default="full")
        # So replay (`build_non_interactive_command`) sees the resolved selection.
        setattr(args, "baseline", baseline)
        setattr(args, "others", others)
        setattr(args, "profile", profile)
    return baseline, others, profile, selection_prompts_used
