from __future__ import annotations

import os
import sys
from typing import Any, Callable


def prepare_all_selection(
    args: Any,
    *,
    filtered_run_records_cb: Callable[[Any], list[tuple[str, Any]]],
    prompt_int_cb: Callable[..., int],
    prompt_text_cb: Callable[..., str],
    prompt_choice_cb: Callable[..., str],
) -> tuple[str, list[str], str, bool]:
    interactive_mode = sys.stdin.isatty()
    baseline = str(getattr(args, "baseline", "") or "").strip()
    others = [str(x).strip() for x in (getattr(args, "others", []) or []) if str(x).strip()]
    profile = str(getattr(args, "profile", "") or "").strip().lower()
    if not baseline or profile not in {"quality", "speed", "full"}:
        if not interactive_mode:
            print(
                "[ERROR] Non-interactive `smartrain analyze all` requires --baseline and --profile.",
                file=sys.stderr,
            )
            sys.exit(2)
        indexed = filtered_run_records_cb(args)
        if len(indexed) < 2:
            print("[ERROR] Need at least two runs for full analysis.")
            sys.exit(1)
        runs_root = os.path.abspath(str(args.models_root))

        def _display_run_dir(path: str) -> str:
            ap = os.path.abspath(path)
            try:
                rel = os.path.relpath(ap, runs_root)
                if not rel.startswith(".."):
                    return rel
            except Exception:
                pass
            return ap

        print(f"{'#':>4}  {'model':<14}  {'dataset':<24}  {'run_dir (relative to runs root)'}")
        print("-" * 120)
        for i, (rd, rec) in enumerate(indexed, start=1):
            print(
                f"{i:4d}  {str(rec.model or '?')[:14]:<14}  "
                + f"{str(rec.dataset_name or '?')[:24]:<24}  {_display_run_dir(rd)}"
            )
        baseline_idx = prompt_int_cb("Baseline run number", default=1)
        others_raw = prompt_text_cb("Other run numbers (comma-separated)", default="").strip()
        try:
            others_idx = [int(x.strip()) for x in others_raw.split(",") if x.strip()]
        except ValueError:
            print("[ERROR] Invalid run numbers.")
            sys.exit(1)
        if baseline_idx < 1 or baseline_idx > len(indexed):
            print("[ERROR] Baseline index out of range.")
            sys.exit(1)
        baseline = indexed[baseline_idx - 1][0]
        others = [indexed[i - 1][0] for i in others_idx if 1 <= i <= len(indexed) and indexed[i - 1][0] != baseline]
        profile = prompt_choice_cb("Profile", ["quality", "speed", "full"], default="full")
    return baseline, others, profile, interactive_mode

