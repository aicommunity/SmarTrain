"""Train resume and calc-confidence CLI commands."""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

from smartrain.cli_entrypoints.support.cli_prompts import print_numbered_options, prompt_text
from smartrain.core.runtime.run_discovery import find_run_directories
from smartrain.core.runtime.workspace_paths import WorkspaceLayout, resolve_workspace_root
from smartrain.core.training.confidence_recommendation import (
    recommendation_file_path,
    recommendations_complete,
    read_recommendation_file,
)
from smartrain.services.training.train_cli_parsers import (
    build_train_calc_confidence_arg_parser,
    build_train_resume_arg_parser,
)


def resume_display_value(diag: Any) -> str:
    dataset_name = os.path.basename(os.path.dirname(diag.run_dir.rstrip(os.sep)))
    run_name = os.path.basename(diag.run_dir.rstrip(os.sep))
    reason = diag.reasons[0] if diag.reasons else "n/a"
    return f"{dataset_name}/{run_name} | {diag.status} | {reason}"


def select_resume_candidate_interactive(
    candidates: list[Any],
    *,
    prompt_text_cb: Callable[..., str] = prompt_text,
) -> Any | None:
    if not candidates:
        print("[ERROR] No incomplete runs found.")
        return None
    options = [resume_display_value(d) for d in candidates]
    print_numbered_options("Incomplete run to resume", options)
    while True:
        raw = prompt_text_cb("Choose run number", default="1").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
        print(f"[ERROR] Incorrect selection: {raw!r}")


def select_runs_for_calc_confidence_interactive(
    run_dirs: list[str],
    *,
    prompt_text_cb: Callable[..., str] = prompt_text,
) -> list[str]:
    if not run_dirs:
        return []
    print("\n[INFO] Available runs:")
    for idx, rd in enumerate(run_dirs, start=1):
        print(f"  {idx}. {rd}")
    raw = prompt_text_cb(
        "Select runs by numbers (comma-separated) or 'all'",
        default="all",
    ).strip()
    if not raw or raw.lower() == "all":
        return run_dirs
    selected: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        if not t.isdigit():
            raise ValueError(f"Invalid token {t!r}. Expected numbers or 'all'.")
        pos = int(t)
        if pos < 1 or pos > len(run_dirs):
            raise ValueError(f"Selection {pos} out of range 1..{len(run_dirs)}.")
        item = run_dirs[pos - 1]
        if item not in seen:
            seen.add(item)
            selected.append(item)
    return selected


def resolve_run_dirs_for_calc_confidence(
    workspace_root: str,
    run_dir_args: list[str],
    select_all: bool,
    non_interactive: bool,
) -> list[str]:
    discovered = sorted(set(os.path.abspath(x) for x in find_run_directories(WorkspaceLayout(workspace_root).runs)))
    if run_dir_args:
        out: list[str] = []
        for raw in run_dir_args:
            rd = str(raw).strip()
            if not rd:
                continue
            if not os.path.isabs(rd):
                rd = os.path.join(WorkspaceLayout(workspace_root).runs, rd)
            out.append(os.path.abspath(rd))
        return sorted(set(out))
    if select_all or non_interactive:
        return discovered
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive mode requires a terminal (TTY).")
    return select_runs_for_calc_confidence_interactive(discovered)


def run_calc_confidence_command(
    argv: list[str],
    *,
    ensure_resume_confidence_recommendations_cb: Callable[..., None],
) -> int:
    args = build_train_calc_confidence_arg_parser().parse_args(argv)
    try:
        workspace_root = resolve_workspace_root(args.workspace)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1
    try:
        run_dirs = resolve_run_dirs_for_calc_confidence(
            workspace_root=workspace_root,
            run_dir_args=list(getattr(args, "run_dir", []) or []),
            select_all=bool(getattr(args, "all", False)),
            non_interactive=bool(getattr(args, "non_interactive", False)),
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2
    if not run_dirs:
        print("[INFO] No runs selected.")
        return 0
    val_batch = getattr(args, "val_batch", None)
    if val_batch is None:
        if bool(getattr(args, "non_interactive", False)) or not sys.stdin.isatty():
            val_batch = 1
        else:
            raw_batch = prompt_text(
                "Val/Test batch for confidence recompute",
                default="1",
            ).strip()
            try:
                val_batch = int(raw_batch) if raw_batch else 1
            except ValueError:
                print(f"[ERROR] Invalid --val-batch value: {raw_batch!r}")
                return 2
    if int(val_batch) <= 0:
        print(f"[ERROR] --val-batch must be > 0, got: {val_batch}")
        return 2

    processed = 0
    updated = 0
    skipped = 0
    failed = 0
    for run_dir in run_dirs:
        processed += 1
        before_test = recommendations_complete(
            read_recommendation_file(recommendation_file_path(run_dir, "test"))
        )
        before_val = recommendations_complete(
            read_recommendation_file(recommendation_file_path(run_dir, "val"))
        )
        try:
            ensure_resume_confidence_recommendations_cb(run_dir, workspace_root, val_batch=int(val_batch))
        except Exception as e:
            failed += 1
            print(f"[ERROR] {run_dir}: {e}")
            continue
        after_test = recommendations_complete(
            read_recommendation_file(recommendation_file_path(run_dir, "test"))
        )
        after_val = recommendations_complete(
            read_recommendation_file(recommendation_file_path(run_dir, "val"))
        )
        if after_test and after_val and (not (before_test and before_val)):
            updated += 1
            print(f"[OK] {run_dir}: confidence recommendations computed.")
        elif before_test and before_val:
            skipped += 1
            print(f"[INFO] {run_dir}: recommendations already present.")
        else:
            skipped += 1
            print(f"[WARN] {run_dir}: recommendations still incomplete.")

    print(
        f"[INFO] calc-confidence summary: processed={processed}, "
        f"updated={updated}, skipped={skipped}, failed={failed}"
    )
    return 1 if failed > 0 else 0


def run_resume_command(
    argv: list[str],
    *,
    run_status_resumable_incomplete: str,
    run_status_training_complete_test_pending: str,
    list_incomplete_runs_cb: Callable[[str], list[Any]],
    diagnose_run_cb: Callable[[str], Any],
    resume_training_in_run_cb: Callable[[str], None],
    update_resume_metadata_cb: Callable[..., None],
    update_resume_test_metadata_cb: Callable[..., None],
    complete_missing_test_with_backoff_cb: Callable[..., None],
    ensure_resume_confidence_recommendations_cb: Callable[..., None],
    ensure_matplotlib_training_runtime_cb: Callable[..., Any],
    maybe_free_cuda_memory_cb: Callable[[], None],
) -> int:
    args = build_train_resume_arg_parser().parse_args(argv)
    try:
        workspace_root = resolve_workspace_root(args.workspace)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1

    if args.non_interactive and not args.run_dir:
        print("[ERROR] In non-interactive mode, --run-dir is required.")
        return 2
    if args.test_batch is not None and int(args.test_batch) <= 0:
        print(f"[ERROR] --test-batch must be > 0, got: {args.test_batch}")
        return 2
    if int(args.test_batch_min) <= 0:
        print(f"[ERROR] --test-batch-min must be > 0, got: {args.test_batch_min}")
        return 2
    if int(args.test_batch_backoff) <= 1:
        print(f"[ERROR] --test-batch-backoff must be > 1, got: {args.test_batch_backoff}")
        return 2

    candidates = list_incomplete_runs_cb(workspace_root)
    chosen: Any | None = None
    if args.run_dir:
        run_dir = args.run_dir
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(WorkspaceLayout(workspace_root).runs, run_dir)
        chosen = next((d for d in candidates if os.path.abspath(run_dir) == d.run_dir), None)
        if chosen is None:
            chosen = diagnose_run_cb(run_dir)
    else:
        if not sys.stdin.isatty():
            print("[ERROR] Interactive resume mode requires a terminal (TTY).")
            return 1
        chosen = select_resume_candidate_interactive(candidates)
        if chosen is None:
            return 1

    if chosen.status not in (
        run_status_resumable_incomplete,
        run_status_training_complete_test_pending,
    ):
        print(f"[ERROR] Run is not resumable: {chosen.run_dir}")
        print(f"[INFO] Status: {chosen.status}")
        print(f"[INFO] Reasons: {', '.join(chosen.reasons)}")
        return 2

    if chosen.status == run_status_training_complete_test_pending:
        try:
            ensure_matplotlib_training_runtime_cb(non_interactive=True)
            maybe_free_cuda_memory_cb()
            complete_missing_test_with_backoff_cb(
                chosen.run_dir,
                workspace_root=workspace_root,
                initial_batch=args.test_batch,
                min_batch=int(args.test_batch_min),
                backoff=int(args.test_batch_backoff),
            )
            print(f"[OK] Missing test stage completed: {chosen.run_dir}")
            return 0
        except Exception as e:
            update_resume_test_metadata_cb(
                chosen.run_dir,
                success=False,
                error=str(e),
                diagnosis=diagnose_run_cb(chosen.run_dir),
            )
            print(f"[ERROR] Failed to complete missing test stage: {chosen.run_dir}")
            print(f"[ERROR] {e}")
            return 1

    try:
        resume_training_in_run_cb(chosen.run_dir)
        ensure_resume_confidence_recommendations_cb(chosen.run_dir, workspace_root)
        update_resume_metadata_cb(
            chosen.run_dir,
            success=True,
            error=None,
            diagnosis=diagnose_run_cb(chosen.run_dir),
        )
        print(f"[OK] Resume completed: {chosen.run_dir}")
        return 0
    except Exception as e:
        update_resume_metadata_cb(
            chosen.run_dir,
            success=False,
            error=str(e),
            diagnosis=diagnose_run_cb(chosen.run_dir),
        )
        print(f"[ERROR] Failed to resume run: {chosen.run_dir}")
        print(f"[ERROR] {e}")
        return 1
