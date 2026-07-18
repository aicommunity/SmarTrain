"""``smartrain update`` — scan/apply workspace legacy → canonical migration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import prompt_yes_no
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.services.update.appliers import apply_plan
from smartrain.services.update.plan import UpdateRisk, UpdateStatus, parse_categories
from smartrain.services.update.scanner import residual_after, scan_workspace


def build_update_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description=(
            "Scan workspace for legacy on-disk layouts and migrate toward the canonical shape "
            "(empty call starts interactive mode)"
        )
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without applying changes",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Apply safe steps without confirmation; ask-steps skipped unless --apply-all",
    )
    p.add_argument(
        "--apply-all",
        action="store_true",
        help="Also apply ask/unsafe steps without prompts (requires --yes)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if residual legacy findings remain (no apply)",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated categories: layout,weights,releases,manifest,tests,metadata,yaml",
    )
    p.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Override report output directory (default: analytics/update-reports/<timestamp>/)",
    )
    return p


def _print_plan(steps) -> None:
    if not steps:
        print("[OK] No legacy findings.")
        return
    print(f"[INFO] Planned steps: {len(steps)}")
    for i, step in enumerate(steps, start=1):
        print(f"  {i}. [{step.risk.value}] {step.category.value}: {step.title}")
        if step.detail:
            print(f"      {step.detail}")


def _write_report(report_dir: Path, plan_dict: dict, result_dict: dict | None) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "plan.json").write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    if result_dict is not None:
        (report_dir / "result.json").write_text(
            json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    parser = build_update_arg_parser()
    args = parser.parse_args(argv)

    if args.apply_all and not args.yes:
        parser.error("--apply-all requires --yes")

    workspace_root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(workspace_root)
    categories = parse_categories(args.only)

    plan = scan_workspace(layout).filtered(categories)
    _print_plan(plan.steps)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.report_dir:
        report_dir = Path(args.report_dir).expanduser().resolve()
    else:
        report_dir = Path(layout.root) / "analytics" / "update-reports" / ts

    if args.check:
        residual = residual_after(layout, categories)
        _write_report(report_dir, residual.to_dict(), None)
        print(f"[OK] Report: {report_dir / 'plan.json'}")
        if residual.steps:
            print(f"[ERROR] Residual legacy findings: {len(residual.steps)}", file=sys.stderr)
            raise SystemExit(1)
        print("[OK] No residual legacy findings.")
        raise SystemExit(0)

    interactive = is_interactive_allowed(argv) and len(argv) == 0 and sys.stdin.isatty()
    dry_run = bool(args.dry_run)
    if interactive and not args.yes and not dry_run:
        if plan.steps and not prompt_yes_no("Apply safe migration steps?", default=True):
            print("[INFO] Cancelled.")
            _write_report(report_dir, plan.to_dict(), None)
            raise SystemExit(0)
        dry_run = False

    if not interactive and not args.yes and not dry_run and not args.check:
        # Non-interactive without flags → dry-run by default when argv empty was not tty
        if len(argv) == 0:
            dry_run = True
        else:
            parser.error("use --dry-run, --yes, or --check (or run without args in a TTY)")

    def ask_cb(step) -> bool:
        if not interactive:
            return False
        return prompt_yes_no(f"Apply unsafe step: {step.title}?", default=False)

    include_ask = bool(args.apply_all)
    results = apply_plan(
        layout,
        plan.steps,
        dry_run=dry_run,
        include_ask=include_ask,
        ask_callback=ask_cb if interactive and not include_ask else None,
    )

    result_payload = {
        "dry_run": dry_run,
        "steps": [s.to_dict() for s in results],
        "applied": sum(1 for s in results if s.status == UpdateStatus.APPLIED),
        "skipped": sum(1 for s in results if s.status == UpdateStatus.SKIPPED),
        "failed": sum(1 for s in results if s.status == UpdateStatus.FAILED),
        "dry_run_count": sum(1 for s in results if s.status == UpdateStatus.DRY_RUN),
    }
    residual = residual_after(layout, categories)
    result_payload["residual_count"] = len(residual.steps)
    plan.residual = residual.steps
    _write_report(report_dir, plan.to_dict(), result_payload)

    print(f"[OK] Report: {report_dir}")
    print(
        f"[OK] applied={result_payload['applied']} skipped={result_payload['skipped']} "
        f"failed={result_payload['failed']} residual={result_payload['residual_count']}"
    )
    if result_payload["failed"]:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
