from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from smartrain.cli_argparse import CliArgumentParser
from smartrain.cli_prompts import print_numbered_options, prompt_choice, prompt_yes_no
from smartrain.external_providers.installer import install_provider, providers_root, uninstall_provider
from smartrain.external_providers.probe import probe_provider_repo
from smartrain.external_providers.registry import list_provider_specs
from smartrain.provider_global_index import list_provider_records


def build_providers_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(prog="smartrain providers", description="Manage external providers install/uninstall/status.")
    sub = p.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install", help="Install selected external providers.")
    install.add_argument("--target", type=str, default=None, help="Target directory (default: current directory).")
    install.add_argument("--provider", action="append", default=[], help="Provider id (repeatable).")
    install.add_argument("--all", action="store_true", help="Install all providers.")
    install.add_argument("-y", "--yes", action="store_true", help="Non-interactive mode.")

    uninstall = sub.add_parser("uninstall", help="Uninstall selected external providers.")
    uninstall.add_argument("--target", type=str, default=None, help="Target directory (default: current directory).")
    uninstall.add_argument("--provider", action="append", default=[], help="Provider id (repeatable).")
    uninstall.add_argument("--all", action="store_true", help="Uninstall all providers.")
    uninstall.add_argument("-y", "--yes", action="store_true", help="Skip confirmation.")

    sub.add_parser("status", help="Show known providers and index records.")
    doctor = sub.add_parser("doctor", help="Validate runnable readiness of installed providers.")
    doctor.add_argument("--verbose", action="store_true", help="Print detailed diagnostic reasons.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_providers_arg_parser().parse_args(argv)
    if args.command == "install":
        return _cmd_install(args)
    if args.command == "uninstall":
        return _cmd_uninstall(args)
    if args.command == "status":
        return _cmd_status()
    if args.command == "doctor":
        return _cmd_doctor(args)
    return 1


def _cmd_install(args: argparse.Namespace) -> int:
    try:
        selected = _resolve_selected(args.provider, bool(args.all), interactive=not bool(args.yes), mode="install")
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 2
    if not selected:
        print("[INFO] No providers selected.")
        return 0
    root = providers_root(args.target)
    print(f"[INFO] Install root: {root}")
    installed = skipped = failed = 0
    for pid in selected:
        res = install_provider(pid, args.target)
        if res.action == "installed":
            installed += 1
        elif res.action == "skipped":
            skipped += 1
        else:
            failed += 1
        print(f"[{res.action.upper()}] {pid}: {res.message}")
    print(f"[INFO] Summary: installed={installed}, skipped={skipped}, failed={failed}")
    return 0 if failed == 0 else 1


def _cmd_uninstall(args: argparse.Namespace) -> int:
    try:
        selected = _resolve_selected(args.provider, bool(args.all), interactive=not bool(args.yes), mode="uninstall")
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 2
    if not selected:
        print("[INFO] No providers selected.")
        return 0
    if not args.yes:
        if not prompt_yes_no("Proceed with uninstall selected providers?", default=False):
            print("[INFO] Uninstall cancelled.")
            return 0
    removed = skipped = failed = 0
    for pid in selected:
        res = uninstall_provider(pid, args.target)
        if res.action == "removed":
            removed += 1
        elif res.action == "skipped":
            skipped += 1
        else:
            failed += 1
        print(f"[{res.action.upper()}] {pid}: {res.message}")
    print(f"[INFO] Summary: removed={removed}, skipped={skipped}, failed={failed}")
    return 0 if failed == 0 else 1


def _cmd_status() -> int:
    records = {str(r.get("provider_id", "")): r for r in list_provider_records()}
    specs = list_provider_specs()
    print("[INFO] Providers status:")
    for spec in specs:
        rec = records.get(spec.id)
        state = str(rec.get("install_state", "not_installed")) if rec else "not_installed"
        repo = str(rec.get("repo_path", "")) if rec else "-"
        print(f"  - {spec.id}: {state} | repo={repo}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    records = {str(r.get("provider_id", "")): r for r in list_provider_records()}
    specs = list_provider_specs()
    print("[INFO] Providers doctor:")
    bad = 0
    verbose = bool(getattr(args, "verbose", False))
    for spec in specs:
        rec = records.get(spec.id)
        if not rec:
            print(f"  - {spec.id}: not_installed")
            if verbose:
                print("      reason: no record in global providers index")
            continue
        repo = str(rec.get("repo_path", ""))
        venv_path = str(rec.get("venv_path", ""))
        state = str(rec.get("install_state", "unknown"))
        last_error = str(rec.get("last_error", "") or "")
        probe = probe_provider_repo(repo, spec, venv_path)
        repo_ok = bool(probe.get("repo_found"))
        entry_ok = bool(probe.get("entrypoints_ok"))
        train_ok = bool(probe.get("train_entry_ok"))
        infer_ok = bool(probe.get("infer_entry_ok"))
        req_ok = bool(probe.get("requirements_ok"))
        venv_ok = bool(probe.get("venv_ready"))
        venv_python = str(probe.get("venv_python") or "")
        runtime_ok = bool(probe.get("runtime_ok", True))
        runtime_reason = str(probe.get("runtime_reason", "") or "")
        status = "ok" if (state == "installed" and repo_ok and entry_ok and venv_ok and runtime_ok) else "broken"
        if status != "ok":
            bad += 1
        print(
            f"  - {spec.id}: {status} | state={state} | repo={repo_ok} | entrypoints={entry_ok} | venv={venv_ok} | runtime={runtime_ok}"
        )
        if verbose:
            print(f"      repo_path: {repo or '-'}")
            print(f"      venv_path: {venv_path or '-'}")
            print(f"      venv_python: {venv_python or '-'}")
            print(
                "      checks: "
                f"train_entry={train_ok}, infer_entry={infer_ok}, requirements={req_ok}, runtime={runtime_ok}, ready_flag={spec.ready}"
            )
            if runtime_reason:
                print(f"      runtime_reason: {runtime_reason}")
            if not spec.ready and spec.note:
                print(f"      note: {spec.note}")
            if last_error:
                print(f"      last_error: {last_error}")
    return 0 if bad == 0 else 1


def _resolve_selected(raw: list[str], all_flag: bool, *, interactive: bool, mode: str) -> list[str]:
    specs = list_provider_specs()
    ids = [s.id for s in specs]
    if all_flag:
        return ids
    explicit = [str(x).strip().lower() for x in (raw or []) if str(x).strip()]
    if explicit:
        unknown = [x for x in explicit if x not in ids]
        if unknown:
            raise ValueError(f"Unknown provider ids: {', '.join(unknown)}")
        return explicit
    if not interactive or not sys.stdin.isatty():
        return []
    opts = [*ids, "all"]
    print_numbered_options(f"Providers to {mode}", opts)
    chosen = prompt_choice("Provider", opts, default="all", show_options=False)
    if chosen == "all":
        return ids
    return [chosen]


if __name__ == "__main__":
    raise SystemExit(main())

