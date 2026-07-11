"""CLI: list active workspace peers and locks."""

from __future__ import annotations

import argparse
import json
import sys

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.core.runtime.workspace_coordination import list_active_locks, list_active_peers
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root


def build_workspace_peers_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Show active smartrain sessions and workspace locks.")
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR})")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return p


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = build_workspace_peers_arg_parser().parse_args(argv)
    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    peers = list_active_peers(layout)
    locks = list_active_locks(layout)
    if args.json:
        print(json.dumps({"peers": peers, "locks": locks}, ensure_ascii=False, indent=2))
        return
    print(f"Workspace: {layout.root}")
    print(f"Active sessions: {len(peers)}")
    for peer in peers:
        host = peer.get("hostname", "?")
        user = peer.get("user", "?")
        cmd = peer.get("command", "?")
        started = peer.get("started_at", "?")
        print(f"  - {host} ({user}) since {started}: {cmd}")
    print(f"Active locks: {len(locks)}")
    for lock in locks:
        name = lock.get("_lock_name", lock.get("lock_name", "?"))
        host = lock.get("hostname", "?")
        cmd = lock.get("command", "?")
        print(f"  - {name}: {host} ({cmd})")


__all__ = ["build_workspace_peers_arg_parser", "main"]
