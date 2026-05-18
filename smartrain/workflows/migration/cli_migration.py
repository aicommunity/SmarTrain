from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from smartrain.run_model_contract.io.legacy import map_legacy_payload, read_legacy_target
from smartrain.run_model_contract.io.read.resolvers import infer_source_kind
from smartrain.run_model_contract.io.write.layout import unified_snapshot_dir
from smartrain.run_model_contract.io.write.writer import write_unified_snapshot
from smartrain.core.runtime.run_discovery import find_run_directories
from smartrain.core.runtime.workspace_paths import WorkspaceLayout, resolve_workspace_root

Mode = Literal["dry-run", "apply", "report-only"]
Kind = Literal["run", "model"]


@dataclass(frozen=True)
class MigrationItem:
    ref: str
    source_kind: Kind
    status: str
    reason: str | None = None
    payload_hash_sha256: str | None = None
    snapshot_path: str | None = None
    rollback_hint: str | None = None


def _payload_hash(payload: Any) -> str:
    raw = json.dumps(asdict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def _discover_model_dirs(models_root: str) -> list[str]:
    root = Path(models_root).expanduser().resolve()
    if not root.is_dir():
        return []
    out: list[str] = []
    exts = {".pt", ".onnx", ".engine", ".trt"}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(f.is_file() and f.suffix.lower() in exts for f in d.rglob("*")):
            out.append(str(d.resolve()))
    return out


def _discover_targets(*, workspace: str, source_kind: str, runs_root: str | None, models_root: str | None) -> list[tuple[str, Kind]]:
    layout = WorkspaceLayout(workspace)
    out: list[tuple[str, Kind]] = []
    if source_kind in {"run", "all"}:
        rr = runs_root or layout.runs
        out.extend((str(Path(p).resolve()), "run") for p in find_run_directories(rr))
    if source_kind in {"model", "all"}:
        mr = models_root or layout.models
        out.extend((str(Path(p).resolve()), "model") for p in _discover_model_dirs(mr))
    uniq: dict[tuple[str, Kind], None] = {}
    for item in out:
        uniq[item] = None
    return list(uniq.keys())


def _existing_snapshot_hash(target_ref: str) -> str | None:
    snap = unified_snapshot_dir(target_ref) / "snapshot.json"
    if not snap.is_file():
        return None
    try:
        payload = json.loads(snap.read_text(encoding="utf-8"))
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(raw).hexdigest()
    except Exception:
        return None


def run_migration(
    *,
    workspace: str,
    source_kind: str,
    mode: Mode,
    runs_root: str | None,
    models_root: str | None,
    continue_on_error: bool,
) -> dict[str, Any]:
    targets = _discover_targets(
        workspace=workspace,
        source_kind=source_kind,
        runs_root=runs_root,
        models_root=models_root,
    )
    items: list[MigrationItem] = []
    for ref, sk in targets:
        try:
            payload = map_legacy_payload(read_legacy_target(ref, source_kind=sk))
            ph = _payload_hash(payload)
            old = _existing_snapshot_hash(ref)
            if old == ph:
                items.append(
                    MigrationItem(
                        ref=ref,
                        source_kind=sk,
                        status="skipped",
                        reason="up_to_date",
                        payload_hash_sha256=ph,
                    )
                )
                continue
            if mode in {"dry-run", "report-only"}:
                items.append(
                    MigrationItem(
                        ref=ref,
                        source_kind=sk,
                        status="planned",
                        reason=mode,
                        payload_hash_sha256=ph,
                        rollback_hint=(
                            "Dry-run/report-only: no files changed. Re-run with --mode apply to execute migration."
                        ),
                    )
                )
                continue
            rep = write_unified_snapshot(payload, ref)
            items.append(
                MigrationItem(
                    ref=ref,
                    source_kind=sk,
                    status="migrated",
                    payload_hash_sha256=ph,
                    snapshot_path=rep.snapshot_path,
                    rollback_hint="Delete .smartrain/unified/snapshot.json and manifest.json to revert this migrated target.",
                )
            )
        except Exception as exc:
            items.append(
                MigrationItem(
                    ref=ref,
                    source_kind=sk,
                    status="failed",
                    reason=str(exc),
                    rollback_hint=(
                        "Fix source artifacts/metadata and retry with --continue-on-error, or scope migration using --source-kind/--runs-root/--models-root."
                    ),
                )
            )
            if not continue_on_error:
                break
    stats = {
        "total": len(items),
        "migrated": sum(1 for x in items if x.status == "migrated"),
        "planned": sum(1 for x in items if x.status == "planned"),
        "skipped": sum(1 for x in items if x.status == "skipped"),
        "failed": sum(1 for x in items if x.status == "failed"),
    }
    return {
        "workspace": workspace,
        "source_kind": source_kind,
        "mode": mode,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": stats,
        "operator_guidance": {
            "dry_run": "Use --mode dry-run to preview; no writes occur.",
            "apply": "Use --mode apply to write unified snapshots.",
            "rollback": "Per-target rollback hint is included in each item (rollback_hint).",
        },
        "items": [asdict(x) for x in items],
    }


def _default_report_path(workspace: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(Path(workspace) / "analytics" / "migration-reports" / f"unified-migration-{ts}.json")


def _write_summary_md(report: dict[str, Any], json_path: str) -> str:
    md_path = str(Path(json_path).with_suffix(".md"))
    s = report.get("stats") or {}
    lines = [
        "# Unified Migration Report",
        "",
        f"- workspace: `{report.get('workspace')}`",
        f"- mode: `{report.get('mode')}`",
        f"- source_kind: `{report.get('source_kind')}`",
        f"- generated_at: `{report.get('generated_at')}`",
        "",
        f"- total: {s.get('total', 0)}",
        f"- migrated: {s.get('migrated', 0)}",
        f"- planned: {s.get('planned', 0)}",
        f"- skipped: {s.get('skipped', 0)}",
        f"- failed: {s.get('failed', 0)}",
    ]
    Path(md_path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return md_path


def _add_migration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=str, default=None, help="Workspace root.")
    parser.add_argument("--source-kind", choices=["run", "model", "all"], default="all")
    parser.add_argument("--runs-root", type=str, default=None, help="Override runs root.")
    parser.add_argument("--models-root", type=str, default=None, help="Override models root.")
    parser.add_argument("--mode", choices=["dry-run", "apply", "report-only"], default="dry-run")
    parser.add_argument("--report", type=str, default=None, help="JSON report path.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue migration after per-target errors.")


def build_migration_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="smartrain migrate", description="Unified migration utilities.")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_unified = sub.add_parser("unified", help="Migrate legacy runs/models into unified snapshots.")
    _add_migration_args(p_unified)
    p_legacy = sub.add_parser("canonical", help=argparse.SUPPRESS)
    _add_migration_args(p_legacy)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_migration_arg_parser()
    args = parser.parse_args(argv)
    ws = resolve_workspace_root(getattr(args, "workspace", None))
    report = run_migration(
        workspace=ws,
        source_kind=str(args.source_kind),
        mode=str(args.mode),  # type: ignore[arg-type]
        runs_root=getattr(args, "runs_root", None),
        models_root=getattr(args, "models_root", None),
        continue_on_error=bool(getattr(args, "continue_on_error", False)),
    )
    rp = str(getattr(args, "report", None) or _default_report_path(ws))
    p = Path(rp).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = _write_summary_md(report, str(p))
    print(f"[OK] Migration report JSON: {p}")
    print(f"[OK] Migration report summary: {md}")


if __name__ == "__main__":
    main()

