from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smartrain.core.runtime.file_lock import locked_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_root() -> Path:
    xdg = (os.environ.get("XDG_CONFIG_HOME") or "").strip()
    if xdg:
        return Path(xdg).expanduser().resolve()
    return (Path.home() / ".config").resolve()


def index_path() -> Path:
    return _config_root() / "smartrain" / "providers" / "index.json"


def read_index() -> dict[str, Any]:
    p = index_path()
    if not p.is_file():
        return {"schema_version": 1, "updated_at": _utc_now(), "providers": []}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("index.json must be object")
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported schema_version: {payload.get('schema_version')}")
        if not isinstance(payload.get("providers"), list):
            raise ValueError("providers must be list")
        return payload
    except Exception:
        backup = p.with_suffix(".json.bak")
        if backup.is_file():
            try:
                payload = json.loads(backup.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and payload.get("schema_version") == 1:
                    return payload
            except Exception:
                pass
        return {"schema_version": 1, "updated_at": _utc_now(), "providers": []}


def write_index(payload: dict[str, Any]) -> None:
    p = index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["schema_version"] = 1
    payload["updated_at"] = _utc_now()
    with locked_file(p):
        if p.is_file():
            backup = p.with_suffix(".json.bak")
            try:
                backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass
        fd, tmp_name = tempfile.mkstemp(prefix="index-", suffix=".json", dir=str(p.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_name, p)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)


def upsert_provider_record(record: dict[str, Any]) -> None:
    payload = read_index()
    providers = list(payload.get("providers", []))
    pid = str(record.get("provider_id", "")).strip()
    if not pid:
        raise ValueError("provider_id is required")
    providers = [x for x in providers if str(x.get("provider_id", "")).strip() != pid]
    providers.append(record)
    payload["providers"] = sorted(providers, key=lambda x: str(x.get("provider_id", "")))
    write_index(payload)


def mark_provider_state(provider_id: str, install_state: str, *, last_error: str | None = None) -> None:
    payload = read_index()
    providers = list(payload.get("providers", []))
    for rec in providers:
        if str(rec.get("provider_id", "")).strip() == str(provider_id).strip():
            rec["install_state"] = install_state
            rec["last_validated_at"] = _utc_now()
            rec["last_error"] = last_error
            break
    payload["providers"] = providers
    write_index(payload)


def list_provider_records() -> list[dict[str, Any]]:
    payload = read_index()
    records: list[dict[str, Any]] = []
    for rec in payload.get("providers", []):
        if not isinstance(rec, dict):
            continue
        records.append(rec)
    return records


def _is_existing_dir(path_value: Any) -> bool:
    path = str(path_value or "").strip()
    return bool(path) and Path(path).expanduser().is_dir()


def reconcile_stale_provider_paths() -> dict[str, int]:
    """
    Mark installed provider records as stale when repo/venv paths disappear.
    Returns counters for observability.
    """
    payload = read_index()
    providers = list(payload.get("providers", []))
    total = 0
    updated = 0
    for rec in providers:
        if not isinstance(rec, dict):
            continue
        total += 1
        state = str(rec.get("install_state", "")).strip().lower()
        if state != "installed":
            continue
        repo_ok = _is_existing_dir(rec.get("repo_path"))
        venv_ok = _is_existing_dir(rec.get("venv_path"))
        if repo_ok and venv_ok:
            continue
        rec["install_state"] = "stale"
        rec["last_validated_at"] = _utc_now()
        reason_parts: list[str] = []
        if not repo_ok:
            reason_parts.append("missing repo_path")
        if not venv_ok:
            reason_parts.append("missing venv_path")
        rec["last_error"] = ", ".join(reason_parts) or "missing provider paths"
        updated += 1
    if updated > 0:
        payload["providers"] = providers
        write_index(payload)
    return {"total": total, "stale_marked": updated}


@dataclass(frozen=True)
class ProviderLocation:
    provider_id: str
    repo_path: str
    venv_path: str
    install_state: str


def get_provider_location(provider_id: str) -> ProviderLocation | None:
    key = str(provider_id).strip()
    for rec in list_provider_records():
        if str(rec.get("provider_id", "")).strip() != key:
            continue
        return ProviderLocation(
            provider_id=key,
            repo_path=str(rec.get("repo_path", "")),
            venv_path=str(rec.get("venv_path", "")),
            install_state=str(rec.get("install_state", "stale")),
        )
    return None

