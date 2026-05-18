from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_file(install_root: str, provider_id: str) -> Path:
    return Path(install_root).expanduser().resolve() / provider_id / ".smartrain_install.json"


def write_provider_state(install_root: str, provider_id: str, payload: dict[str, Any]) -> None:
    p = state_file(install_root, provider_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["updated_at"] = _utc_now()
    p.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")


def read_provider_state(install_root: str, provider_id: str) -> dict[str, Any] | None:
    p = state_file(install_root, provider_id)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None

