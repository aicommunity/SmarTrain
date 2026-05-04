from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from smartrain.adapters.canonical.write.layout import canonical_snapshot_dir
from smartrain.adapters.canonical.write.manifest import build_manifest
from smartrain.domain.canonical.models import CanonicalPayload


@dataclass(frozen=True)
class WriteReport:
    target_root: str
    canonical_root: str
    snapshot_path: str
    manifest_path: str
    payload_hash_sha256: str


def _payload_json_bytes(payload: CanonicalPayload) -> tuple[bytes, str]:
    body_dict: dict[str, Any] = asdict(payload)
    raw = json.dumps(body_dict, ensure_ascii=False, indent=2, sort_keys=True)
    b = raw.encode("utf-8")
    digest = hashlib.sha256(b).hexdigest()
    return b, digest


def write_canonical_snapshot(payload: CanonicalPayload, target_root: str) -> WriteReport:
    """
    Persist canonical payload + manifest under the target run/model directory.

    Idempotent overwrite of snapshot/manifest for the same target_root.
    """
    root = Path(target_root).expanduser().resolve()
    snap = canonical_snapshot_dir(root)
    snap.mkdir(parents=True, exist_ok=True)
    body, digest = _payload_json_bytes(payload)
    snap_path = snap / "snapshot.json"
    snap_path.write_bytes(body)
    manifest = build_manifest(payload=payload, payload_hash=digest)
    man_path = snap / "manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return WriteReport(
        target_root=str(root),
        canonical_root=str(snap),
        snapshot_path=str(snap_path),
        manifest_path=str(man_path),
        payload_hash_sha256=digest,
    )
