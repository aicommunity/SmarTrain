from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    aggregate_artifacts_hash_sha256: str
    manifest_hash_sha256: str


def _payload_json_bytes(payload: CanonicalPayload) -> tuple[bytes, str]:
    body_dict: dict[str, Any] = asdict(payload)
    raw = json.dumps(body_dict, ensure_ascii=False, indent=2, sort_keys=True)
    b = raw.encode("utf-8")
    digest = hashlib.sha256(b).hexdigest()
    return b, digest


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _aggregate_hash(parts: dict[str, str]) -> str:
    packed = "|".join(f"{key}:{parts[key]}" for key in sorted(parts))
    return _sha256_bytes(packed.encode("utf-8"))


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
    artifact_hashes = {
        "snapshot.json": {
            "path": "snapshot.json",
            "sha256": digest,
            "size_bytes": len(body),
        }
    }
    aggregate_hash = _aggregate_hash({"snapshot.json": digest})
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = build_manifest(
        payload=payload,
        payload_hash=digest,
        created_at=created_at,
        source_run_ref=str(root),
        policy_mode="canonical_only",
        artifact_hashes=artifact_hashes,
        aggregate_artifacts_hash_sha256=aggregate_hash,
    )
    man_path = snap / "manifest.json"
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    man_path.write_bytes(manifest_bytes)
    manifest_hash = _sha256_bytes(manifest_bytes)
    return WriteReport(
        target_root=str(root),
        canonical_root=str(snap),
        snapshot_path=str(snap_path),
        manifest_path=str(man_path),
        payload_hash_sha256=digest,
        aggregate_artifacts_hash_sha256=aggregate_hash,
        manifest_hash_sha256=manifest_hash,
    )
