from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smartrain.run_model_contract.io.write.layout import unified_snapshot_write_dir
from smartrain.run_model_contract.io.write.manifest import build_manifest
from smartrain.run_model_contract.domain.models import UnifiedPayload
from smartrain.core.runtime.path_portable import posix_relpath


@dataclass(frozen=True)
class WriteReport:
    target_root: str
    unified_root: str
    snapshot_path: str
    manifest_path: str
    payload_hash_sha256: str
    aggregate_artifacts_hash_sha256: str
    manifest_hash_sha256: str


def _payload_json_bytes(payload: UnifiedPayload) -> tuple[bytes, str]:
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


def write_unified_snapshot(payload: UnifiedPayload, target_root: str) -> WriteReport:
    """
    Persist canonical payload + manifest under the target run/model directory.

    Idempotent overwrite of snapshot/manifest for the same target_root.
    """
    root = Path(target_root).expanduser().resolve()
    snap = unified_snapshot_write_dir(root)
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
        source_run_ref=posix_relpath(str(root), str(root.parent.parent.parent)) if len(root.parts) >= 3 else root.name,
        policy_mode="unified_only",
        artifact_hashes=artifact_hashes,
        aggregate_artifacts_hash_sha256=aggregate_hash,
    )
    man_path = snap / "manifest.json"
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    man_path.write_bytes(manifest_bytes)
    manifest_hash = _sha256_bytes(manifest_bytes)
    return WriteReport(
        target_root=str(root),
        unified_root=str(snap),
        snapshot_path=str(snap_path),
        manifest_path=str(man_path),
        payload_hash_sha256=digest,
        aggregate_artifacts_hash_sha256=aggregate_hash,
        manifest_hash_sha256=manifest_hash,
    )
