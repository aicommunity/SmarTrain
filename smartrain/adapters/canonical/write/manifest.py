from __future__ import annotations

from typing import Any

from smartrain.domain.canonical.models import CanonicalPayload


def build_manifest(*, payload: CanonicalPayload, payload_hash: str) -> dict[str, Any]:
    """Machine-readable manifest alongside snapshot.json (PR 6.4 contract)."""
    return {
        "schema_version": payload.schema_version,
        "producer": payload.producer,
        "generated_at": payload.generated_at,
        "payload_hash_sha256": payload_hash,
        "snapshot_file": "snapshot.json",
    }
