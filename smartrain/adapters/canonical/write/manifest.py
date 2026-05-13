from __future__ import annotations

from typing import Any

from smartrain.domain.canonical.models import CanonicalPayload


def build_manifest(
    *,
    payload: CanonicalPayload,
    payload_hash: str,
    created_at: str,
    source_run_ref: str,
    policy_mode: str,
    artifact_hashes: dict[str, dict[str, Any]],
    aggregate_artifacts_hash_sha256: str,
) -> dict[str, Any]:
    """Machine-readable manifest alongside snapshot.json (PR 6.4 contract)."""
    return {
        "schema_version": payload.schema_version,
        "producer": payload.producer,
        "generated_at": payload.generated_at,
        "created_at": created_at,
        "source_run_ref": source_run_ref,
        "policy_mode": policy_mode,
        "hash_algorithm": "sha256",
        "payload_hash_sha256": payload_hash,
        "aggregate_artifacts_hash_sha256": aggregate_artifacts_hash_sha256,
        "artifacts": artifact_hashes,
        "snapshot_file": "snapshot.json",
    }
