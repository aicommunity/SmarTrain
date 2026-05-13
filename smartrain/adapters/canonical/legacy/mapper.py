from __future__ import annotations

from dataclasses import replace

from smartrain.domain.canonical.models import CanonicalPayload


def map_legacy_payload(payload: CanonicalPayload) -> CanonicalPayload:
    """
    Normalize legacy-loaded payload into canonical migration producer lineage.
    """
    producer = str(payload.producer or "").strip() or "canonical.legacy_reader"
    return replace(payload, producer=f"canonical.legacy_mapper:{producer}")

