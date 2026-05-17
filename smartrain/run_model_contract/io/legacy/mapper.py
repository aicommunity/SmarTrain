from __future__ import annotations

from dataclasses import replace

from smartrain.unified.domain.models import UnifiedPayload


def map_legacy_payload(payload: UnifiedPayload) -> UnifiedPayload:
    """
    Normalize legacy-loaded payload into canonical migration producer lineage.
    """
    producer = str(payload.producer or "").strip() or "canonical.legacy_reader"
    return replace(payload, producer=f"canonical.legacy_mapper:{producer}")

