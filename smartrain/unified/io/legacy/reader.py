from __future__ import annotations

from smartrain.unified.domain.models import UnifiedPayload
from smartrain.orchestrators.unified_gateway import UnifiedGatewayOptions, load_target


def read_legacy_target(ref: str, *, source_kind: str | None = None) -> UnifiedPayload:
    """
    Read potentially legacy run/model source through canonical adapters.

    Validation is disabled at read step to allow migration report generation even
    for partially malformed historical artifacts.
    """
    return load_target(ref, source_kind=source_kind, options=UnifiedGatewayOptions(validate=False))

