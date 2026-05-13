from __future__ import annotations

from smartrain.domain.canonical.models import CanonicalPayload
from smartrain.orchestrators.canonical_gateway import CanonicalGatewayOptions, load_target


def read_legacy_target(ref: str, *, source_kind: str | None = None) -> CanonicalPayload:
    """
    Read potentially legacy run/model source through canonical adapters.

    Validation is disabled at read step to allow migration report generation even
    for partially malformed historical artifacts.
    """
    return load_target(ref, source_kind=source_kind, options=CanonicalGatewayOptions(validate=False))

