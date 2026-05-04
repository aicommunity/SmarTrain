from __future__ import annotations

from dataclasses import dataclass

from smartrain.adapters.canonical.read.factory import ReadAdapterFactory
from smartrain.adapters.canonical.write.writer import WriteReport, write_canonical_snapshot
from smartrain.domain.canonical.models import CanonicalPayload
from smartrain.domain.canonical.validators import validate_payload


@dataclass(frozen=True)
class CanonicalGatewayOptions:
    validate: bool = True


def load_target(ref: str, *, source_kind: str | None = None, options: CanonicalGatewayOptions | None = None) -> CanonicalPayload:
    opts = options or CanonicalGatewayOptions()
    adapter = ReadAdapterFactory().resolve(source_kind, ref)
    payload = adapter.read(ref, options={})
    if opts.validate:
        validate_payload(payload)
    return payload


def persist_canonical_snapshot(ref: str, *, source_kind: str | None = None) -> WriteReport:
    """Load canonical payload for ref and write snapshot + manifest under target root."""
    payload = load_target(ref, source_kind=source_kind, options=CanonicalGatewayOptions(validate=True))
    return write_canonical_snapshot(payload, ref)

