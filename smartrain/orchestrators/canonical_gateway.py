from __future__ import annotations

from dataclasses import dataclass

from smartrain.adapters.canonical.read.factory import ReadAdapterFactory
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

