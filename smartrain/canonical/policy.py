from __future__ import annotations

import os
import sys
from functools import lru_cache


DEFAULT_REMOVAL_TARGET = "0.0.3"


@lru_cache(maxsize=1)
def emit_legacy_read_deprecation_warnings(removal_target: str = DEFAULT_REMOVAL_TARGET) -> None:
    """
    Emit deterministic deprecation warnings for legacy canonical-read fallbacks.

    Policy source: `docs/refactor/06-deprecation-and-alias-policy.md`.
    """
    legacy_fallback_allowed = str(os.getenv("SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK", "")).strip() == "1"
    canonical_read_off_requested = str(os.getenv("SMARTTRAIN_CANONICAL_READ", "")).strip() == "0"

    if not (legacy_fallback_allowed and canonical_read_off_requested):
        return

    print(
        "[DEPRECATION] SMARTTRAIN_CANONICAL_READ is deprecated; canonical read is always enabled. "
        f"Removal target: {removal_target}.",
        file=sys.stderr,
    )
    print(
        "[DEPRECATION] SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK is deprecated; legacy read fallback is removed "
        f"and canonical-only mode is enforced. Removal target: {removal_target}.",
        file=sys.stderr,
    )

