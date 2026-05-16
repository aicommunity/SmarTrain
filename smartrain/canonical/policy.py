from __future__ import annotations

from functools import lru_cache


DEFAULT_REMOVAL_TARGET = "0.0.3"


@lru_cache(maxsize=1)
def emit_legacy_read_deprecation_warnings(removal_target: str = DEFAULT_REMOVAL_TARGET) -> None:
    """Legacy canonical-read env toggles were removed; kept as a stable no-op hook."""
    _ = removal_target
