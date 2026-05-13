from __future__ import annotations

import os

INTERACTIVE_ALLOWED_ENV = "SMART_TRAIN_INTERACTIVE_ALLOWED"


def is_interactive_allowed(argv: list[str] | tuple[str, ...] | None = None) -> bool:
    """
    Unified contract:
    - if env flag is set by CLI router, trust it;
    - otherwise (direct module run/tests), allow interactive only when argv is empty.
    """
    env = os.environ.get(INTERACTIVE_ALLOWED_ENV)
    if env is not None:
        return str(env).strip() in {"1", "true", "yes", "on"}
    return not bool(argv)

