from __future__ import annotations

import os
import warnings

from smartrain.run_model_contract.io.write.dual_write import DualWriteMode, normalize_dual_write_mode

__all__ = ["unified_write_enabled", "unified_dual_write_mode"]


def unified_write_enabled() -> bool:
    if str(os.getenv("SMARTTRAIN_UNIFIED_WRITE", "")).strip() == "1":
        return True
    if str(os.getenv("SMARTTRAIN_CANONICAL_WRITE", "")).strip() == "1":
        warnings.warn(
            "SMARTTRAIN_CANONICAL_WRITE is deprecated; use SMARTTRAIN_UNIFIED_WRITE",
            DeprecationWarning,
            stacklevel=2,
        )
        return True
    return False


def unified_dual_write_mode() -> DualWriteMode:
    raw = os.getenv("SMARTTRAIN_UNIFIED_DUAL_WRITE_MODE") or os.getenv("SMARTTRAIN_CANONICAL_DUAL_WRITE_MODE") or "unified_only"
    return normalize_dual_write_mode(raw)
