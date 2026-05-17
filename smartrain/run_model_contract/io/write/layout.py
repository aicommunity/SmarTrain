from __future__ import annotations

from pathlib import Path

_UNIFIED_SEG = "unified"
_LEGACY_SEG = "canonical"


def unified_snapshot_write_dir(target_root: str | Path) -> Path:
    """Directory for new unified snapshots (write target)."""
    return Path(target_root).expanduser().resolve() / ".smartrain" / _UNIFIED_SEG


def unified_snapshot_dir(target_root: str | Path) -> Path:
    """Resolve existing snapshot directory (unified preferred, legacy fallback)."""
    root = Path(target_root).expanduser().resolve()
    base = root / ".smartrain"
    unified = base / _UNIFIED_SEG
    legacy = base / _LEGACY_SEG
    if (unified / "snapshot.json").is_file():
        return unified
    if (legacy / "snapshot.json").is_file():
        return legacy
    return unified_snapshot_write_dir(root)
