from __future__ import annotations

from pathlib import Path


def canonical_snapshot_dir(target_root: str | Path) -> Path:
    """Deterministic on-disk layout for canonical snapshot under a run or model root."""
    return Path(target_root).expanduser().resolve() / ".smartrain" / "canonical"
