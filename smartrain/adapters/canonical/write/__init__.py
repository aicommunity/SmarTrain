"""Canonical write adapters (snapshot + manifest + dual-write)."""

from smartrain.adapters.canonical.write.dual_write import DualWriteReport, run_dual_write
from smartrain.adapters.canonical.write.layout import canonical_snapshot_dir
from smartrain.adapters.canonical.write.writer import WriteReport, write_canonical_snapshot

__all__ = [
    "DualWriteReport",
    "WriteReport",
    "canonical_snapshot_dir",
    "run_dual_write",
    "write_canonical_snapshot",
]
