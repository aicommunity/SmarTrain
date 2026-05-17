"""Unified write adapters (snapshot + manifest + dual-write)."""

from smartrain.run_model_contract.io.write.dual_write import DualWriteReport, run_dual_write
from smartrain.run_model_contract.io.write.layout import unified_snapshot_dir
from smartrain.run_model_contract.io.write.snapshot_hook import maybe_dual_write_unified_snapshot
from smartrain.run_model_contract.io.write.writer import WriteReport, write_unified_snapshot

__all__ = [
    "DualWriteReport",
    "WriteReport",
    "unified_snapshot_dir",
    "maybe_dual_write_unified_snapshot",
    "run_dual_write",
    "write_unified_snapshot",
]
