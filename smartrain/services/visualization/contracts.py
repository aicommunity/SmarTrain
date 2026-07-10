from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


VisMode = Literal["dataset", "model", "run"]
FrameStatus = Literal["ok", "skipped", "error"]


@dataclass(frozen=True)
class VisRequest:
    mode: VisMode
    workspace_root: Path
    dataset: str | None
    model_name: str | None
    run_ref: str | None
    weights: str | None
    splits: tuple[str, ...] | None
    limit: int | None
    conf: float | None
    device: str | None
    overwrite: bool
    non_interactive: bool


@dataclass(frozen=True)
class FrameRecord:
    split: str
    source_abs: Path
    source_rel: str
    label_abs: Path | None
    output_abs: Path


@dataclass(frozen=True)
class VisFrameStatus:
    source_rel: str
    split: str
    status: FrameStatus
    reason: str | None
    gt_count: int
    pred_count: int
    output_rel: str | None


@dataclass(frozen=True)
class VisSummary:
    mode: str
    target: dict[str, str | None]
    total_frames: int
    ok_frames: int
    skipped_frames: int
    error_frames: int
    started_at: str
    finished_at: str
    config: dict[str, object]

