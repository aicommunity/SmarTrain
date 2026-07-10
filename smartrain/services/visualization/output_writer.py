from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from smartrain.services.visualization.contracts import VisFrameStatus, VisSummary


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_visualized_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def append_index_row(index_path: Path, row: VisFrameStatus) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def write_summary(summary_path: Path, summary: VisSummary) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")


def write_config(config_path: Path, payload: dict[str, object]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

