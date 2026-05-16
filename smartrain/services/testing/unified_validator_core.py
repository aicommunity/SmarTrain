from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalProvenance:
    inference_source: str
    gt_source: str
    nms_profile: str

    def as_dict(self) -> dict[str, str]:
        return {
            "inference_source": self.inference_source,
            "gt_source": self.gt_source,
            "nms_profile": self.nms_profile,
        }


def normalize_eval_params(
    *,
    imgsz: int | None,
    conf: float | None,
    iou: float | None,
    default_imgsz: int = 640,
    default_conf: float = 0.001,
    default_iou: float = 0.7,
) -> dict[str, Any]:
    return {
        "imgsz": int(imgsz if imgsz is not None else default_imgsz),
        "conf": float(conf if conf is not None else default_conf),
        "iou": float(iou if iou is not None else default_iou),
    }

