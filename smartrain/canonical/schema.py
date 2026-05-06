from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from smartrain.core.training.train_profile import task_to_metadata_task_type


SCHEMA_V2 = "2.0.0"


def wrap_inference_report_v2(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    provider = model.get("provider") if isinstance(model.get("provider"), dict) else {}
    task_type = task_to_metadata_task_type(payload.get("task_type"))
    out = dict(payload)
    out["schema_version"] = SCHEMA_V2
    out["task_type"] = task_type
    out["backend_type"] = str(provider.get("id") or "ultralytics")
    out["producer"] = "smartrain.inference_cli"
    out["schema_created_at"] = datetime.now(timezone.utc).isoformat()
    out["v2"] = {
        "artifacts": {
            "inference_report": payload.get("output"),
            "environment_profile": (payload.get("artifacts") or {}).get("environment_profile"),
        },
        "metrics": {
            "namespace": task_type,
            "summary": payload.get("summary"),
        },
        "provenance": {
            "source_model": {
                "source": model.get("source"),
                "name": model.get("name"),
                "weights_relative": model.get("weights_relative"),
            }
        },
    }
    return out

