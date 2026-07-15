from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_RELEASE_STEM_RE = re.compile(
    r"^(?P<task>[a-z][a-z0-9]*)_(?P<model>.+)_(?P<dt>\d{8}_\d{6})$",
    flags=re.IGNORECASE,
)

_TASK_DENORMALIZE = {
    "detect": "detect",
    "classify": "classify",
}


def sanitize_release_stem(name: str) -> str:
    s = re.sub(r"[^\w.\-+]+", "_", str(name), flags=re.UNICODE).strip("._")
    return s[:180] if s else "unknown"


def normalize_release_task(task: str | None) -> str:
    raw = (task or "").strip().lower()
    mapping = {
        "detection": "detect",
        "det": "detect",
        "classification": "classify",
    }
    return mapping.get(raw, raw or "detect")


def model_token_from_version(model_version: str | None) -> str:
    model_token = Path(str(model_version or "model")).name
    if model_token.endswith(".pt"):
        model_token = model_token[:-3]
    if model_token.endswith(".yaml"):
        model_token = model_token[:-5]
    model_token = re.sub(r"[^a-zA-Z0-9._+-]+", "-", model_token).strip("-") or "model"
    return model_token


def _format_batch_token(batch: int | float) -> str:
    if isinstance(batch, float) and not batch.is_integer():
        return "b" + str(batch).replace(".", "p")
    return f"b{int(batch)}"


def build_model_weights_stem(
    task_type: str | None,
    model_version: str,
    epochs: int,
    batch: int | float,
    img_size: int,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Public weight filename stem (folder names stay independent).

    Example: ``detect_yolo11s_20260706_030258_640px_400epochs_b16``
    """
    task = normalize_release_task(task_type)
    model_token = model_token_from_version(model_version)
    ts = timestamp or datetime.now()
    dt = ts.strftime("%Y%m%d_%H%M%S")
    return f"{task}_{model_token}_{dt}_{int(img_size)}px_{int(epochs)}epochs_{_format_batch_token(batch)}"


def parse_training_start_timestamp(payload: Any) -> datetime | None:
    if not isinstance(payload, dict):
        return None
    timestamps = payload.get("timestamps")
    if not isinstance(timestamps, dict):
        return None
    training = timestamps.get("training")
    if not isinstance(training, dict):
        return None
    raw = training.get("start")
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def build_model_weights_stem_from_metadata(payload: Any, *, fallback_timestamp: datetime | None = None) -> str | None:
    """Derive weights stem from training/release metadata when fields are available."""
    if not isinstance(payload, dict):
        return None
    ti = payload.get("training_info")
    if not isinstance(ti, dict):
        training = payload.get("training")
        if isinstance(training, dict):
            ti = training.get("training_info") if isinstance(training.get("training_info"), dict) else None
    if not isinstance(ti, dict):
        return None
    hp = ti.get("hyperparameters") if isinstance(ti.get("hyperparameters"), dict) else {}
    try:
        epochs = int(hp.get("epochs"))
        img_size = int(hp.get("image_size"))
        batch_raw = hp.get("batch_size")
        if isinstance(batch_raw, float) and not float(batch_raw).is_integer():
            batch: int | float = float(batch_raw)
        else:
            batch = int(batch_raw)
    except (TypeError, ValueError):
        return None
    model = ti.get("model")
    if not model:
        return None
    ts = parse_training_start_timestamp(payload) or fallback_timestamp
    if ts is None:
        training = payload.get("training")
        if isinstance(training, dict):
            ts = parse_training_start_timestamp({"timestamps": training.get("timestamps")})
            if ts is None:
                ts = parse_training_start_timestamp(training)
    return build_model_weights_stem(
        ti.get("task_type"),
        str(model),
        epochs,
        batch,
        img_size,
        timestamp=ts,
    )

def is_release_metadata(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    source = payload.get("source")
    artifacts = payload.get("artifacts")
    if not isinstance(source, dict) or not isinstance(artifacts, dict):
        return False
    return bool(source.get("source_run")) and bool(artifacts.get("release_dir"))


def is_registry_bundle_path(path: Path) -> bool:
    for parent in [path, *path.parents]:
        if (parent / "model_manifest.json").is_file():
            return True
    return False


def release_json_path_for_pt(pt_path: Path) -> Path:
    return pt_path.with_suffix(".json")


def load_release_metadata(json_path: Path) -> dict[str, Any] | None:
    if not json_path.is_file():
        return None
    try:
        import json

        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not is_release_metadata(payload):
        return None
    return payload


@dataclass(frozen=True)
class ParsedReleaseStem:
    task: str
    model: str
    datetime: str


def parse_canonical_release_stem(stem: str) -> ParsedReleaseStem | None:
    """Parse legacy ``task_model_YYYYMMDD_HHMMSS`` stems only.

    Modern weight stems (``detect_…_640px_…epochs_b…``) and run-folder names
    do not match this pattern and return ``None``.
    """
    m = _RELEASE_STEM_RE.match(str(stem).strip())
    if not m:
        return None
    return ParsedReleaseStem(
        task=m.group("task").lower(),
        model=m.group("model"),
        datetime=m.group("dt"),
    )


def task_type_from_release_stem_task(task: str) -> str:
    return _TASK_DENORMALIZE.get(task.lower(), task)
