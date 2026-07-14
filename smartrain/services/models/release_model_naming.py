from __future__ import annotations

import re
from dataclasses import dataclass
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

    Modern releases use the training run folder name (``build_run_name`` style);
    those stems do not match this pattern and return ``None``.
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
