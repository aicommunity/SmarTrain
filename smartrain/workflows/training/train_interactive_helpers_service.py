from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from smartrain.core.training.train_model_catalog import TrainModelCatalog


def load_available_datasets(layout) -> list[str]:
    info_path = layout.work_datasets_info_path()
    if not os.path.isfile(info_path):
        return []
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    return sorted(str(k) for k in data.keys())


def prompt_dataset_name(available: list[str], *, prompt_choice_cb) -> str:
    return prompt_choice_cb("Dataset", available, default=available[0])


def installed_external_provider_records(*, reconcile_paths_cb, list_records_cb) -> list[dict[str, Any]]:
    reconcile_paths_cb()
    recs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in list_records_cb():
        if str(rec.get("install_state", "")).strip().lower() != "installed":
            continue
        pid = str(rec.get("provider_id", "")).strip().lower()
        if not pid or pid in seen:
            continue
        repo_path = Path(str(rec.get("repo_path", "")).strip()).expanduser()
        venv_path = Path(str(rec.get("venv_path", "")).strip()).expanduser()
        if not repo_path.is_dir() or not venv_path.is_dir():
            continue
        seen.add(pid)
        recs.append(rec)
    recs.sort(key=lambda x: str(x.get("provider_id", "")).strip().lower())
    return recs


def installed_external_provider_ids(*, installed_records_cb) -> list[str]:
    return [str(r.get("provider_id", "")).strip().lower() for r in installed_records_cb()]


def get_installed_external_provider_record(provider_id: str, *, installed_records_cb) -> dict[str, Any] | None:
    key = str(provider_id or "").strip().lower()
    if not key:
        return None
    for rec in installed_records_cb():
        pid = str(rec.get("provider_id", "")).strip().lower()
        if pid == key:
            return rec
    return None


def train_model_picker_options(
    default_model: str,
    *,
    installed_records_cb,
    manual_model_entry: str,
) -> list[str]:
    catalog = TrainModelCatalog()
    options = list(catalog.supported_aliases())
    external_records = installed_records_cb()
    if external_records:
        for rec in external_records:
            pid = str(rec.get("provider_id", "")).strip().lower()
            if not pid:
                continue
            repo_path = str(rec.get("repo_path", "")).strip() or None
            ext_catalog = TrainModelCatalog(provider=pid, provider_repo_path=repo_path)
            options.extend(f"{pid}:{alias}" for alias in ext_catalog.supported_aliases())
    if default_model and default_model not in options:
        options.append(default_model)
    options.append(manual_model_entry)
    dedup: list[str] = []
    seen: set[str] = set()
    for item in options:
        if item in seen:
            continue
        seen.add(item)
        dedup.append(item)
    return dedup


def apply_external_provider_defaults(args, *, get_installed_record_cb) -> None:
    provider = str(getattr(args, "external_provider", "") or "").strip().lower()
    if not provider:
        return
    if getattr(args, "model", None) is None:
        rec = get_installed_record_cb(provider)
        repo_path = str(rec.get("repo_path", "")).strip() if isinstance(rec, dict) else None
        aliases = TrainModelCatalog(provider=provider, provider_repo_path=repo_path or None).supported_aliases()
        if aliases:
            args.model = aliases[0]
    if getattr(args, "epochs", None) is None:
        args.epochs = 70
    if getattr(args, "batch", None) is None:
        args.batch = 8
    if getattr(args, "img_size", None) is None:
        args.img_size = 640


def model_matches_task(alias: str, task: str) -> bool:
    low = alias.lower()
    if ":" in low:
        _, low = low.split(":", 1)
    task_low = (task or "").strip().lower()
    if task_low == "segment":
        return "-seg" in low
    if task_low == "classify":
        return "-cls" in low
    if task_low == "pose":
        return "-pose" in low
    if task_low == "obb":
        return "-obb" in low
    return all(marker not in low for marker in ("-seg", "-cls", "-pose", "-obb"))


def format_numbered_columns(items: list[str], *, columns: int = 4) -> list[str]:
    if not items:
        return []
    indexed = [f"{idx + 1}) {name}" for idx, name in enumerate(items)]
    term_w = shutil.get_terminal_size(fallback=(120, 20)).columns
    col_width = max(len(x) for x in indexed) + 2
    cols = max(1, min(columns, max(1, term_w // col_width)))
    rows = (len(indexed) + cols - 1) // cols
    lines: list[str] = []
    for row in range(rows):
        parts: list[str] = []
        for col in range(cols):
            pos = col * rows + row
            if pos >= len(indexed):
                continue
            cell = indexed[pos]
            if col < cols - 1:
                parts.append(cell.ljust(col_width))
            else:
                parts.append(cell)
        lines.append("".join(parts).rstrip())
    return lines


def pick_model_interactive(options: list[str], default_alias: str, *, format_columns_cb, prompt_input_cb) -> str:
    print("[INFO] Model options:")
    for line in format_columns_cb(options, columns=4):
        print(f"  {line}")
    while True:
        raw = prompt_input_cb(
            "Model (--model, number/value): ",
            default=default_alias,
        ).strip()
        if not raw:
            return default_alias
        if raw in options:
            return raw
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print(f"[ERROR] Incorrect selection: {raw!r}")

