from __future__ import annotations

import os


def find_yaml_file(folder_path: str) -> str | None:
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower() in ("data.yaml", "data.yml"):
                return os.path.join(root, file_name)
    return None


def find_obj_names_file(folder_path: str) -> str | None:
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower() == "obj.names":
                return os.path.join(root, file_name)
    return None


def find_obj_data_file(folder_path: str) -> str | None:
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower() == "obj.data":
                return os.path.join(root, file_name)
    return None


def load_obj_names(file_path: str) -> list[str] | None:
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle.readlines() if line.strip()]
    except Exception as exc:
        print(f"[ERROR] Failed to read {file_path}: {exc}")
        return None


def load_obj_data(file_path: str) -> int | None:
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        for line in content.split("\n"):
            item = line.strip()
            if item.startswith("classes"):
                parts = item.split("=")
                if len(parts) == 2:
                    return int(parts[1].strip())
        return None
    except Exception as exc:
        print(f"[ERROR] Failed to read {file_path}: {exc}")
        return None

