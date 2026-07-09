from __future__ import annotations

import sys

from smartrain.core.runtime.device_selector import default_device_value, prompt_device_selection


def prompt_input(label: str, default: str = "", completer=None, show_default_hint: bool = True) -> str:
    from prompt_toolkit import prompt

    prompt_label = f"{label} [default: {default}]: " if (default != "" and show_default_hint) else label
    value = str(prompt(prompt_label, default=str(default), completer=completer, complete_while_typing=True)).strip()
    if value:
        return value
    if default != "":
        if sys.stdin.isatty():
            try:
                sys.stdout.write("\x1b[1A\r")
                sys.stdout.write(f"{prompt_label}{default}\n")
                sys.stdout.flush()
            except Exception:
                print(default)
        else:
            print(default)
    return str(default)


def prompt_yes_no(label: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    default_text = "y" if default else "n"
    raw = prompt_input(f"{label} [{suffix}]: ", default=default_text, show_default_hint=False).strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true", "yes", "d")


def prompt_int(label: str, default: int) -> int:
    while True:
        raw = prompt_input(f"{label}: ", default=str(default)).strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"[ERROR] Expected integer, received: {raw!r}")


def prompt_optional_int(label: str, default: int | None = None) -> int | None:
    default_text = "" if default is None else str(default)
    while True:
        raw = prompt_input(f"{label}: ", default=default_text).strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"[ERROR] Expecting an integer or empty value, received: {raw!r}")


def prompt_optional_float(label: str, default: float | None = None) -> float | None:
    default_text = "" if default is None else str(default)
    while True:
        raw = prompt_input(f"{label}: ", default=default_text).strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"[ERROR] Expecting a number or empty value, received: {raw!r}")


def prompt_train_device(default: str | None = None) -> str:
    return prompt_device_selection(title="Train devices", default_device=default or default_device_value())

