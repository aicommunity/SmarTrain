"""Train interactive prompts — shared yes/no/int from cli_prompts; text keeps completer support."""

from __future__ import annotations

from smartrain.cli_entrypoints.support.cli_prompts import _echo_default_if_used
from smartrain.cli_entrypoints.support.cli_prompts import _prompt_raw
from smartrain.cli_entrypoints.support.cli_prompts import prompt_int as _shared_prompt_int
from smartrain.cli_entrypoints.support.cli_prompts import prompt_yes_no as _shared_prompt_yes_no
from smartrain.core.runtime.device_selector import default_device_value, prompt_device_selection


def prompt_input(label: str, default: str = "", completer=None, show_default_hint: bool = True) -> str:
    prompt_label = f"{label} [default: {default}]: " if (default != "" and show_default_hint) else label
    value = _prompt_raw(prompt_label, default=str(default), completer=completer).strip()
    if value:
        return value
    if default != "":
        _echo_default_if_used(str(default), prompt_label)
    return str(default)


def prompt_yes_no(label: str, default: bool = False) -> bool:
    return _shared_prompt_yes_no(label, default=default)


def prompt_int(label: str, default: int) -> int:
    return _shared_prompt_int(label, default=default)


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
