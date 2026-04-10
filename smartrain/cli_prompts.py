from __future__ import annotations

import sys
from typing import Sequence

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

YES_TOKENS = {"y", "yes", "1", "true", "yes", "d"}
NO_TOKENS = {"n", "no", "0", "false", "No", "n"}


def is_interactive_tty() -> bool:
    return bool(sys.stdin.isatty())


def _prompt_label(label: str, default: str | None = None) -> str:
    if default is None or default == "":
        return f"{label}: "
    return f"{label} [default: {default}]: "


def _echo_default_if_used(default_value: str | None, prompt_label: str) -> None:
    if default_value is None:
        return
    if default_value == "":
        return
    # We are trying to show default "in the input line" as if the user typed it.
    if is_interactive_tty():
        try:
            sys.stdout.write("\x1b[1A\r")
            sys.stdout.write(f"{prompt_label}{default_value}\n")
            sys.stdout.flush()
            return
        except Exception:
            pass
    print(default_value)


def prompt_text(label: str, default: str | None = None, choices: Sequence[str] | None = None) -> str:
    prompt_label = _prompt_label(label, default)
    completer = WordCompleter(list(choices), ignore_case=True) if choices else None
    raw = prompt(
        prompt_label,
        default="",
        completer=completer,
        complete_while_typing=True,
    ).strip()
    if raw:
        return raw
    fallback = "" if default is None else str(default)
    _echo_default_if_used(fallback, prompt_label)
    return fallback


def prompt_yes_no(label: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    prompt_label = f"{label} [{suffix}]: "
    raw = prompt(prompt_label, default="").strip().lower()
    if not raw:
        _echo_default_if_used("y" if default else "n", prompt_label)
        return default
    if raw in YES_TOKENS:
        return True
    if raw in NO_TOKENS:
        return False
    return default


def prompt_int(label: str, default: int) -> int:
    while True:
        raw = prompt_text(label, default=str(default)).strip()
        try:
            return int(raw)
        except ValueError:
            print(f"[ERROR] Expected integer, received: {raw!r}")


def prompt_optional_int(label: str, default: int | None = None) -> int | None:
    while True:
        prompt_label = _prompt_label(label, "" if default is None else str(default))
        raw = prompt(prompt_label, default="").strip()
        if not raw:
            _echo_default_if_used("" if default is None else str(default), prompt_label)
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"[ERROR] Expecting an integer or empty value, received: {raw!r}")


def prompt_optional_float(label: str, default: float | None = None) -> float | None:
    while True:
        prompt_label = _prompt_label(label, "" if default is None else str(default))
        raw = prompt(prompt_label, default="").strip()
        if not raw:
            _echo_default_if_used("" if default is None else str(default), prompt_label)
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"[ERROR] Expecting a number or empty value, received: {raw!r}")


def prompt_choice(label: str, options: Sequence[str], default: str | None = None) -> str:
    if not options:
        raise ValueError("options is empty")
    print(f"[INFO] Options for {label}:")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    choice_default = default if default in options else options[0]
    while True:
        raw = prompt(
            _prompt_label(f"{label} (number or value)", choice_default),
            default="",
            completer=WordCompleter(list(options), ignore_case=True),
            complete_while_typing=True,
        ).strip()
        if not raw:
            _echo_default_if_used(choice_default, _prompt_label(f"{label} (number or value)", choice_default))
            return choice_default
        if raw in options:
            return raw
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print(f"[ERROR] Incorrect selection: {raw!r}")


def prompt_multi_choice_csv(
    label: str,
    options: Sequence[str],
    default_values: Sequence[str] | None = None,
) -> list[str]:
    if not options:
        return []
    print(f"[INFO] Options for {label}:")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    default_csv = ",".join(default_values or [])
    while True:
        raw = prompt(
            _prompt_label(f"{label} (CSV of numbers or values)", default_csv),
            default="",
            completer=WordCompleter(list(options), ignore_case=True),
            complete_while_typing=True,
        ).strip()
        if not raw:
            defaults = list(default_values or [])
            _echo_default_if_used(
                ",".join(defaults),
                _prompt_label(f"{label} (CSV of numbers or values)", default_csv),
            )
            return defaults
        out: list[str] = []
        ok = True
        for part in (x.strip() for x in raw.split(",")):
            if not part:
                continue
            if part in options:
                val = part
            elif part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(options):
                    val = options[idx - 1]
                else:
                    ok = False
                    break
            else:
                val = part
                if val not in options:
                    ok = False
                    break
            if val not in out:
                out.append(val)
        if ok:
            return out
        print(f"[ERROR] Invalid list: {raw!r}")
