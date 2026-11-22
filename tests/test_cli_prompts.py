from __future__ import annotations

import pytest

from smartrain.cli_entrypoints.support import cli_prompts


class _FakeConsoleFailure(Exception):
    """Mimics prompt_toolkit.output.win32.NoConsoleScreenBufferError without importing win32."""

    __module__ = "prompt_toolkit.output.win32"


def test_prompt_yes_no_default_true_empty(monkeypatch) -> None:
    monkeypatch.setattr(cli_prompts, "prompt", lambda *args, **kwargs: "")
    assert cli_prompts.prompt_yes_no("q", default=True) is True


def test_prompt_yes_no_default_false_no(monkeypatch) -> None:
    monkeypatch.setattr(cli_prompts, "prompt", lambda *args, **kwargs: "n")
    assert cli_prompts.prompt_yes_no("q", default=True) is False


def test_prompt_choice_by_index(monkeypatch) -> None:
    monkeypatch.setattr(cli_prompts, "prompt", lambda *args, **kwargs: "2")
    assert cli_prompts.prompt_choice("mode", ["a", "b", "c"], default="a") == "b"


def test_prompt_multichoice_csv_by_indices(monkeypatch) -> None:
    monkeypatch.setattr(cli_prompts, "prompt", lambda *args, **kwargs: "1,3")
    assert cli_prompts.prompt_multi_choice_csv("ds", ["one", "two", "three"], default_values=[]) == [
        "one",
        "three",
    ]


def test_prompt_choice_show_options_false_skips_list(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_prompts, "prompt", lambda *a, **kwargs: "2")
    assert cli_prompts.prompt_choice("Pick", ["a", "b", "c"], default="a", show_options=False) == "b"
    out = capsys.readouterr().out
    assert "Options for Pick" not in out


def test_prompt_prefilled_text_uses_default_in_prompt(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_prompt(prompt_label: str, *, default: str = "", **kwargs: object) -> str:
        seen["label"] = prompt_label
        seen["default"] = default
        return default

    monkeypatch.setattr(cli_prompts, "prompt", fake_prompt)
    assert cli_prompts.prompt_prefilled_text("New release name", "detect_yolo_20260115_120000") == (
        "detect_yolo_20260115_120000"
    )
    assert seen["default"] == "detect_yolo_20260115_120000"
    assert seen["label"] == "New release name: "


def test_prompt_prefilled_text_returns_edited_value(monkeypatch) -> None:
    monkeypatch.setattr(cli_prompts, "prompt", lambda *args, **kwargs: "my_new_name")
    assert cli_prompts.prompt_prefilled_text("New release name", "old_name") == "my_new_name"


def test_prompt_multichoice_csv_accepts_numeric_named_values(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_prompts,
        "prompt",
        lambda *args, **kwargs: "170325, 2026-04-04_19-37-54-merged, oringinal_old",
    )
    options = [
        "170325",
        "2026-04-04_19-37-54-merged",
        "oringinal_old",
    ]
    assert cli_prompts.prompt_multi_choice_csv("Input datasets", options, default_values=[]) == options


def test_is_prompt_toolkit_console_failure_narrow(monkeypatch) -> None:
    assert cli_prompts._is_prompt_toolkit_console_failure(_FakeConsoleFailure()) is True

    class BufferErrorLike(Exception):
        __module__ = "prompt_toolkit.buffer"

    assert cli_prompts._is_prompt_toolkit_console_failure(BufferErrorLike("boom")) is False
    assert cli_prompts._is_prompt_toolkit_console_failure(ValueError("nope")) is False


def test_prompt_raw_falls_back_on_console_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_prompts, "_PLAIN_INPUT_FALLBACK", False)

    def boom(*args, **kwargs):
        raise _FakeConsoleFailure("no console")

    monkeypatch.setattr(cli_prompts, "prompt", boom)
    monkeypatch.setattr("builtins.input", lambda _label: "from-input")
    assert cli_prompts._prompt_raw("Q: ") == "from-input"
    assert cli_prompts._PLAIN_INPUT_FALLBACK is True
    assert "Advanced prompt unavailable" in capsys.readouterr().out


def test_prompt_raw_does_not_fallback_on_non_console_prompt_toolkit_error(monkeypatch) -> None:
    monkeypatch.setattr(cli_prompts, "_PLAIN_INPUT_FALLBACK", False)

    class BufferErrorLike(Exception):
        __module__ = "prompt_toolkit.buffer"

    def boom(*args, **kwargs):
        raise BufferErrorLike("completion failed")

    monkeypatch.setattr(cli_prompts, "prompt", boom)
    with pytest.raises(BufferErrorLike):
        cli_prompts._prompt_raw("Q: ")
    assert cli_prompts._PLAIN_INPUT_FALLBACK is False
