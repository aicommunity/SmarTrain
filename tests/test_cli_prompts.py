from __future__ import annotations

from smartrain.cli_entrypoints.support import cli_prompts


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
