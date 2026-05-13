"""Tests for analyze all data.yaml resolution (no spurious prompts on full CLI)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from smartrain.workflows.analyze.analyze_all_data_yaml_service import resolve_all_data_yaml_context


def test_multiple_data_yaml_no_prompt_when_selection_not_interactive() -> None:
    """Full CLI (baseline/profile/others) must not open data.yaml mode prompt in a TTY."""
    prompt_choice = MagicMock(side_effect=AssertionError("prompt_choice must not be called"))
    prompt_text = MagicMock(side_effect=AssertionError("prompt_text must not be called"))

    def build_map(
        selected_run_dirs: list[str],
        workspace: object,
        preferred_split: str | None,
    ) -> tuple[dict[str, str], dict[str, str], list[str]]:
        m = {
            selected_run_dirs[0]: "/w/datasets/a/data.yaml",
            selected_run_dirs[1]: "/w/datasets/b/data.yaml",
        }
        src = {rd: "test" for rd in selected_run_dirs}
        return m, src, []

    args = SimpleNamespace(workspace=None, data_yaml=None, report_languages="ru,en")
    baseline = "/w/runs/run_a"
    others = ["/w/runs/run_b"]

    langs, data_yaml, selected, run_map, unresolved = resolve_all_data_yaml_context(
        args=args,
        baseline=baseline,
        others=others,
        profile="full",
        selection_prompts_used=False,
        build_run_data_yaml_map_cb=build_map,
        auto_select_data_yaml_cb=MagicMock(return_value=None),
        prompt_choice_cb=prompt_choice,
        prompt_text_cb=prompt_text,
    )

    prompt_choice.assert_not_called()
    prompt_text.assert_not_called()
    assert langs == ["ru", "en"]
    assert data_yaml == ""
    assert selected == [baseline, *others]
    assert len(set(run_map.values())) == 2


def test_multiple_data_yaml_prompt_when_selection_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    prompt_choice = MagicMock(return_value="auto_per_run")

    def build_map(
        selected_run_dirs: list[str],
        workspace: object,
        preferred_split: str | None,
    ) -> tuple[dict[str, str], dict[str, str], list[str]]:
        return (
            {rd: f"/yaml_{i}.yaml" for i, rd in enumerate(selected_run_dirs)},
            {rd: "meta" for rd in selected_run_dirs},
            [],
        )

    args = SimpleNamespace(workspace=None, data_yaml=None, report_languages="en")
    resolve_all_data_yaml_context(
        args=args,
        baseline="/r/a",
        others=["/r/b"],
        profile="full",
        selection_prompts_used=True,
        build_run_data_yaml_map_cb=build_map,
        auto_select_data_yaml_cb=MagicMock(return_value=None),
        prompt_choice_cb=prompt_choice,
        prompt_text_cb=MagicMock(return_value=""),
    )
    prompt_choice.assert_called_once()
