"""Analyze CLI prompts (patchable via workflows.analyze.results_analyzer re-exports)."""

from __future__ import annotations

from smartrain.cli_support.cli_prompts import prompt_choice, prompt_int, prompt_text

__all__ = ["prompt_choice", "prompt_int", "prompt_text"]
