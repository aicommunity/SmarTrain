from __future__ import annotations

import os
import sys
from typing import Any, Callable


def resolve_all_data_yaml_context(
    *,
    args: Any,
    baseline: str,
    others: list[str],
    profile: str,
    interactive_mode: bool,
    build_run_data_yaml_map_cb: Callable[..., tuple[dict[str, str], dict[str, str], list[str]]],
    auto_select_data_yaml_cb: Callable[..., str | None],
    prompt_choice_cb: Callable[..., str],
    prompt_text_cb: Callable[..., str],
) -> tuple[list[str], str, list[str], dict[str, str], list[str]]:
    report_languages_raw = str(getattr(args, "report_languages", "ru,en") or "ru,en")
    report_languages = [x.strip() for x in report_languages_raw.split(",") if x.strip()]
    if not report_languages:
        report_languages = ["ru", "en"]

    data_yaml = str(getattr(args, "data_yaml", "") or "").strip()
    selected_run_dirs = [baseline] + others
    run_data_yaml_map, run_data_yaml_source, unresolved_data_yaml_runs = build_run_data_yaml_map_cb(
        selected_run_dirs,
        args.workspace,
        preferred_split="test" if profile in ("speed", "full") else None,
    )

    unique_data_yaml = sorted(set(run_data_yaml_map.values()))
    if profile in ("speed", "full"):
        if data_yaml:
            for rd in selected_run_dirs:
                run_data_yaml_map.setdefault(rd, data_yaml)
            unique_data_yaml = sorted(set(run_data_yaml_map.values()))
        elif interactive_mode and len(unique_data_yaml) > 1:
            print("[INFO] Multiple datasets detected across selected runs.")
            for rd in selected_run_dirs:
                dy = run_data_yaml_map.get(rd)
                src = run_data_yaml_source.get(rd, "unknown")
                print(f"[INFO]  - {os.path.basename(rd.rstrip(os.sep))}: {dy or 'UNRESOLVED'} (source: {src})")
            mode = prompt_choice_cb(
                "Data.yaml mode",
                ["auto_per_run", "single_shared"],
                default="auto_per_run",
                show_options=False,
            )
            if mode == "single_shared":
                auto_yaml = auto_select_data_yaml_cb(
                    baseline,
                    others,
                    args.workspace,
                    preferred_split="test",
                )
                if auto_yaml:
                    data_yaml = auto_yaml
                    for rd in selected_run_dirs:
                        run_data_yaml_map[rd] = data_yaml
                    unique_data_yaml = [data_yaml]
        elif not data_yaml and len(unique_data_yaml) == 1:
            data_yaml = unique_data_yaml[0]
        elif not data_yaml and interactive_mode and not run_data_yaml_map:
            data_yaml = prompt_text_cb("Path to data.yaml (required for speed/full)", default="").strip()
            if data_yaml:
                for rd in selected_run_dirs:
                    run_data_yaml_map[rd] = data_yaml
                unique_data_yaml = [data_yaml]

        if not data_yaml and not run_data_yaml_map and not interactive_mode:
            print(
                "[ERROR] No data.yaml resolved for selected runs; use --data-yaml or ensure metadata/runtime yaml is present.",
                file=sys.stderr,
            )
            sys.exit(2)

    return report_languages, data_yaml, selected_run_dirs, run_data_yaml_map, unresolved_data_yaml_runs

