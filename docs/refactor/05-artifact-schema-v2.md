# Artifact Schema v2

## Required Top-Level Fields

- `schema_version`
- `task_type`
- `backend_type`
- `producer`
- `created_at`
- `artifacts`
- `metrics`
- `provenance`

## Artifact Sections

- `artifacts.model`
- `artifacts.test`
- `artifacts.inference`
- `artifacts.analyze` (when generated)

## Metrics Rules

- Metrics are namespaced by task type.
- Detection metrics keep compatibility aliases for transition period.
- Every metric payload stores source backend and calculation origin.

## Compatibility

- Reader supports legacy payloads via adapters.
- Writer defaults to v2.

## Prediction bundle (normative relative paths)

Paths are relative to a **run root** or **model export root** unless noted:

| Relative path | Role |
|---------------|------|
| `predictions.jsonl` | Line-delimited prediction records (optional). |
| `predictions.json` | JSON array or object payload (optional). |
| `deep_diagnostics/debug_test.jsonl` | Test-split deep-diagnostics dump (optional). |
| `deep_diagnostics/debug_val.jsonl` | Val-split deep-diagnostics dump (optional). |

[`canonical_gateway.load_predictions`](../../smartrain/orchestrators/canonical_gateway.py) in **`predictions_strict=True`** mode discovers **only** these templates (recursive under the root). Heuristic `*pred*` globs are used only when strict mode is off.

