# Environment variables

Smart Train reads configuration from CLI flags and environment variables. CLI flags take precedence over environment variables when both are set.

## Workspace and CLI behavior

| Variable | Default | Description |
|----------|---------|-------------|
| `SMART_TRAIN_WORKSPACE` | current directory | Workspace root used by commands when `--workspace` is omitted. |
| `SMART_TRAIN_FORCE_NON_INTERACTIVE` | unset | When set to a truthy value, disables interactive prompts. |
| `SMART_TRAIN_INTERACTIVE_ALLOWED` | managed by CLI | Internal flag set by the Typer router while forwarding to argparse commands. |
| `SMART_TRAIN_SKIP_TORCH_POLICY` | unset | When set to `1`, skips CUDA/torch policy checks during external provider installation. |

## Unified run/model contract

| Variable | Default | Description |
|----------|---------|-------------|
| `SMARTTRAIN_UNIFIED_WRITE` | unset | When set to `1`, enables writing unified snapshots under `.smartrain/unified/`. |
| `SMARTTRAIN_CANONICAL_WRITE` | unset | Deprecated alias for `SMARTTRAIN_UNIFIED_WRITE`. |
| `SMARTTRAIN_UNIFIED_DUAL_WRITE_MODE` | `unified_only` | Dual-write mode: `unified_only`, `dual_write_strict`, or `dual_write_best_effort`. |
| `SMARTTRAIN_CANONICAL_DUAL_WRITE_MODE` | unset | Deprecated alias for `SMARTTRAIN_UNIFIED_DUAL_WRITE_MODE`. |
| `SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK` | unset | Emergency legacy read fallback during unified cutover (see migration docs). |

## Model test and ONNX runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `SMARTTRAIN_TEST_EVAL_SLOT` | unset | Selects the evaluation slot used by test artifact paths. |
| `SMARTTRAIN_ONNX_PROVIDER_POLICY` | `gpu_preferred` | ONNX runtime provider selection policy for native test runners. |
| `SMARTTRAIN_ONNX_IMGSZ_STRICT` | `0` | When truthy, enforces strict image-size matching for ONNX evaluation. |
| `SMARTTRAIN_ONNX_USE_SUBPROCESS` | `1` | When falsy, runs ONNX evaluation in-process instead of a subprocess worker. |

## Optional install extras

| Extra | Packages | Purpose |
|-------|----------|---------|
| `export` | `pypandoc-binary`, `weasyprint` | PDF/ODT export for dataset and analyze reports |
| `dev` | `pytest`, `ruff`, `mypy` | Development and CI tooling |
| `clearml` | `clearml` | ClearML experiment tracking |
| `sahi` | `sahi` | SAHI tiled inference CLI |

Install example: `pip install -e ".[dev,export]"`.

## Notes

- Two prefixes are used intentionally: `SMART_TRAIN_*` for workspace/CLI runtime and `SMARTTRAIN_*` for artifact contract and backend policy.
- Secrets for optional integrations (for example ClearML) should be provided through the corresponding third-party tooling or shell environment, not committed into the workspace.
