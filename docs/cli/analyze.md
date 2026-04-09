> Russian version: [../ru/cli/analyze.md](../ru/cli/analyze.md)

# CLI: run analysis

`smartrain analyze` operates on run artifact catalogs.

Default search root: directory `runs` inside the workspace (or path from `--models-root`).

## Subcommands

- `scan`
- `export-table`
- `compare`
- `interactive`
- `pr-curves`
- `inference-benchmark`
- `inference-plot`

## Examples

```bash
smartrain analyze scan
smartrain analyze export-table -o runs_summary.csv
smartrain analyze compare --baseline /path/to/run_a --others /path/to/run_b /path/to/run_c --out-csv cmp.csv
smartrain analyze pr-curves --run /path/to/run
smartrain analyze inference-benchmark --model /path/to/best.pt --source /path/to/images
smartrain analyze inference-plot --csv benchmark.csv --out benchmark.png
```

## Artifacts

- `export-table` generates a summary CSV for the found runs.
- `compare` can create a comparison table and PNG graphics.
- `inference-benchmark` generates a CSV with inference measurements.
- `inference-plot` builds a visualization based on the CSV from `inference-benchmark`.

`smartrain plot` remains a legacy wrapper and delegates to `analyze`.
